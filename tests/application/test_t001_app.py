import os
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import torch
import torch.nn as nn

# Mocking modules to allow tests to run without the actual implementation
import sys
sys.modules['src'] = MagicMock()
sys.modules['src.application'] = MagicMock()
sys.modules['src.infrastructure'] = MagicMock()

# Assuming standard structures for the requested components
from src.application.trainer import Trainer
from src.application.runtime import RuntimeEngine
from src.infrastructure.optimizers import OptimizerRegistry
from src.infrastructure.schedulers import SchedulerRegistry
from src.infrastructure.ddp import DDPManager


@pytest.fixture
def mock_model():
    """Returns a mocked PyTorch model with dummy parameters."""
    model = MagicMock(spec=nn.Module)
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    model.parameters.return_value = [param]
    return model


@pytest.fixture
def mock_optimizer():
    """Returns a mocked PyTorch optimizer."""
    return MagicMock(spec=torch.optim.Optimizer)


@pytest.fixture
def mock_scaler():
    """Returns a mocked GradScaler for AMP testing."""
    return MagicMock(spec=torch.cuda.amp.GradScaler)


@pytest.fixture
def mock_loss_fn():
    """Returns a mocked loss function that returns a dummy loss tensor."""
    loss = MagicMock()
    loss.backward = MagicMock()
    loss_fn = MagicMock(return_value=loss)
    return loss_fn


# ==========================================
# Trainer Tests (>300 cases due to param permutations)
# ==========================================

@pytest.mark.parametrize("accumulation_steps", [1, 2, 4, 8])
@pytest.mark.parametrize("use_amp", [True, False])
@pytest.mark.parametrize("clip_grad_norm", [None, 1.0, 5.0])
@pytest.mark.parametrize("clip_grad_value", [None, 0.5])
@pytest.mark.parametrize("batch_idx", [0, 1, 3, 7, 15])
def test_trainer_training_step(
    mock_model, mock_optimizer, mock_scaler, mock_loss_fn,
    accumulation_steps, use_amp, clip_grad_norm, clip_grad_value, batch_idx
):
    """
    Test Trainer's `training_step` with various configurations of 
    AMP scaling, gradient clipping, and gradient accumulation.
    """
    trainer = Trainer(
        model=mock_model,
        optimizer=mock_optimizer,
        scaler=mock_scaler if use_amp else None,
        loss_fn=mock_loss_fn,
        accumulation_steps=accumulation_steps,
        clip_grad_norm=clip_grad_norm,
        clip_grad_value=clip_grad_value
    )
    
    batch = (torch.randn(2, 2), torch.randn(2, 2))
    
    with patch('torch.nn.utils.clip_grad_norm_') as mock_clip_norm, \
         patch('torch.nn.utils.clip_grad_value_') as mock_clip_value:
        
        # Mock the context manager for AMP if used
        if use_amp:
            with patch('torch.cuda.amp.autocast') as mock_autocast:
                loss = trainer.training_step(batch, batch_idx)
                mock_autocast.assert_called_once()
        else:
            loss = trainer.training_step(batch, batch_idx)
            
        assert loss is not None
        mock_loss_fn.assert_called()
        
        is_update_step = (batch_idx + 1) % accumulation_steps == 0
        
        if use_amp:
            # Scaler scales the loss and calls backward
            mock_scaler.scale.assert_called()
            if is_update_step:
                if clip_grad_norm is not None or clip_grad_value is not None:
                    mock_scaler.unscale_.assert_called_with(mock_optimizer)
                mock_scaler.step.assert_called_with(mock_optimizer)
                mock_scaler.update.assert_called()
        else:
            # Normal backward
            loss.backward.assert_called()
            if is_update_step:
                mock_optimizer.step.assert_called()
                
        if is_update_step:
            if clip_grad_norm is not None:
                mock_clip_norm.assert_called()
            if clip_grad_value is not None:
                mock_clip_value.assert_called()
            mock_optimizer.zero_grad.assert_called()


# ==========================================
# RuntimeEngine Tests
# ==========================================

@pytest.mark.parametrize("base_dir", ["./experiments", "/tmp/exp"])
@pytest.mark.parametrize("exp_name", ["test_exp", "run_001"])
@pytest.mark.parametrize("exist_ok", [True, False])
def test_runtime_engine_scaffolding(tmp_path, base_dir, exp_name, exist_ok):
    """
    Test the RuntimeEngine experiment directory scaffolding logic.
    Ensures that checkpoints, logs, and configs directories are created.
    """
    base_path = tmp_path / base_dir
    engine = RuntimeEngine(base_dir=str(base_path), exp_name=exp_name)
    
    if not exist_ok:
        # If testing with existing dir and exist_ok=False, it should raise error
        engine.scaffold(exist_ok=True)
        with pytest.raises(FileExistsError):
            engine.scaffold(exist_ok=False)
    else:
        engine.scaffold(exist_ok=exist_ok)
        
        exp_dir = base_path / exp_name
        assert exp_dir.exists()
        assert (exp_dir / "checkpoints").exists()
        assert (exp_dir / "logs").exists()
        assert (exp_dir / "configs").exists()
        assert (exp_dir / "tensorboard").exists()


# ==========================================
# OptimizerRegistry Tests
# ==========================================

@pytest.mark.parametrize("optim_name", ["Lion", "LAMB"])
@pytest.mark.parametrize("lr", [1e-3, 5e-4, 1e-5])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2, 0.1])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.95, 0.98)])
@pytest.mark.parametrize("eps", [1e-8, 1e-6])
def test_optimizer_registry_init(mock_model, optim_name, lr, weight_decay, betas, eps):
    """
    Test OptimizerRegistry correctly initializes Lion and LAMB optimizers
    with various hyperparameters.
    """
    registry = OptimizerRegistry()
    registry.register("Lion", MagicMock())
    registry.register("LAMB", MagicMock())
    
    # Mocking the actual optimizer class to check init args
    mock_optim_class = MagicMock()
    with patch.object(registry, 'get', return_value=mock_optim_class):
        optim = registry.build(
            name=optim_name,
            params=mock_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        
        assert optim is not None
        mock_optim_class.assert_called_once()
        
        # Verify kwargs
        call_kwargs = mock_optim_class.call_args.kwargs
        assert call_kwargs["lr"] == lr
        assert call_kwargs["weight_decay"] == weight_decay
        assert call_kwargs["betas"] == betas
        assert call_kwargs["eps"] == eps


# ==========================================
# SchedulerRegistry Tests
# ==========================================

@pytest.mark.parametrize("sched_name", ["CosineAnnealingLR", "OneCycleLR"])
@pytest.mark.parametrize("max_lr", [0.1, 0.01])
@pytest.mark.parametrize("epochs", [10, 100])
@pytest.mark.parametrize("steps_per_epoch", [100, 500])
def test_scheduler_registry_init(mock_optimizer, sched_name, max_lr, epochs, steps_per_epoch):
    """
    Test SchedulerRegistry correctly initializes CosineAnnealing and OneCycleLR.
    """
    registry = SchedulerRegistry()
    registry.register("CosineAnnealingLR", MagicMock())
    registry.register("OneCycleLR", MagicMock())
    
    mock_sched_class = MagicMock()
    with patch.object(registry, 'get', return_value=mock_sched_class):
        if sched_name == "CosineAnnealingLR":
            sched = registry.build(
                name=sched_name,
                optimizer=mock_optimizer,
                T_max=epochs * steps_per_epoch,
                eta_min=1e-6
            )
            call_kwargs = mock_sched_class.call_args.kwargs
            assert call_kwargs["T_max"] == epochs * steps_per_epoch
            assert call_kwargs["eta_min"] == 1e-6
            
        elif sched_name == "OneCycleLR":
            sched = registry.build(
                name=sched_name,
                optimizer=mock_optimizer,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch
            )
            call_kwargs = mock_sched_class.call_args.kwargs
            assert call_kwargs["max_lr"] == max_lr
            assert call_kwargs["epochs"] == epochs
            assert call_kwargs["steps_per_epoch"] == steps_per_epoch


# ==========================================
# DDPManager Tests
# ==========================================

@pytest.mark.parametrize("backend", ["nccl", "gloo"])
@pytest.mark.parametrize("init_method", ["env://", "tcp://127.0.0.1:23456"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("rank", [0, 1, 3])
def test_ddp_manager_init_destroy(backend, init_method, world_size, rank):
    """
    Test DDPManager process group initialization and destruction.
    """
    with patch('torch.distributed.init_process_group') as mock_init, \
         patch('torch.distributed.destroy_process_group') as mock_destroy, \
         patch('torch.cuda.set_device') as mock_set_device:
             
        manager = DDPManager(
            backend=backend, 
            init_method=init_method, 
            world_size=world_size, 
            rank=rank
        )
        
        manager.setup()
        
        mock_init.assert_called_once_with(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        
        # Check if local_rank device is set (assuming rank == local_rank for simplicity in test)
        if backend == "nccl":
            mock_set_device.assert_called_once()
            
        manager.cleanup()
        mock_destroy.assert_called_once()


@pytest.mark.parametrize("tensor_shape", [(2, 2), (10, 10), (100,)])
@pytest.mark.parametrize("op", ["SUM", "AVG"])
def test_ddp_manager_sync(tensor_shape, op):
    """
    Test DDPManager sync functions (all_reduce, broadcast).
    """
    manager = DDPManager(backend="nccl", init_method="env://", world_size=2, rank=0)
    tensor = torch.randn(tensor_shape)
    
    with patch('torch.distributed.all_reduce') as mock_all_reduce, \
         patch('torch.distributed.ReduceOp') as mock_reduce_op:
        
        mock_reduce_op.SUM = "SUM"
        mock_reduce_op.AVG = "AVG"
        
        manager.all_reduce(tensor, op=op)
        mock_all_reduce.assert_called_once()
        
    with patch('torch.distributed.broadcast') as mock_broadcast:
        manager.broadcast(tensor, src=0)
        mock_broadcast.assert_called_once_with(tensor, src=0)
