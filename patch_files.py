import re
import os
from pathlib import Path

# --- Patch trainer.py ---
trainer_path = r"D:\Bio-OncoGraph\bio_oncograph\application\trainer.py"
with open(trainer_path, "r", encoding="utf-8") as f:
    trainer_code = f.read()

# Add contextlib import if not present
if "import contextlib" not in trainer_code:
    trainer_code = trainer_code.replace("import logging", "import logging\nimport contextlib")

# Update Trainer.__init__ signature
init_old = '''    def __init__(
        self,
        model: BaseModel,
        loss_fn: BaseLoss,
        optimizer: Any,
        event_bus: EventBus,
        train_loader: Iterable[Any],
        val_loader: Optional[Iterable[Any]] = None,
        test_loader: Optional[Iterable[Any]] = None,
        metrics: Optional[IMetrics] = None,
        checkpoint: Optional[ICheckpoint] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        max_epochs: int = 100,
        gradient_clip_max_norm: float = 0.0,
    ) -> None:'''

init_new = '''    def __init__(
        self,
        model: BaseModel,
        loss_fn: BaseLoss,
        optimizer: Any,
        event_bus: EventBus,
        train_loader: Iterable[Any],
        val_loader: Optional[Iterable[Any]] = None,
        test_loader: Optional[Iterable[Any]] = None,
        metrics: Optional[IMetrics] = None,
        checkpoint: Optional[ICheckpoint] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        max_epochs: int = 100,
        gradient_clip_max_norm: float = 0.0,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        sync_ddp: bool = False,
    ) -> None:'''
trainer_code = trainer_code.replace(init_old, init_new)

# Update Trainer.__init__ body
init_body_old = '''        self._max_epochs = max_epochs
        self._gradient_clip_max_norm = gradient_clip_max_norm

        # Mutable state'''
init_body_new = '''        self._max_epochs = max_epochs
        self._gradient_clip_max_norm = gradient_clip_max_norm
        self._gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self._use_amp = use_amp
        self._sync_ddp = sync_ddp
        
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        # Mutable state'''
trainer_code = trainer_code.replace(init_body_old, init_body_new)

# Update train_epoch
train_epoch_old = '''    def train_epoch(self) -> Dict[str, float]:
        \"\"\"
        Execute a single training epoch.

        Returns:
            Dictionary with at least `train_loss`.
        \"\"\"
        self._model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self._train_loader:
            self._optimizer.zero_grad()
            output = self._model(batch)
            loss_output = self._loss_fn(output, batch.survival_target)
            loss_output.total_loss.backward()

            if self._gradient_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self._gradient_clip_max_norm,
                )

            self._optimizer.step()
            total_loss += loss_output.total_loss.item()
            num_batches += 1
            self._global_step += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}'''

train_epoch_new = '''    def train_epoch(self) -> Dict[str, float]:
        \"\"\"
        Execute a single training epoch with AMP, gradient accumulation, and NaN handling.

        Returns:
            Dictionary with at least `train_loss`.
        \"\"\"
        self._model.train()
        total_loss = 0.0
        num_batches = 0
        self._optimizer.zero_grad()

        for i, batch in enumerate(self._train_loader):
            # Handle DDP synchronization logic for gradient accumulation
            is_accumulating = (i + 1) % self._gradient_accumulation_steps != 0 and (i + 1) != len(self._train_loader)
            if self._sync_ddp and is_accumulating and hasattr(self._model, "no_sync"):
                sync_context = self._model.no_sync()
            else:
                sync_context = contextlib.nullcontext()

            with sync_context:
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    output = self._model(batch)
                    loss_output = self._loss_fn(output, batch.survival_target)
                    loss = loss_output.total_loss / self._gradient_accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss encountered, skipping batch.")
                    continue

                self._scaler.scale(loss).backward()

            if not is_accumulating:
                if self._gradient_clip_max_norm > 0:
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        self._gradient_clip_max_norm,
                    )
                
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
                self._global_step += 1

            total_loss += loss_output.total_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}'''
trainer_code = trainer_code.replace(train_epoch_old, train_epoch_new)

with open(trainer_path, "w", encoding="utf-8") as f:
    f.write(trainer_code)

# --- Patch runtime.py ---
runtime_path = r"D:\Bio-OncoGraph\bio_oncograph\application\runtime.py"
with open(runtime_path, "r", encoding="utf-8") as f:
    runtime_code = f.read()

import_os_path = "import os\nfrom pathlib import Path\n"
if "import os" not in runtime_code:
    runtime_code = runtime_code.replace("import logging", f"import logging\n{import_os_path}")

init_old_rt = '''    def initialize(self, trainer: Trainer) -> None:
        \"\"\"
        Initialise the runtime with a trainer.

        Transitions: `CREATED → INITIALIZED`,
        `STOPPED → INITIALIZED`, `FINISHED → INITIALIZED`.

        Args:
            trainer: A configured `Trainer` instance.

        Raises:
            InvalidStateTransitionError: If the current state does
                not allow transitioning to `INITIALIZED`.
        \"\"\"
        self._transition(RuntimeState.INITIALIZED)
        self._trainer = trainer
        logger.info("Runtime initialized")'''

init_new_rt = '''    def initialize(self, trainer: Trainer, experiment_name: str = "default_experiment", base_dir: str = "runs") -> None:
        \"\"\"
        Initialise the runtime with a trainer and create experiment folder structure.

        Transitions: `CREATED → INITIALIZED`,
        `STOPPED → INITIALIZED`, `FINISHED → INITIALIZED`.

        Args:
            trainer: A configured `Trainer` instance.
            experiment_name: Name of the experiment.
            base_dir: Base directory for runs.

        Raises:
            InvalidStateTransitionError: If the current state does
                not allow transitioning to `INITIALIZED`.
        \"\"\"
        self._transition(RuntimeState.INITIALIZED)
        self._trainer = trainer
        
        # Create standard experiment folder structure
        exp_dir = Path(base_dir) / experiment_name
        self._setup_experiment_folders(exp_dir)
        self._setup_logging(exp_dir)
        
        logger.info(f"Runtime initialized for experiment: {experiment_name}")

    def _setup_experiment_folders(self, exp_dir: Path) -> None:
        (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (exp_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
        (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        # Touch a dummy config.yaml if it doesn't exist
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            config_path.write_text("# Auto-generated config\\n")

    def _setup_logging(self, exp_dir: Path) -> None:
        log_file = exp_dir / "logs" / "runtime.log"
        file_handler = logging.FileHandler(str(log_file))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
'''
runtime_code = runtime_code.replace(init_old_rt, init_new_rt)

with open(runtime_path, "w", encoding="utf-8") as f:
    f.write(runtime_code)

print("Patched trainer.py and runtime.py")
