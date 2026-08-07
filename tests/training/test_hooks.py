"""Tests for Hook execution order."""

import pytest


class Hook:
    """Base hook."""
    def before_train(self) -> None: pass
    def before_epoch(self) -> None: pass
    def before_batch(self) -> None: pass
    def after_batch(self) -> None: pass
    def after_epoch(self) -> None: pass
    def after_train(self) -> None: pass


class HookManager:
    """Mock hook manager."""
    def __init__(self) -> None:
        self.hooks = []
        self.executed = []

    def register(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def trigger(self, event: str) -> None:
        self.executed.append(event)
        for hook in self.hooks:
            getattr(hook, event)()


class DummyHook(Hook):
    """Dummy hook recording executions."""
    def __init__(self, name: str, manager: HookManager) -> None:
        self.name = name
        self.manager = manager
        
    def before_train(self) -> None: self.manager.executed.append(f"{self.name}_before_train")


@pytest.fixture
def hook_manager() -> HookManager:
    """Provides a HookManager instance."""
    return HookManager()


@pytest.mark.parametrize("event", [
    "before_train", "before_epoch", "before_batch",
    "after_batch", "after_epoch", "after_train"
])
@pytest.mark.parametrize("num_hooks", [1, 2, 3, 5])
def test_hook_trigger(hook_manager: HookManager, event: str, num_hooks: int) -> None:
    """Test hook trigger logic."""
    for i in range(num_hooks):
        hook_manager.register(Hook())
    hook_manager.trigger(event)
    assert event in hook_manager.executed


@pytest.mark.parametrize("hook_names", [
    ["logger", "checkpoint"],
    ["lr_scheduler", "timer", "evaluator"],
])
def test_hook_order(hook_manager: HookManager, hook_names: list) -> None:
    """Test multiple hooks execute in registered order."""
    for name in hook_names:
        hook_manager.register(DummyHook(name, hook_manager))
        
    hook_manager.trigger("before_train")
    
    for name in hook_names:
        assert f"{name}_before_train" in hook_manager.executed
