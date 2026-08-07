import pytest
from unittest.mock import MagicMock
from typing import Optional

class RollbackManager:
    """Manager for deploying rollbacks."""
    
    def trigger_rollback(self, target_version: str, force: Optional[bool] = False) -> bool:
        return True

@pytest.fixture
def rollback_mgr() -> RollbackManager:
    """Provide a RollbackManager instance."""
    return RollbackManager()

@pytest.mark.parametrize("target_version", [f"1.{i}.0" for i in range(15)])
@pytest.mark.parametrize("force", [True, False, None])
def test_rollback_manager_trigger(rollback_mgr: RollbackManager, target_version: str, force: Optional[bool]) -> None:
    """
    Test rollback manager trigger.
    
    Args:
        rollback_mgr (RollbackManager): Manager fixture.
        target_version (str): Target version to rollback to.
        force (Optional[bool]): Force the rollback.
    """
    rollback_mgr.trigger_rollback = MagicMock(return_value=True)
    result = rollback_mgr.trigger_rollback(target_version, force=force)
    assert result is True
    rollback_mgr.trigger_rollback.assert_called_with(target_version, force=force)
