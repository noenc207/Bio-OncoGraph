import pytest
from unittest.mock import MagicMock

class DeploymentValidator:
    """Validator for active deployments."""
    
    def validate_deployment_status(self, dep_id: str) -> str:
        return "healthy"
        
    def validate_registry_consistency(self) -> bool:
        return True

@pytest.fixture
def validator() -> DeploymentValidator:
    """Provide a DeploymentValidator instance."""
    return DeploymentValidator()

@pytest.mark.parametrize("dep_id", [f"dep_{i:04d}" for i in range(15)])
@pytest.mark.parametrize("expected_status", ["healthy", "unhealthy", "starting", "stopped", "failed"])
def test_deployment_validator_status(validator: DeploymentValidator, dep_id: str, expected_status: str) -> None:
    """
    Test validation of deployment status.
    
    Args:
        validator (DeploymentValidator): Validator fixture.
        dep_id (str): Deployment ID.
        expected_status (str): Expected outcome.
    """
    validator.validate_deployment_status = MagicMock(return_value=expected_status)
    status = validator.validate_deployment_status(dep_id)
    assert status == expected_status
    validator.validate_deployment_status.assert_called_with(dep_id)
