import pytest
import os
import json

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """Setup test environment with required directories and files."""
    # Create data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create log files
    (tmp_path / "dataset_creation.log").touch()
    (tmp_path / "owl_processing.log").touch()
    
    # Create token cost file
    token_cost = {
        "total_eval_count": 0,
        "total_eval_duration_ns": 0,
        "tasks": {}
    }
    with open(tmp_path / "token_cost.json", "w") as f:
        json.dump(token_cost, f)
    
    # Set environment variables
    os.environ["TEST_MODE"] = "true"
    
    yield
    
    # Cleanup
    os.environ.pop("TEST_MODE", None)
