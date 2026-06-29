import os
import pytest
import numpy as np
from src.db_client import (
    init_db, save_patient, get_patient_record, log_audit_action,
    get_audit_logs, get_model_registry, compute_cache_key,
    get_cached_segmentation, cache_segmentation, DB_PATH
)

@pytest.fixture(autouse=True)
def setup_test_db():
    # If test DB exists, remove it to start clean
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except PermissionError:
            pass
    init_db()
    yield
    # Cleanup after test if possible
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except PermissionError:
            pass

def test_patient_upsert_and_fetch():
    save_patient("TEST_PATIENT_001", 45, "Male", "FLAIR")
    record = get_patient_record("TEST_PATIENT_001")
    
    assert record is not None
    assert record["patient_id"] == "TEST_PATIENT_001"
    assert record["age"] == 45
    assert record["gender"] == "Male"
    assert record["modality"] == "FLAIR"

def test_audit_logs():
    log_audit_action("TEST_ACTION", "TEST_PATIENT_001", "Details of test action")
    logs = get_audit_logs(limit=5)
    
    assert len(logs) > 0
    test_log = [l for l in logs if l["user_action"] == "TEST_ACTION"]
    assert len(test_log) == 1
    assert test_log[0]["patient_id"] == "TEST_PATIENT_001"
    assert test_log[0]["details"] == "Details of test action"

def test_model_registry_seeding():
    registry = get_model_registry()
    assert len(registry) > 0
    models_names = [m["model_name"] for m in registry]
    assert "U-Net" in models_names
    assert "Attention U-Net" in models_names

def test_segmentation_caching():
    dummy_img = np.random.rand(10, 10)
    dummy_mask = np.ones((10, 10))
    key = compute_cache_key(dummy_img, "U-Net")
    
    # Assert initially uncached
    assert get_cached_segmentation(key) is None
    
    # Cache and retrieve
    cache_segmentation(key, dummy_mask)
    retrieved_mask = get_cached_segmentation(key)
    
    assert retrieved_mask is not None
    assert np.array_equal(retrieved_mask, dummy_mask)
