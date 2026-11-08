import json

import pytest

from shrimp_forecast.registry import Registry, publish


def test_unpromoted_run_does_not_change_serving(tmp_path, bundle):
    run_id = publish(bundle, tmp_path)
    assert (tmp_path / "farmgate" / run_id / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        Registry(tmp_path).load("farmgate")


def test_cache_and_atomic_promotion(tmp_path, bundle):
    first = publish(bundle, tmp_path, promote=True)
    registry = Registry(tmp_path)
    assert registry.load("farmgate") is registry.load("farmgate")
    second = publish(bundle, tmp_path, promote=True)
    assert first != second
    assert registry.load("farmgate")["run_id"] == second
    assert (tmp_path / "farmgate" / first / "model.joblib").exists()


def test_corrupted_artifact_is_rejected_before_deserialization(tmp_path, bundle):
    run_id = publish(bundle, tmp_path, promote=True)
    path = tmp_path / "farmgate" / run_id / "model.joblib"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        Registry(tmp_path).load("farmgate")


def test_dependency_mismatch_requires_retraining(tmp_path, bundle):
    run_id = publish(bundle, tmp_path, promote=True)
    path = tmp_path / "farmgate" / run_id / "manifest.json"
    metadata = json.loads(path.read_text())
    metadata["environment"]["packages"]["scikit-learn"] = "0.0"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="dependency mismatch"):
        Registry(tmp_path).load("farmgate")


def test_pointer_path_traversal_rejected(tmp_path):
    (tmp_path / "farmgate").mkdir()
    (tmp_path / "farmgate/CURRENT.json").write_text('{"run_id":"../../anything"}')
    with pytest.raises(ValueError, match="identifier"):
        Registry(tmp_path).load("farmgate")


def test_source_mismatch_requires_retraining(tmp_path, bundle):
    run_id = publish(bundle, tmp_path, promote=True)
    path = tmp_path / "farmgate" / run_id / "manifest.json"
    metadata = json.loads(path.read_text())
    metadata["environment"]["source_sha256"] = "stale"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source mismatch"):
        Registry(tmp_path).load("farmgate")
