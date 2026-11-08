import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings
from shrimp_forecast.data import validate_observations
from shrimp_forecast.engine import TrainConfig, train
from shrimp_forecast.registry import publish


@pytest.fixture(scope="session")
def panel():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(12):
        value = 4 + i / 2
        for date in pd.date_range("2024-01-01", periods=24, freq="W-MON"):
            value *= np.exp(rng.normal(0.001, 0.02))
            rows.append(
                {
                    "date": date,
                    "available_at": date,
                    "series_id": f"in:vannamei:{40 + i}",
                    "country": "India",
                    "species": "vannamei",
                    "size": str(40 + i),
                    "price_usd_kg": value,
                }
            )
    return validate_observations(pd.DataFrame(rows), "farmgate")


@pytest.fixture(scope="session")
def bundle(panel):
    return train(panel, "farmgate", TrainConfig(max_validation_origins=2))


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory, bundle):
    root = tmp_path_factory.mktemp("artifacts")
    publish(bundle, root, promote=True)
    return root


@pytest.fixture()
def client(artifacts, tmp_path):
    settings = Settings(
        artifact_dir=artifacts, upload_dir=tmp_path / "uploads", admin_token="unit-test-token"
    )
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture()
def source_csv(panel):
    from shrimp_forecast.data import canonical_bytes

    return canonical_bytes(panel)
