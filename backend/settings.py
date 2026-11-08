"""Environment configuration for serving."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from shrimp_forecast.data import ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHRIMP_", env_file=".env", extra="ignore")
    artifact_dir: Path = ROOT / "artifacts"
    upload_dir: Path = ROOT / "data/uploads"
    admin_token: SecretStr | None = None
    max_upload_bytes: int = 2_000_000
    max_upload_rows: int = 50_000
    allowed_hosts: list[str] = ["127.0.0.1", "localhost", "testserver"]
