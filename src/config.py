from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "model": {
        "contamination": 0.05,
        "random_state": 42,
    }
}


def load_config(project_root: Path) -> dict:
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        return DEFAULT_CONFIG

    with config_path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    model_cfg = config_data.get("model", {})
    merged = {
        "model": {
            "contamination": model_cfg.get("contamination", DEFAULT_CONFIG["model"]["contamination"]),
            "random_state": model_cfg.get("random_state", DEFAULT_CONFIG["model"]["random_state"]),
        }
    }
    return merged
