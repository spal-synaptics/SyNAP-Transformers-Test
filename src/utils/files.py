import os
from pathlib import Path


def get_model_configs(config_loc: str | os.PathLike, model_types: list[str], exclude: list[str, os.PathLike] = None) -> list[Path]:
    config_loc = Path(config_loc).resolve()
    model_configs: list[Path] = []
    exclude: list[Path] = [Path(f).resolve() for f in exclude] if exclude else []
    if config_loc.is_dir():
        for config in config_loc.glob("*.json"):
            if config in exclude:
                continue
            for model_type in model_types:
                if model_type in config.name:
                    model_configs.append(config)
    else:
        if config_loc not in exclude:
            model_configs.append(config_loc)
    return sorted(model_configs)


if __name__ == "__main__":
    pass
