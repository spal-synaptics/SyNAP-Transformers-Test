import os
from pathlib import Path


def get_model_configs(config_loc: str | os.PathLike, model_types: list[str]) -> list[Path]:
    config_loc = Path(config_loc).resolve()
    model_configs: list[Path] = []
    if config_loc.is_dir():
        for config in config_loc.glob("*.json"):
            for model_type in model_types:
                if model_type in config.name:
                    model_configs.append(config)
    else:
        model_configs.append(config_loc)
    return sorted(model_configs)


if __name__ == "__main__":
    print(get_model_configs("config", ["gguf", "synap"]))
    print(get_model_configs("config/all-MiniLM-L6-v2-qdq.synap.json", ["gguf", "synap"]))
    print(get_model_configs("config/all-MiniLM-L6-v2-Q8_0.gguf.json", ["gguf", "synap"]))
