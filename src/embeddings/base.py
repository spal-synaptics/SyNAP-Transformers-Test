import json
import os
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path


class BaseEmbeddingsModel(ABC):

    def __init__(
        self,
        model_name: str,
        model_path: str | os.PathLike,
        token_len: int,
        normalize: bool,
        export_dir: str | os.PathLike | None
    ):
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.token_len = token_len
        self.normalize = normalize
        self.export_dir = Path(export_dir or f"export/{self.model_path.stem}")
        self.export_dir.mkdir(exist_ok=True, parents=True)

        self._infer_times = deque(maxlen=100)

    def __repr__(self) -> str:
        return f"{self.model_name} ({self.model_path}, token_len: {self.token_len})"
    
    @property
    def last_infer_time(self) -> float | None:
        return self._infer_times[-1] if self._infer_times else None
    
    @property
    def avg_infer_time(self) -> float | None:
        return (sum(self._infer_times) / self.n_infer) if self._infer_times else None
    
    @property
    def n_infer(self) -> int:
        return len(self._infer_times)
    
    @abstractmethod
    def generate(self, text: str) -> list[float]:
        ...

    @classmethod
    def from_config(cls, config: dict | str | os.PathLike) -> "BaseEmbeddingsModel":
        if not isinstance(config, dict):
            with open(config) as f:
                model_config = json.load(f)
        else:
            model_config = config
        return cls(**model_config)
