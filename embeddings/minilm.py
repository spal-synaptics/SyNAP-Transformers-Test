import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
from llama_cpp import Llama
from synap import Network
from transformers import AutoTokenizer


class BaseEmbeddings(ABC):

    def __init__(
        self,
        model_type: Literal["llama", "synap"],
        model_path: str | os.PathLike,
        token_len: int,
        normalize: bool,
        export_dir: str | os.PathLike | None
    ):
        if model_type not in ["llama", "synap"]:
            raise ValueError(f"Unsupported model type: '{model_type}")
        self.model_type = model_type
        self.model_path = Path(model_path)
        self.token_len = token_len
        self.normalize = normalize
        self.export_dir = Path(export_dir or f"export/{self.model_path.stem}")
        self.export_dir.mkdir(exist_ok=True, parents=True)

    def __repr__(self) -> str:
        return f"{self.model_type.capitalize()} ({self.model_path}, token_len: {self.token_len})"
    
    @property
    def name(self) -> str:
        return self.model_path.name
    
    @abstractmethod
    def generate(self, text: str) -> list[float]:
        ...


class EmbeddingsLlama(BaseEmbeddings):
    def __init__(
        self,
        model_path: str | os.PathLike,
        token_len: int,
        normalize: bool = False,
        n_threads: int | None = None,
        export_dir: str | os.PathLike | None = None
    ):
        super().__init__(
            "llama",
            model_path,
            token_len,
            normalize,
            export_dir
        )
        self.model = Llama(
            model_path=str(self.model_path),
            n_threads=n_threads,
            n_threads_batch=n_threads,
            n_ctx=self.token_len,
            embedding=True,
            verbose=False
        )

    def generate(self, text: str) -> list[float]:
        embedding = self.model.embed(text, normalize=self.normalize)
        if embedding is None:
            raise ValueError("No embedding returned")
        return embedding

    @classmethod
    def from_config(cls, config_json: str | os.PathLike) -> "EmbeddingsLlama":
        with open(config_json) as f:
            config: dict = json.load(f)
        return cls(
            config["model_path"], 
            config["token_len"], 
            config.get("n_threads"), 
            config.get("normalize", False)
        )


class EmbeddingsSynap(BaseEmbeddings):

    def __init__(
        self,
        model_path: str,
        token_len: int,
        hf_model: str,
        normalize: bool = False,
        export_dir: str | os.PathLike | None = None
    ):
        super().__init__(
            "synap",
            model_path,
            token_len,
            normalize,
            export_dir
        )
        self.model = Network(str(self.model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

    @staticmethod
    def mean_pooling(
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        mask_expanded = attention_mask[..., None]
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    @staticmethod
    def normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def _get_input_tokens(self, input: str) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(input, return_tensors="np", padding="max_length", truncation=True, max_length=self.token_len)
        tokens_np = {}
        model_inputs_info = {inp.name: inp.data_type.np_type() for inp in self.model.inputs}
        for inp_name, inp_tokens in tokens.items():
            dtype = model_inputs_info[inp_name]
            tokens_np[inp_name] = inp_tokens.astype(dtype)
        return tokens_np

    def generate(self, text: str) -> list[float]:
        tokens = self._get_input_tokens(text)
        attn_mask = tokens["attention_mask"]

        for inp in self.model.inputs:
            inp.assign(tokens[inp.name])
        model_outputs = self.model.predict()
        token_embeddings = model_outputs[1].to_numpy()
        embeddings = self.mean_pooling(token_embeddings, attn_mask)
        if self.normalize:
            embeddings = self.normalize(embeddings)

        return embeddings.squeeze(0).tolist()

    @classmethod
    def from_config(cls, config_json: str | os.PathLike) -> "EmbeddingsSynap":
        with open(config_json) as f:
            config: dict = json.load(f)
        return cls(
            config["model_path"], 
            config["token_len"], 
            config["hf_model"], 
            config.get("normalize", False)
        )


if __name__ == "__main__":
    pass