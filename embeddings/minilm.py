import json
import os
import sys

import numpy as np
from llama_cpp import Llama
from synap import Network
from transformers import AutoTokenizer


class EmbeddingsLlama:
    def __init__(
        self,
        model_path: str | os.PathLike,
        token_len: int,
        normalize: bool = False,
        n_threads: int | None = None,
    ):
        self.token_len = token_len
        self.model_path = model_path
        self.normalize = normalize
        self.llm = Llama(
            model_path=self.model_path,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            n_ctx=self.token_len,
            embedding=True,
            verbose=False
        )

    def __repr__(self) -> str:
        return f"Llama ({self.model_path}, token_len: {self.token_len})"

    def generate(self, text: str) -> list[float]:
        embedding = self.llm.embed(text, normalize=self.normalize)
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


class EmbeddingsSynap:

    def __init__(
        self,
        model_path: str,
        token_len: int,
        hf_model: str,
        normalize: bool = False,
    ):
        self.model_path = model_path
        self.model = Network(self.model_path)
        self.token_len = token_len
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.normalize = normalize

    def __repr__(self) -> str:
        return f"SyNAP ({self.model_path}, token_len: {self.token_len})"

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