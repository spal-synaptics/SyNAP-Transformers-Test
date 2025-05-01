import argparse
import json
import os
from typing import Any, Final

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings.minilm import BaseEmbeddingsModel, minilm_factory
from .utils.export import ExportEmbeddings
from .utils.model import get_model_configs

QA_FILE: Final = "data/dishwasher/qa_dishwasher.json"
MODEL_CONFIGS_DIR: Final = "config"


class TextAgent:
    def __init__(
        self, 
        qa_file: str, 
        *, 
        embedding_models: list[BaseEmbeddingsModel],
        export_embeddings: bool = False,
    ):
        with open(qa_file, "r") as f:
            self.qa_pairs = json.load(f)
        self.embedding_models = embedding_models
        self.export_embeddings = export_embeddings
        self.qa_embeddings = {model.model_name: self.load_embeddings(model, self.export_embeddings) for model in self.embedding_models}

    def load_embeddings(self, model: BaseEmbeddingsModel, export_embeddings: bool = False) -> np.ndarray:
        if not isinstance(model, BaseEmbeddingsModel):
            raise RuntimeError(f"Unsupported embeddings model: {model}")
        texts = [pair["question"] + " " + pair["answer"] for pair in self.qa_pairs]
        embeddings = []
        for text in tqdm(texts, desc=f"Computing embeddings: {model}"):
            embeddings.append(model.generate(text))
        embeddings = np.array(embeddings)

        if export_embeddings:
            with ExportEmbeddings(self.qa_pairs, ["question", "answer"], model.export_dir) as exporter:
                exporter.save_embeddings(embeddings, model.model_name)

        return embeddings
    
    def embed_query(self, query: str, model: BaseEmbeddingsModel) -> tuple[int, np.ndarray]:
        query_emb = model.generate(query)
        sims = cosine_similarity([query_emb], self.qa_embeddings[model.model_name]).flatten()
        return np.argmax(sims), sims

    def answer_query(self, query: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for model in self.embedding_models:
            best_idx, sims = self.embed_query(query, model)
            results.append({
                "model": model.model_name,
                "answer": self.qa_pairs[best_idx]["answer"],
                "similarity": float(sims[best_idx]),
                "infer_time": model.last_infer_time,
            })
        return results
    

def compare_text_embeddings(qa_file: str, *, model_configs: list[str | os.PathLike], export_embeddings: bool = False):
    models = [minilm_factory(config) for config in model_configs]
    agent = TextAgent(qa_file, embedding_models=models, export_embeddings=export_embeddings)
    BOLD: Final = "\033[1m"
    YELLOW: Final = "\033[93m"
    GREEN: Final = "\033[32m"
    RESET: Final = "\033[0m"
    while True:
        query = input(YELLOW + "> " + RESET)
        if query.lower() in ("exit", "quit"):
            break
        results = agent.answer_query(query)
        for result in results:
            model, answer, similarity, infer_time = result["model"], result["answer"], result["similarity"], result["infer_time"]
            print(BOLD + f"{model:>{15}}: " + RESET + GREEN + answer + RESET)
            print(YELLOW + f"{' ':15}  [{infer_time * 1000:.3f} ms, Similarity: {similarity:.6f}]" + RESET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-file",
        type=str,
        default=QA_FILE,
        help="Path to Question-Answer pairs (default: %(default)s)"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=MODEL_CONFIGS_DIR,
        metavar="JSON | DIR",
        help="Path to JSON model config or directory containing multiple model configs (default: %(default)s)"
    )
    parser.add_argument(
        "--exclude-model",
        type=str,
        nargs="+",
        metavar="JSON",
        help="Configs of models to exclude"
    )
    parser.add_argument(
        "--export-embeddings",
        action="store_true",
        default=False,
        help="Export embeddings to TSV format"
    )
    args = parser.parse_args()

    model_configs = get_model_configs(args.model_config, ["gguf", "synap"], args.exclude_model)
    compare_text_embeddings(args.qa_file, model_configs=model_configs, export_embeddings=args.export_embeddings)
