import argparse
import json
from typing import Final

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from embeddings.minilm import EmbeddingsLlama, EmbeddingsSynap

from utils.export import ExportEmbeddings

QA_FILE: Final = "data/dishwasher/qa_dishwasher.json"
LLAMA_CONFIG: Final = "config/MiniLM-L6-v2.llama.json"
SYNAP_CONFIG: Final = "config/MiniLM-L6-v2.synap.json"


class TextAgent:
    def __init__(
        self, 
        qa_file: str, 
        *, 
        embed_llama: EmbeddingsLlama, 
        embed_synap: EmbeddingsSynap,
        embeddings_export_dir: str | None = None,
    ):
        with open(qa_file, "r") as f:
            self.qa_pairs = json.load(f)
        self.embed_llama = embed_llama
        self.embed_synap = embed_synap
        self.qa_embeds_llama = self.load_embeddings(self.embed_llama)
        self.qa_embeds_synap = self.load_embeddings(self.embed_synap)
        if embeddings_export_dir:
            exporter = ExportEmbeddings(self.qa_pairs, embeddings_export_dir)
            exporter.save_embeddings(self.qa_embeds_llama, "llama")
            exporter.save_embeddings(self.qa_embeds_synap, "synap")
            exporter.gen_metadata(["question", "answer"])

    def load_embeddings(self, model: EmbeddingsLlama | EmbeddingsSynap) -> np.ndarray:
        if model is not self.embed_llama and model is not self.embed_synap:
            raise RuntimeError(f"Unsupported embeddings model: {model}")
        texts = [pair["question"] + " " + pair["answer"] for pair in self.qa_pairs]
        embeddings = []
        for text in tqdm(texts, desc=f"Computing embeddings ({model})"):
            embeddings.append(model.generate(text))
        return np.array(embeddings)
    
    def embed_query(self, query: str, model: EmbeddingsLlama | EmbeddingsSynap) -> tuple[int, np.ndarray]:
        if model is self.embed_llama:
            qa_embeds = self.qa_embeds_llama
        elif model is self.embed_synap:
            qa_embeds = self.qa_embeds_synap
        else:
            raise RuntimeError(f"Unsupported embeddings model: {model}")
        query_emb = model.generate(query)
        sims = cosine_similarity([query_emb], qa_embeds).flatten()
        return np.argmax(sims), sims

    def answer_query(self, query: str):
        best_idx_llama, sims_llama = self.embed_query(query, self.embed_llama)
        best_idx_synap, sims_synap = self.embed_query(query, self.embed_synap)

        return {
            "answer_llama": self.qa_pairs[best_idx_llama]["answer"],
            "similarity_llama": float(sims_llama[best_idx_llama]),
            "answer_synap": self.qa_pairs[best_idx_synap]["answer"],
            "similarity_synap": float(sims_synap[best_idx_synap])
        }


def compare_text_embeddings(qa_file: str, *, llama_config: str, synap_config: str, export_dir: str | None = None):
    llama_model = EmbeddingsLlama.from_config(llama_config)
    synap_model = EmbeddingsSynap.from_config(synap_config)
    agent = TextAgent(qa_file, embed_llama=llama_model, embed_synap=synap_model, embeddings_export_dir=export_dir)

    YELLOW: Final = "\033[93m"
    RESET: Final = "\033[0m"
    while True:
        query = input(YELLOW + "> " + RESET)
        if query.lower() in ("exit", "quit"):
            break
        result = agent.answer_query(query)
        print(YELLOW + "Answer (Llama): " + result["answer_llama"] + RESET)
        print(YELLOW + "Similarity (Llama): " + str(result["similarity_llama"]) + RESET)
        print(YELLOW + "Answer (SyNAP): " + result["answer_synap"] + RESET)
        print(YELLOW + "Similarity (SyNAP): " + str(result["similarity_synap"]) + RESET)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-file",
        type=str,
        default=QA_FILE,
        help="Path to Question-Answer pairs (default: %(default)s)"
    )
    parser.add_argument(
        "--llama-config",
        type=str,
        default=LLAMA_CONFIG,
        help="Path to Llama model config (default: %(default)s)"
    )
    parser.add_argument(
        "--synap-config",
        type=str,
        default=SYNAP_CONFIG,
        help="Path to SyNAP model config (default: %(default)s)"
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        metavar="DIR",
        help="Directory to export QA embeddings in TSV format"
    )
    args = parser.parse_args()

    compare_text_embeddings(args.qa_file, llama_config=args.llama_config, synap_config=args.synap_config, export_dir=args.export_dir)
