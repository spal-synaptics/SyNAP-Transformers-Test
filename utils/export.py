import os
import json
from pathlib import Path

import numpy as np


class ExportEmbeddings:

    def __init__(
        self, 
        text_data: list[dict],
        export_dir: str | os.PathLike, 
    ):
        self.text_data = text_data
        self.n_records = len(self.text_data)
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True, parents=True)

    def save_embeddings(self, embeddings: np.ndarray, embeddings_name: str, delim: str = "\t"):
        np.savetxt(self.export_dir / f"embeddings-{embeddings_name}.tsv", embeddings, delimiter=delim)

    def gen_metadata(self, column_labels: list[str]):
        with open(self.export_dir / "embeddings-metadata", "w") as f:
            f.write("\t".join(column_labels) + "\n")
            for record in self.text_data:
                samples = list(str(r) for r in record.values())
                f.write("\t".join(samples) + "\n")

    # def update_config(self, tensor_names: list[str], embed_len: int,):
    #     with open(self.export_dir / f"projector-config.json") as f:
    #         config = json.load(f)
    #     for tensor in tensor_names:
    #         config["embeddings"].update({
    #             "tensorName": tensor,
    #             "tensorShape": [self.n_records, embed_len],
    #             "tensorPath": 
    #         })
