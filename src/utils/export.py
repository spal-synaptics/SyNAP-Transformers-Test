import os
from pathlib import Path

import numpy as np


class ExportEmbeddings:

    def __init__(
        self, 
        text_data: list[dict], 
        column_labels: list[str], 
        export_dir: str | os.PathLike, 
    ):
        self.text_data = text_data
        self.column_labels = column_labels
        self.n_records = len(self.text_data)
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_generated: bool = False

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.gen_metadata()

    def save_embeddings(self, embeddings: np.ndarray, embeddings_name: str, delim: str = "\t"):
        np.savetxt(self.export_dir / f"embeddings-{embeddings_name}.tsv", embeddings, delimiter=delim)

    def gen_metadata(self):
        if not self.metadata_generated:
            with open(self.export_dir / "embeddings-metadata.tsv", "w") as f:
                f.write("\t".join(self.column_labels) + "\n")
                for record in self.text_data:
                    samples = list(str(r) for r in record.values())
                    f.write("\t".join(samples) + "\n")
            self.metadata_generated = True
