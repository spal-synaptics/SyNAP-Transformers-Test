import argparse
from pathlib import Path

from .embeddings.minilm import MiniLMLlama, MiniLMSynap

SAMPLE_INPUT = "Although recent advancements in artificial intelligence have significantly improved natural language understanding, challenges remain in ensuring models grasp contextual nuance, especially when processing complex, multi-clause sentences like this one."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to SyNAP or GGUF model"
    )
    parser.add_argument(
        "-r", "--repeat",
        type=int,
        default=100,
        help="Number of iterations to repeat inference (default: %(default)s)"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=SAMPLE_INPUT,
        help="Input text for inference (default: \"%(default)s)\""
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        help="Hugging Face model identifier (required for SyNAP models)"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if model_path.suffix == ".gguf":
        model = MiniLMLlama(
            model_name=model_path.stem, 
            model_path=str(model_path), 
            n_threads=args.threads
        )
    elif model_path.suffix == ".synap":
        if not args.hf_model:
            raise ValueError("Hugging Face model ID required for SyNAP model tokenizer")
        model = MiniLMSynap(
            model_name=model_path.stem,
            model_path=str(model_path),
            hf_model=args.hf_model
        )

    for _ in range(args.repeat):
        model.generate(args.input)

    print(f"Total inference time ({model.n_infer} iters): {model.total_infer_time * 1000:.4f} ms")
    print(f"Average inference time: {model.avg_infer_time * 1000:.4f} ms")
