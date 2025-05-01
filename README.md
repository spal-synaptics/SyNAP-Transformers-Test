# Testing Transformers on SyNAP NPU

## Supported Tasks and Models
### Text Embeddings
* MiniLM: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Testing Models
### Text Embeddings
#### Run
```bash
python -m src.text
```

By default, all models specified in [config](config/) will be loaded. Models can be excluded with `--exclude path/to/model_config.json`

Other options, run `-h` to see defaults:
* `--qa-file`: Path to Question-Answer pairs
* `--model-config`: Path to JSON model config or directory containing multiple model configs
* `--export-embeddings`: Export embeddings to TSV format, useful for visualizing with tools like [TensorFlow Projector](https://projector.tensorflow.org/)

#### Profile
```bash
python -m src.profile_text -m path/to/model
```
The model path must be a .synap model or a .gguf model.

Other options, run `-h` to see defaults:
* `--hf-model`: Hugging Face model identifier (required for SyNAP models)
* `-r <N> / --repeat <N>`: Number of iterations to repeat inference
* `-i <text> / --input <text>`: Input text to use for inference
* `-j <N> / --threads <N>`: Number of cores to use for CPU execution (only applies to GGUF)