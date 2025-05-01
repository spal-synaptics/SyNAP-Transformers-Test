import json
import os
import zipfile
from pathlib import Path
from typing import Any

from synap.types import DataType, Dimensions, Layout, Shape

_MODEL_META_FILE = "0/model.json"

_data_types = {
    "u8": DataType.uint8,
    "uint8": DataType.uint8,
    "quin8": DataType.uint8,
    "i8": DataType.int8,
    "int8": DataType.int8,
    "qint8": DataType.int8,
    "u16": DataType.uint16,
    "uint16": DataType.uint16,
    "quint16": DataType.uint16,
    "i16": DataType.int16,
    "int16": DataType.int16,
    "qint16": DataType.int16,
    "u32": DataType.uint32,
    "uint32": DataType.uint32,
    "quint32": DataType.uint32,
    "i32": DataType.int32,
    "int32": DataType.int32,
    "qint32": DataType.int32,
    "f16": DataType.float16,
    "float16": DataType.float16,
    "fp16": DataType.float16,
    "f32": DataType.float32,
    "float32": DataType.float32,
    "fp32": DataType.float32,
}


def _parse_tensor_info(tensor_info: dict) -> dict:
    parsed_info = {}
    if quant_info := tensor_info.get("quantize"):
        parsed_info["data_type"] = _data_types[quant_info["qtype"]]
        parsed_info["quant_info"] = quant_info
    else:
        parsed_info["data_type"] = _data_types[tensor_info["dtype"]]
    parsed_info["format"] = tensor_info["data_format"]
    layout = tensor_info["format"]
    if layout == "nhwc":
        parsed_info["layout"] = Layout.nhwc
    elif layout == "nchw":
        parsed_info["layout"] = Layout.nchw
    else:
        parsed_info["layout"] = Layout.none
    parsed_info["name"] = tensor_info.get("name", "")
    parsed_info["shape"] = Shape(tensor_info["shape"])
    parsed_info["dimensions"] = Dimensions(parsed_info["shape"], parsed_info["layout"])
    return parsed_info


def get_model_configs(config_loc: str | os.PathLike, model_types: list[str], exclude: list[str, os.PathLike] = None) -> list[Path]:
    config_loc = Path(config_loc).resolve()
    model_configs: list[Path] = []
    exclude: list[Path] = [Path(f).resolve() for f in exclude] if exclude else []
    if config_loc.is_dir():
        for config in config_loc.glob("*.json"):
            if config in exclude:
                continue
            for model_type in model_types:
                if model_type in config.name:
                    model_configs.append(config)
    else:
        if config_loc not in exclude:
            model_configs.append(config_loc)
    return sorted(model_configs)


def get_model_metadata(model: str) -> dict[str, Any]:
    try:
        model_metadata: dict[str, list] = {"inputs": [], "outputs": []}
        with zipfile.ZipFile(model, "r") as mod_info:
            if _MODEL_META_FILE not in mod_info.namelist():
                raise FileNotFoundError("Missing model metadata")
            with mod_info.open(_MODEL_META_FILE, "r") as meta_f:
                metadata = json.load(meta_f)
                inputs: dict = metadata["Inputs"]
                for inp in inputs.values():
                    model_metadata["inputs"].append(_parse_tensor_info(inp))
                outputs: dict = metadata["Outputs"]
                for out in outputs.values():
                    model_metadata["outputs"].append(_parse_tensor_info(out))
                return model_metadata
    except (zipfile.BadZipFile, FileNotFoundError) as e:
        raise RuntimeError(f"Error: Invalid SyNAP model '{model}': {e.args[0]}")
    except KeyError as e:
        raise RuntimeError(f"Error: Missing model metadata '{e.args[0]}' for SyNAP model '{model}'")
    except (NotImplementedError, ValueError) as e:
        raise RuntimeError(f"Error: Invalid SyNAP model '{model}': {e.args[0]}")


if __name__ == "__main__":
    pass
