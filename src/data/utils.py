from pathlib import Path


def is_json_file(file_path: Path):
    if file_path.suffix == ".json":
        return True
    return False


def is_jsonl_file(file_path: Path):
    if file_path.suffix == ".jsonl":
        return True
    return False
