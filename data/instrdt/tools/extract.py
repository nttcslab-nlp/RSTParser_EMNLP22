import argparse
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="input json file",
    )
    parser.add_argument(
        "--ids",
        type=Path,
        required=True,
        help="text file contains doc_ids",
    )
    parser.add_argument(
        "--tgt",
        type=Path,
        required=True,
        help="output json file",
    )
    config = parser.parse_args()

    doc_ids = None
    with open(config.ids) as f:
        doc_ids = {line.strip(): None for line in f}

    assert len(doc_ids) > 0

    doc_id2data = {}
    with open(config.src) as f:
        for data in json.load(f):
            doc_id = data["doc_id"]
            if doc_id in doc_ids:
                assert doc_id not in doc_id2data, "Duplicate doc_ids"
                doc_id2data[doc_id] = data

    dataset = [doc_id2data[doc_id] for doc_id in doc_ids]
    with open(config.tgt, "w") as f:
        json.dump(dataset, f)

    return


if __name__ == "__main__":
    main()
