import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import tempfile
import warnings

import torch
from transformers import logging

from data.datamodule import DataModule
from data.tree import AttachTree
from models.classifier import Classifiers
from models.parser import Parsers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.set_verbosity_warning()
logging.set_verbosity_error()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        help="checkpoint (.ckpt) file",
    )
    parser.add_argument(
        "--save-dir", type=Path, required=True, help="path to output directory"
    )

    # dataset
    parser.add_argument(
        "--corpus",
        default="RSTDT",
        choices=["RSTDT", "InstrDT"],
        help="corpus type (label set is in src/data/dataset.py)",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        default=None,
        required=True,
        help="path to raw document file or directory",
    )
    args = parser.parse_args()
    args.train_file = None
    args.valid_file = None
    args.test_file = None
    args.data_dir = None
    args.num_workers = 0
    args.batch_size = 1
    args.batch_unit_type = "document"
    return args


def raw_docs2json(documents: List[Path]):
    raise NotImplementedError
    docs = []
    for doc_file in documents:
        doc = {
            "rst_tree": None,
            "tokens": None,
            "edu_start_indices": None,
            "tokenized_edu_strings": None,
            "edu_starts_sentence": None,
        }
        with open(doc_file) as f:
            edus = []
            for line in f:
                line = line.strip()
                if line:  # not-empty
                    edu = line
            docs.append(doc)

    return docs


def raw_docs2dataset(documents: Union[List, Path], config: argparse.Namespace):
    # list up target documents
    doc_files = []
    if isinstance(documents, Path):
        if documents.is_file():  # docuemnt file
            doc_file = documents
            doc_files = [doc_file]
        else:  # directory contains document files
            doc_files = [doc_file for doc_file in documents.iterdir()]
    else:  # list of docuemnt file
        doc_files = documents

    assert len(doc_files) > 0

    # convert to datamodule via json formatted dataset
    dataset = None
    with tempfile.NamedTemporaryFile() as fp:
        json.dump(raw_docs2json(doc_files), fp)

        config.data_dir = os.path.dirname(fp.name)
        config.test_file = os.path.basename(fp.name)
        dataset = DataModule.from_config(config, parser=None)

    return dataset


def main():
    config = get_config()
    device = torch.device("cuda:0")  # hard codded
    ckpt_path = config.ckpt_path
    save_dir = config.save_dir
    dataset = raw_docs2dataset(config.documents, config)
    parse(ckpt_path, dataset, save_dir, device)
    print("trees for given docs are seved into {}".format(save_dir))
    return


def parse(
    ckpt_path: Union[Path, dict],
    dataset: DataModule,
    save_dir: Path,
    device: torch.device,
):
    # load params from checkpoint
    if isinstance(ckpt_path, Path):
        checkpoint = torch.load(ckpt_path)

    assert "state_dict" in checkpoint
    hparams = checkpoint["hyper_parameters"]

    # build classifier with pre-trained weights
    model_type = hparams["model_type"]
    classifier_class = Classifiers.classifier_dict[model_type]
    classifier = classifier_class.load_from_checkpoint(ckpt_path, map_location=device)

    # build parser
    parser_class = Parsers.parser_dict[model_type]
    parser = parser_class(classifier)

    classifier.eval()
    classifier.to(device)
    classifier.set_parser(parser)

    # build dataloader
    dataset.set_parser(parser)
    parse_set = dataset.test_dataloader()

    # parse
    with torch.no_grad():
        output = parser.parse_dataset(parse_set)
        save_tree(output, save_dir)

    return


def save_tree(output: Dict, save_dir: Optional[Path] = None):
    if save_dir is None:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, tree in zip(output["doc_id"], output["pred_tree"]):
        assert isinstance(tree, AttachTree)
        with open(save_dir / "{}.tree".format(doc_id), "w") as f:
            print(tree, file=f)

    return


if __name__ == "__main__":
    main()
