import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union
import warnings

import torch
from transformers import logging

from average_ckpt import average_checkpoints
from data.datamodule import DataModule
from data.tree import AttachTree
from metrics import RSTParseval
from models.classifier import Classifiers
from models.parser import Parsers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.set_verbosity_warning()
logging.set_verbosity_error()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="directory contains checkpoint (.ckpt) files",
    )
    parser.add_argument(
        "--average-top-k",
        type=int,
        default=None,
        help="number of checkpoints to compute checkpoint average weights",
    )
    parser.add_argument("--save-dir", type=Path, required=True, help="path to output directory")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="min-batch size (batch-size=1 is only available when evaluation)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="number of workers for dataloader (0 is enough)",
    )

    # dataset
    parser.add_argument(
        "--corpus",
        default="RSTDT",
        choices=["RSTDT", "InstrDT"],
        help="corpus type (label set is in src/data/dataset.py)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data/",
        help="dataset directory which contain train/valid/test json files",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default="train.json",
        help="file name of training dataset",
    )
    parser.add_argument(
        "--valid-file",
        type=Path,
        default="valid.json",
        help="file name of valiation file",
    )
    parser.add_argument(
        "--test-file", type=Path, default="test.json", help="file name of test dataset"
    )
    config = parser.parse_args()
    device = torch.device("cuda:0")  # hard codded
    # device = torch.device('cpu')

    # load dataset
    dataset = DataModule.from_config(config, parser=None)

    # ckpt list
    ckpt_path_list = []
    for ckpt_path in config.ckpt_dir.iterdir():
        if ckpt_path.suffix != ".ckpt":
            continue
        if "average" in str(ckpt_path):
            continue
        if "last" in str(ckpt_path):
            continue
        if "best" in str(ckpt_path):
            continue
        ckpt_path_list.append(ckpt_path)

    # evaluate each checkpoint
    scores = []
    for ckpt_path in ckpt_path_list:
        print(ckpt_path)
        # ckpt_path: hoge/fuga/epoch=n-step=m.ckpt
        model_name = ckpt_path.stem
        save_dir = config.save_dir / model_name
        valid_score = test(ckpt_path, dataset, save_dir, device)
        scores.append({"path": ckpt_path, "score": valid_score})

    sorted_scores = sorted(scores, reverse=True, key=lambda x: x["score"]["RSTParseval-F"])

    # select the best checkpoint with the validation score
    best_ckpt_path = sorted_scores[0]["path"]
    print("the best model: {}".format(best_ckpt_path))
    shutil.copyfile(best_ckpt_path, config.ckpt_dir / "best.ckpt")
    print("the best model was saved as {}".format(config.ckpt_dir / "best.ckpt"))
    save_dir = config.save_dir / "best"
    print("evaluate the best model")
    test(best_ckpt_path, dataset, save_dir, device)
    print("trees of the best model are seved into {}".format(save_dir))

    # select top_k models with the validation score
    ckpt_path_list_for_avg = [m["path"] for m in sorted_scores[: config.average_top_k]]
    print("models for weight average:")
    for path in ckpt_path_list_for_avg:
        print(" - {}".format(path))

    avg_ckpt_path = config.ckpt_dir / "average.ckpt"
    avg_ckpt = average_checkpoints(ckpt_path_list_for_avg)
    torch.save(avg_ckpt, avg_ckpt_path)
    print("the averaged model was saved as {}".format(avg_ckpt_path))

    print("evaluate the averaged model")
    save_dir = config.save_dir / "average"
    test(avg_ckpt_path, dataset, save_dir, device)
    print("trees of the averaged model are seved into {}".format(save_dir))

    return


def test(
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
    test_set = dataset.test_dataloader()
    valid_set = dataset.val_dataloader()[0]

    metric = RSTParseval()
    with torch.no_grad():
        output = parser.parse_dataset(valid_set)
        metric.update(output["pred_tree"], output["gold_tree"])
        valid_score = metric.compute()
        save_tree(output, save_dir / "valid")
        print(valid_score)
        metric.reset()

        output = parser.parse_dataset(test_set)
        metric.update(output["pred_tree"], output["gold_tree"])
        test_score = metric.compute()
        save_tree(output, save_dir / "test")
        print(test_score)
        metric.reset()

    return valid_score


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
