import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch.multiprocessing
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodule import DataModule
from models.classifier import Classifiers
from models.parser import Parsers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.multiprocessing.set_sharing_strategy("file_system")


def main():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--corpus", default="RSTDT", choices=["RSTDT", "InstrDT"])
    parser.add_argument("--data-dir", type=Path, default="data/")
    parser.add_argument("--train-file", type=Path, default="train.json")
    parser.add_argument("--valid-file", type=Path, default="valid.json")
    parser.add_argument("--test-file", type=Path, default="test.json")

    # model parameters
    parser.add_argument(
        "--model-type",
        required=True,
        choices=[
            "top_down_v1",
            "top_down_v2",
            "shift_reduce_v1",
            "shift_reduce_v2",
            "shift_reduce_v3",
        ],
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout-p", type=float, default=0.2)
    parser.add_argument(
        "--disable-penalty", action="store_true"
    )  # for top_down_v1, 2, shift_reduce_v3
    parser.add_argument("--disable-org-sent", action="store_true")
    parser.add_argument("--disable-org-para", action="store_true")

    # bert settings
    # parser.add_argument('--encoder-type', required=True,
    #                     choices=['bert', 'elmo', 'glove'])
    parser.add_argument(
        "--bert-model-name",
        required=True,
        choices=[
            "bert-base-cased",
            "bert-large-cased",
            "roberta-base",
            "roberta-large",
            "xlnet-base-cased",
            "xlnet-large-cased",
            "spanbert-base-cased",
            "spanbert-large-cased",
            "electra-base-discriminator",
            "electra-large-discriminator",
            "mpnet-base",
            "deberta-base",
            "deberta-large",
        ],
    )
    parser.add_argument("--bert-max-length", type=int, default=512)
    parser.add_argument("--bert-stride", type=int, default=30)

    # optimizer setteings
    parser.add_argument("--lr-for-encoder", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--disable-lr-schedule", action="store_true")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)

    # training settings
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-unit-type",
        choices=["span", "span_fast", "document"],
        default="span_fast",
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--train-from", type=Path, default=None)
    parser.add_argument("--disable-span-level-validation", action="store_true")
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)

    # save directorie and model name
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--model-version", type=int, default=None)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)

    # random seed
    parser.add_argument("--seed", type=int, default=None)
    config = parser.parse_args()

    if config.batch_unit_type == "span_fast":
        assert config.batch_size > 0
        assert config.accumulate_grad_batches == 1

    if config.seed is not None:
        seed_everything(config.seed, workers=True)

    classifier = Classifiers.from_config(config)
    parser = Parsers.from_config(config, classifier)
    classifier.set_parser(parser)  # for validation step.

    data_module = DataModule.from_config(config, parser)
    training_steps = len(data_module.train_dataloader())
    classifier.set_training_steps_par_epoch(training_steps)

    logger = TensorBoardLogger(
        save_dir=config.save_dir,
        name=config.model_name,
        version=config.model_version,
        default_hp_metric=False,
    )

    callbacks = [LearningRateMonitor(logging_interval="step")]

    if not config.disable_early_stopping:
        monitor_metric = "valid/OriginalParseval-F"
        callbacks.append(
            EarlyStopping(monitor=monitor_metric, mode="max", patience=config.patience)
        )

    if config.no_save_checkpoint:
        enable_checkpointing = False
    else:
        enable_checkpointing = True
        monitor_metric = "valid/OriginalParseval-F"
        callbacks.append(
            ModelCheckpoint(monitor=monitor_metric, mode="max", save_last=True, save_top_k=3)
        )

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        gpus=config.num_gpus,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        default_root_dir=config.save_dir,
        resume_from_checkpoint=config.train_from,
        # accelerator='ddp',
        # plugins=DDPPlugin(find_unused_parameters=True),
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        logger=logger,
        val_check_interval=0.33,  # check the validation 3 times per epoch.
        reload_dataloaders_every_n_epochs=1 if config.batch_unit_type == "span_fast" else 0,
        num_sanity_val_steps=0,
        # detect_anomaly=True,
    )

    trainer.fit(classifier, data_module)

    return


if __name__ == "__main__":
    main()
