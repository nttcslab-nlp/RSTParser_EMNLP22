from pathlib import Path
from typing import List, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.batch import Batch
from data.dataset import RSTDT, InstrDT
from models.parser import ParserBase


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        corpus: str,
        data_dir: Union[Path, str],
        train_file: Union[str, None] = None,
        valid_file: Union[str, None] = None,
        test_file: Union[str, None] = None,
        parser: Union[ParserBase, None] = None,
        batch_unit_type: str = "document",
        batch_size: int = 1,
        num_workers: int = 0,
        disable_span_level_validation: bool = False,
    ):
        super(DataModule, self).__init__()
        self.corpus = corpus
        self.parser = parser
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.batch_unit_type = batch_unit_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.disable_span_level_validation = disable_span_level_validation

        if self.batch_unit_type != "span_fast" and self.batch_size != 1:
            raise ValueError(
                "Please use the `--accumulate-grad-batches` instead of the --batch-size."
            )

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.setup()

    @classmethod
    def from_config(cls, config, parser: ParserBase):
        if not hasattr(config, "batch_unit_type"):
            # for test
            config.batch_unit_type = "document"

        params = {
            "corpus": config.corpus,
            "parser": parser,
            "data_dir": config.data_dir,
            "train_file": config.train_file,
            "valid_file": config.valid_file,
            "test_file": config.test_file,
            "batch_unit_type": config.batch_unit_type,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "disable_span_level_validation": getattr(
                config, "disable_span_level_validation", False
            ),
        }
        return cls(**params)

    def setup(self, stage=None):
        corpus2DATASET = {
            "RSTDT": RSTDT,
            "InstrDT": InstrDT,
        }
        assert self.corpus in corpus2DATASET
        DATASET = corpus2DATASET[self.corpus]

        if self.train_file is not None:
            self.train_dataset = DATASET(self.data_dir / self.train_file)
            if self.parser is not None:
                self.train_dataset.numericalize_document(self.parser.classifier.encoder)

        if self.valid_file is not None:
            self.valid_dataset = DATASET(self.data_dir / self.valid_file)
            if self.parser is not None:
                self.valid_dataset.numericalize_document(self.parser.classifier.encoder)

        if self.test_file is not None:
            self.test_dataset = DATASET(self.data_dir / self.test_file)
            if self.parser is not None:
                self.test_dataset.numericalize_document(self.parser.classifier.encoder)

        return

    def set_parser(self, parser: ParserBase):
        self.parser = parser

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        unit_type = self.batch_unit_type
        if unit_type == "span_fast":
            dataloader_batch_size = 1
            samples_batch_size = self.batch_size
        else:
            dataloader_batch_size = self.batch_size
            samples_batch_size = None

        samples = self.parser.generate_training_samples(
            self.train_dataset, unit_type, samples_batch_size
        )
        return DataLoader(
            samples,
            batch_size=dataloader_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn_wrapper(unit_type),
            pin_memory=False,
        )

    def val_dataloader(self):
        if self.valid_dataset is None:
            return None
        batch_size = 1 if self.batch_unit_type == "span_fast" else self.batch_size
        assert batch_size == 1
        doc_samples = self.parser.generate_training_samples(self.valid_dataset, "document")
        doc_dataloader = DataLoader(
            doc_samples,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn_wrapper("document"),
            pin_memory=False,
        )

        if self.disable_span_level_validation:
            return doc_dataloader

        span_samples = self.parser.generate_training_samples(self.valid_dataset, "span")
        span_dataloader = DataLoader(
            span_samples,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn_wrapper("span"),
            pin_memory=False,
        )

        return [doc_dataloader, span_dataloader]

    def test_dataloader(self):
        if self.test_dataset is None:
            return None

        unit_type = "document"
        batch_size = 1 if self.batch_unit_type == "span_fast" else self.batch_size
        assert batch_size == 1
        samples = self.parser.generate_training_samples(self.test_dataset, unit_type)
        return DataLoader(
            samples,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn_wrapper(unit_type="document"),
            pin_memory=False,
        )


class collate_fn_wrapper:
    def __init__(self, unit_type: str):
        self.unit_type = unit_type

    def __call__(self, samples: List[dict]):
        # data[i]['doc']: Doc
        # data[i]['span']: dict
        # data[i]['label']: dict
        return Batch(samples, unit_type=self.unit_type)
