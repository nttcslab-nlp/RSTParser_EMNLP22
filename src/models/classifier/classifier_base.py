from typing import List, Union

import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from data.batch import Batch
from data.dataset import RSTDT, InstrDT
from metrics import OriginalParseval, RSTParseval
from models.encoder import BertEncoder


class ClassifierBase(pl.LightningModule):
    def __init__(
        self,
        model_type: str,
        bert_model_name: str,
        bert_max_length: int,
        bert_stride: int,
        corpus: str,
        accumulate_grad_batches: int,
        batch_unit_type: str,
        lr_for_encoder: float,
        lr: float,
        disable_lr_schedule: bool = False,
        disable_org_sent: bool = False,
        disable_org_para: bool = False,
    ):
        super(ClassifierBase, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.lr_for_encoder = lr_for_encoder
        self.disable_lr_schedule = disable_lr_schedule

        corpus2DATASET = {
            "RSTDT": RSTDT,
            "InstrDT": InstrDT,
        }
        assert corpus in corpus2DATASET
        self.DATASET = corpus2DATASET[corpus]

        self.met_rst_parseval = RSTParseval()
        self.met_ori_parseval = OriginalParseval()

        self.met_rst_parseval_oracle = RSTParseval()
        self.met_ori_parseval_oracle = OriginalParseval()

        self.parser = None
        self.encoder = BertEncoder(bert_model_name, bert_max_length, bert_stride)

        self.disable_org_sent = disable_org_sent
        self.disable_org_para = disable_org_para
        self.disable_org_feat = self.disable_org_sent and self.disable_org_para

    @classmethod
    def from_config(cls, config):
        params = cls.params_from_config(config)
        return cls(**params)

    @classmethod
    def params_from_config(cls, config):
        return {
            "model_type": config.model_type,
            "bert_model_name": config.bert_model_name,
            "bert_max_length": config.bert_max_length,
            "bert_stride": config.bert_stride,
            "corpus": config.corpus,
            "accumulate_grad_batches": config.accumulate_grad_batches,
            "batch_unit_type": config.batch_unit_type,
            "lr_for_encoder": config.lr_for_encoder,
            "lr": config.lr,
            "disable_lr_schedule": config.disable_lr_schedule,
            "disable_org_sent": config.disable_org_sent,
            "disable_org_para": config.disable_org_para,
        }

    def set_parser(self, parser):
        self.parser = parser

    def set_training_steps_par_epoch(self, training_steps):
        self.training_steps_par_epoch = training_steps

    def init_org_embeddings(self):
        raise NotImplementedError

    def get_org_embedding_dim(self):
        if self.disable_org_feat:
            return 0

        n_dim = self.org_embed.embedding_dim
        n_feat = self.org_embed.num_embeddings // 2
        feat_embed_dim = n_feat * n_dim
        return feat_embed_dim

    def forward(self):
        raise NotImplementedError

    def training_loss(self, batch: Batch):
        raise NotImplementedError

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp_metric/RSTParseval-S": 0,
                "hp_metric/RSTParseval-N": 0,
                "hp_metric/RSTParseval-R": 0,
                "hp_metric/RSTParseval-F": 0,
                "hp_metric/OriginalParseval-S": 0,
                "hp_metric/OriginalParseval-N": 0,
                "hp_metric/OriginalParseval-R": 0,
                "hp_metric/OriginalParseval-F": 0,
            },
        )

    def training_step(self, batch: Batch, batch_idx: Union[int, None] = None):
        batch.to_device(self.device)
        loss_dict = self.training_loss(batch)

        for name, value in loss_dict.items():
            self.log(
                "train/{}".format(name),
                value,
                on_epoch=True,
                on_step=True,
                batch_size=len(batch),
            )

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx: int, dataloader_idx: Union[int, None] = None):
        batch.to_device(self.device)
        if batch.unit_type == "span":
            loss_dict = self.training_loss(batch)

            for name, value in loss_dict.items():
                self.log(
                    "valid/{}".format(name),
                    value,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    batch_size=len(batch),
                )

            return loss_dict["loss"]

        elif batch.unit_type == "document":
            doc = batch.doc
            pred_tree = self.parser.parse(doc)
            gold_tree = doc.tree

            self.met_rst_parseval([pred_tree], [gold_tree])
            self.met_ori_parseval([pred_tree], [gold_tree])

            pred_tree = self.parser.parse_with_naked_tree(doc, doc.tree)
            gold_tree = doc.tree

            self.met_rst_parseval_oracle([pred_tree], [gold_tree])
            self.met_ori_parseval_oracle([pred_tree], [gold_tree])

            return
        else:
            raise ValueError

    def test_step(self, batch, batch_idx=None):
        batch.to_device(self.device)
        assert batch.unit_type == "document"

        doc = batch.doc
        pred_tree = self.parser.parse(doc)
        gold_tree = doc.tree

        self.met_rst_parseval([pred_tree], [gold_tree])
        self.met_ori_parseval([pred_tree], [gold_tree])

        return

    def validation_epoch_end(self, outputs: List):
        scores = self.met_rst_parseval.compute()
        for name, value in scores.items():
            self.log("valid/{}".format(name), value, prog_bar=True)
            self.log("hp_metric/{}".format(name), value, prog_bar=True)

        self.met_rst_parseval.reset()

        scores = self.met_ori_parseval.compute()
        for name, value in scores.items():
            self.log("valid/{}".format(name), value, prog_bar=True)
            self.log("hp_metric/{}".format(name), value, prog_bar=True)

        self.met_ori_parseval.reset()

        scores = self.met_rst_parseval_oracle.compute()
        for name, value in scores.items():
            self.log("valid/{}_oracle".format(name), value, prog_bar=False)

        self.met_rst_parseval_oracle.reset()

        scores = self.met_ori_parseval_oracle.compute()
        for name, value in scores.items():
            self.log("valid/{}_oracle".format(name), value, prog_bar=False)

        self.met_ori_parseval_oracle.reset()

        return

    def test_epoch_end(self, outputs):
        scores = self.met_rst_parseval.compute()
        for name, value in scores.items():
            self.log("test/{}".format(name), value, prog_bar=True)

        self.met_rst_parseval.reset()

        scores = self.met_ori_parseval.compute()
        for name, value in scores.items():
            self.log("test/{}".format(name), value, prog_bar=True)

        self.met_ori_parseval.reset()

        return

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            # BERT parameters
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay) and "encoder" in n
                ],
                "weight_decay": 0.01,
                "lr": self.lr_for_encoder,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and "encoder" in n
                ],
                "weight_decay": 0.0,
                "lr": self.lr_for_encoder,
            },
            # Other parameters
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay) and "encoder" not in n
                ],
                "weight_decay": 0.01,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and "encoder" not in n
                ],
                "weight_decay": 0.0,
                "lr": self.lr,
            },
        ]
        optimizer = AdamW(params)
        optimizers = [optimizer]

        if self.disable_lr_schedule:
            return optimizers

        num_epochs = self.trainer.max_epochs
        accum_size = self.trainer.accumulate_grad_batches
        num_training_steps = self.training_steps_par_epoch / accum_size * num_epochs
        num_warmup_steps = int(num_training_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        schedulers = [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        return optimizers, schedulers
