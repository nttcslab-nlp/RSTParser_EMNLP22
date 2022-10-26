import torch.nn as nn

from data.batch import Batch
from data.doc import Doc
from models.classifier import ClassifierBase


class ShiftReduceClassifierBase(ClassifierBase):
    def __init__(self, hidden_dim: int, dropout_p: float = 0.2, *args, **kwargs):
        super(ShiftReduceClassifierBase, self).__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.org_embed = self.init_org_embeddings()

    @classmethod
    def params_from_config(cls, config):
        params = super().params_from_config(config)
        params.update(
            {
                "hidden_dim": config.hidden_dim,
                "dropout_p": config.dropout_p,
            }
        )
        return params

    def init_org_embeddings(self):
        if self.disable_org_feat:
            return None

        num_feat = 0
        if not self.disable_org_sent:
            num_feat += 17
        if not self.disable_org_para:
            num_feat += 11

        return nn.Embedding(num_feat * 2, 10)

    def forward(self, doc: Doc, spans: dict, feats: dict):
        raise NotImplementedError

    def training_loss(self, batch: Batch):
        doc = batch.doc
        spans = batch.span
        feats = batch.feat
        output = self(doc, spans, feats)

        loss_dict = self.compute_loss(output, batch)
        return loss_dict

    def compute_loss(self, output, batch):
        raise NotImplementedError

    def predict(self, document_embedding, span: dict, feat: dict):
        raise NotImplementedError
