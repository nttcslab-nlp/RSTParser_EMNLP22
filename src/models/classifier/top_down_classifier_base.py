from typing import Dict, Tuple

import torch
import torch.nn as nn

from data.batch import Batch
from data.doc import Doc
from models.classifier import ClassifierBase
from models.classifier.linear import DeepBiAffine


class TopDownClassifierBase(ClassifierBase):
    def __init__(
        self,
        hidden_dim: int,
        dropout_p: float = 0.2,
        disable_penalty: bool = False,
        *args,
        **kwargs
    ):
        super(TopDownClassifierBase, self).__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.disable_penalty = disable_penalty

        self.org_embed = self.init_org_embeddings()

        embed_dim = self.encoder.get_embed_dim()
        feat_embed_dim = self.get_org_embedding_dim()

        self.out_linear_split = DeepBiAffine(
            embed_dim, self.hidden_dim, 1, self.dropout_p, feat_embed_dim=feat_embed_dim
        )

        self.xent_split_loss = nn.CrossEntropyLoss(reduction="none")

    @classmethod
    def params_from_config(cls, config):
        params = super().params_from_config(config)
        params.update(
            {
                "hidden_dim": config.hidden_dim,
                "dropout_p": config.dropout_p,
                "disable_penalty": config.disable_penalty,
            }
        )
        return params

    def init_org_embeddings(self):
        if self.disable_org_feat:
            return None

        num_feat = 0
        if not self.disable_org_sent:
            num_feat += 10
        if not self.disable_org_para:
            num_feat += 6

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

    def compute_loss(self, output, batch: Batch):
        raise NotImplementedError

    def compute_split_loss(self, output, batch: Batch):
        labels = batch.label
        spans = batch.span

        spl_idx = labels["spl"]
        spl_losses = [
            self.xent_split_loss(scores.unsqueeze(0), idx.unsqueeze(0))
            for scores, idx in zip(output["spl_scores"], spl_idx)
        ]

        if not self.disable_penalty:
            # Segmentation loss with penalty (Koto et al., 2021)
            beta = 0.35
            spl_losses = [
                (1 + (span["j"] - span["i"])) ** beta * loss
                for loss, span in zip(spl_losses, spans)
            ]

        spl_loss = torch.mean(torch.stack(spl_losses, dim=0))
        return spl_loss

    def compute_label_loss(self, output, batch: Batch):
        raise NotImplementedError

    def predict_split(
        self,
        document_embedding,
        span: Tuple[int],
        doc: Doc,
        return_scores: bool = False,
    ):
        i, j = span
        left_embeddings, right_embeddings, org_indices = [], [], []
        for k in range(i + 1, j):
            left_embeddings.append(self.encoder.get_span_embedding(document_embedding, (i, k)))
            right_embeddings.append(self.encoder.get_span_embedding(document_embedding, (k, j)))
            if not self.disable_org_feat:
                org_idx = self.parser.get_organization_features((i, k), (k, j), doc, self.device)
                org_indices.append(org_idx)

        left_embeddings = torch.stack(left_embeddings, dim=0)
        right_embeddings = torch.stack(right_embeddings, dim=0)

        org_embeddings = None
        if not self.disable_org_feat:
            n = len(org_indices)  # num of split points
            org_indices = torch.stack(org_indices)
            org_embeddings = self.org_embed(org_indices).view(n, -1)

        split_scores = self.out_linear_split(
            left_embeddings, right_embeddings, org_embeddings
        ).squeeze(1)

        if return_scores:
            return split_scores

        k = torch.argmax(split_scores) + span[0] + 1
        return k.item()

    def predict_label(
        self,
        document_embedding,
        left_span: Tuple[int],
        right_span: Tuple[int],
        doc: Doc,
    ):
        raise NotImplementedError

    def predict_split_fast(
        self,
        document_embedding,
        span: Tuple[int],
        doc: Doc,
        span_to_org_indices: Dict,
        return_scores: bool = False,
    ):
        i, j = span
        left_spans = []
        right_spans = []
        for k in range(i + 1, j):
            left_spans.append((i, k))
            right_spans.append((k, j))

        left_embeddings = self.encoder.batch_get_span_embedding(document_embedding, left_spans)
        right_embeddings = self.encoder.batch_get_span_embedding(document_embedding, right_spans)

        if not self.disable_org_feat:
            org_indices = span_to_org_indices[span]

        org_embeddings = None
        if not self.disable_org_feat:
            n = len(org_indices)  # num of split points
            org_indices = torch.stack(org_indices).to(self.device)
            org_embeddings = self.org_embed(org_indices).view(n, -1)

        split_scores = self.out_linear_split(
            left_embeddings, right_embeddings, org_embeddings
        ).squeeze(1)

        if return_scores:
            return split_scores

        k = torch.argmax(split_scores) + span[0] + 1
        return k.item()
