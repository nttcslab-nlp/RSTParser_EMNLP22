import torch
import torch.nn as nn

from data.batch import Batch
from data.doc import Doc
from models.classifier import ShiftReduceClassifierBase
from models.classifier.linear import FeedForward


class ShiftReduceClassifierV3(ShiftReduceClassifierBase):
    def __init__(self, disable_penalty: bool = False, *args, **kwargs):
        super(ShiftReduceClassifierV3, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.act_nuc_vocab = self.DATASET.act_nuc_vocab
        self.rel_vocab = self.DATASET.relation_vocab

        embed_dim = self.encoder.get_embed_dim() * 3
        feat_embed_dim = self.get_org_embedding_dim()
        embed_dim += feat_embed_dim

        self.out_linear_act_nuc = FeedForward(
            embed_dim, self.hidden_dim, len(self.act_nuc_vocab), self.dropout_p
        )
        self.out_linear_relation = FeedForward(
            embed_dim, self.hidden_dim, len(self.rel_vocab), self.dropout_p
        )

        self.disable_penalty = disable_penalty
        assert self.act_nuc_vocab["<pad>"] == self.rel_vocab["<pad>"]
        pad_idx = self.act_nuc_vocab["<pad>"]
        self.pad_idx = pad_idx

        self.act_nuc_weight = None
        if not self.disable_penalty:
            act2weight = {
                "shift_<pad>": 3 / 6,
                "reduce_nucleus-satellite": 1 / 6,
                "reduce_satellite-nucleus": 1 / 6,
                "reduce_nucleus-nucleus": 1 / 6,
                "<pad>": 0,
            }
            weight = torch.tensor([act2weight[act] for act in self.act_nuc_vocab.itos])
            self.act_nuc_weight = nn.Parameter(weight, requires_grad=False)

        self.xent_act_nuc_loss = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="none")
        self.xent_relation_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)

    @classmethod
    def params_from_config(cls, config):
        params = super().params_from_config(config)
        params.update(
            {
                "disable_penalty": config.disable_penalty,
            }
        )
        return params

    def forward(self, doc: Doc, spans: dict, feats: dict):
        document_embedding = self.encoder(doc)

        span_embeddings = []
        for span, feat in zip(spans, feats):
            s1_emb = self.encoder.get_span_embedding(document_embedding, span["s1"])
            s2_emb = self.encoder.get_span_embedding(document_embedding, span["s2"])
            q1_emb = self.encoder.get_span_embedding(document_embedding, span["q1"])
            embedding = torch.cat((s1_emb, s2_emb, q1_emb), dim=0)

            if not self.disable_org_feat:
                org_emb = self.org_embed(feat["org"]).view(-1)
                embedding = torch.cat((embedding, org_emb), dim=0)

            span_embeddings.append(embedding)

        span_embeddings = torch.stack(span_embeddings, dim=0)

        # predict label scores for act_nuc and rel
        act_nuc_scores = self.out_linear_act_nuc(span_embeddings)
        rel_scores = self.out_linear_relation(span_embeddings)

        output = {
            "act_nuc_scores": act_nuc_scores,
            "rel_scores": rel_scores,
        }
        return output

    def compute_loss(self, output, batch: Batch):
        labels = batch.label
        act_nuc_idx = labels["act_nuc"]
        rel_idx = labels["rel"]
        act_nuc_scores = output["act_nuc_scores"]
        act_nuc_losses = self.xent_act_nuc_loss(act_nuc_scores, act_nuc_idx)

        # weighting for inbalance of shift-reduce action (Guz et al., 2020)
        if not self.disable_penalty:
            weight = self.act_nuc_weight[act_nuc_idx]
            act_nuc_losses = weight * act_nuc_losses

        act_nuc_loss = torch.mean(act_nuc_losses, dim=0)
        rel_loss = self.xent_relation_loss(output["rel_scores"], rel_idx)
        if torch.all(rel_idx == self.pad_idx):
            rel_loss = torch.zeros_like(rel_loss)

        loss = (act_nuc_loss + rel_loss) / 2

        return {
            "loss": loss,
            "act_nuc_loss": act_nuc_loss,
            "rel_loss": rel_loss,
        }

    def predict(self, document_embedding, span: dict, feat: dict):
        s1_emb = self.encoder.get_span_embedding(document_embedding, span["s1"])
        s2_emb = self.encoder.get_span_embedding(document_embedding, span["s2"])
        q1_emb = self.encoder.get_span_embedding(document_embedding, span["q1"])
        embedding = torch.cat((s1_emb, s2_emb, q1_emb), dim=0)
        if not self.disable_org_feat:
            org_emb = self.org_embed(feat["org"]).view(-1)
            embedding = torch.cat((embedding, org_emb), dim=0)

        act_nuc_scores = self.out_linear_act_nuc(embedding)
        rel_scores = self.out_linear_relation(embedding)
        return act_nuc_scores, rel_scores

    def act_nuc_to_act_scores(self, act_nuc_scores: torch.Tensor):
        shift_idx = self.act_nuc_vocab["shift_<pad>"]
        reduce_idxs = [
            self.act_nuc_vocab["reduce_{}".format(ns)]
            for ns in ["nucleus-satellite", "satellite-nucleus", "nucleus-nucleus"]
        ]

        # compute probability
        Z = sum(
            [act_nuc_scores[shift_idx].exp()]
            + [act_nuc_scores[reduce_idx].exp() for reduce_idx in reduce_idxs]
        )
        shift_prob = act_nuc_scores[shift_idx].exp() / Z
        reduce_prob = 1 - shift_prob  # (reduce_n-s, reduce_s-n, reduce_n-n)

        # compute log-likelihood for each action
        shift_log = shift_prob.clamp(min=1e-6).log()
        reduce_log = reduce_prob.clamp(min=1e-6).log()

        return {"shift": shift_log, "reduce": reduce_log}
