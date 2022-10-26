from typing import Tuple

import torch
import torch.nn as nn

from data.batch import Batch
from data.doc import Doc
from models.classifier import TopDownClassifierBase
from models.classifier.linear import DeepBiAffine


class TopDownClassifierV1(TopDownClassifierBase):
    def __init__(self, *args, **kwargs):
        super(TopDownClassifierV1, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.nuc_vocab = self.DATASET.nucleus_vocab
        self.rel_vocab = self.DATASET.relation_vocab

        embed_dim = self.encoder.get_embed_dim()
        feat_embed_dim = self.get_org_embedding_dim()

        self.out_linear_nucleus = DeepBiAffine(
            embed_dim,
            self.hidden_dim,
            len(self.nuc_vocab),
            self.dropout_p,
            feat_embed_dim,
        )
        self.out_linear_relation = DeepBiAffine(
            embed_dim,
            self.hidden_dim,
            len(self.rel_vocab),
            self.dropout_p,
            feat_embed_dim,
        )

        assert self.nuc_vocab["<pad>"] == self.rel_vocab["<pad>"]
        pad_idx = self.nuc_vocab["<pad>"]
        self.xent_label_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, doc: Doc, spans: dict, feats: dict):
        document_embedding = self.encoder(doc)

        spl_scores = []
        left_embeddings, right_embeddings, org_embeddings = [], [], []
        for span in spans:
            i, j, k = span["i"], span["j"], span["k"]
            # predict split scores
            spl_scores.append(
                self.predict_split(document_embedding, (i, j), doc, return_scores=True)
            )

            left_emb = self.encoder.get_span_embedding(document_embedding, (i, k))
            right_emb = self.encoder.get_span_embedding(document_embedding, (k, j))
            left_embeddings.append(left_emb)
            right_embeddings.append(right_emb)
            if not self.disable_org_feat:
                org_emb = self.org_embed(
                    self.parser.get_organization_features((i, k), (k, j), doc, self.device)
                ).view(-1)
                org_embeddings.append(org_emb)

        left_embeddings = torch.stack(left_embeddings, dim=0)
        right_embeddings = torch.stack(right_embeddings, dim=0)
        org_embeddings = None if self.disable_org_feat else torch.stack(org_embeddings, dim=0)

        # predict label scores for nuc and rel
        nuc_scores = self.out_linear_nucleus(left_embeddings, right_embeddings, org_embeddings)
        rel_scores = self.out_linear_relation(left_embeddings, right_embeddings, org_embeddings)

        output = {
            "spl_scores": spl_scores,
            "nuc_scores": nuc_scores,
            "rel_scores": rel_scores,
        }
        return output

    def compute_loss(self, output, batch: Batch):
        spl_loss = self.compute_split_loss(output, batch)
        nuc_loss, rel_loss = self.compute_label_loss(output, batch)
        loss = (spl_loss + nuc_loss + rel_loss) / 3

        return {
            "loss": loss,
            "spl_loss": spl_loss,
            "nuc_loss": nuc_loss,
            "rel_loss": rel_loss,
        }

    def compute_label_loss(self, output, batch: Batch):
        labels = batch.label
        nuc_idx = labels["nuc"]
        rel_idx = labels["rel"]
        nuc_loss = self.xent_label_loss(output["nuc_scores"], nuc_idx)
        rel_loss = self.xent_label_loss(output["rel_scores"], rel_idx)
        return nuc_loss, rel_loss

    def predict_label(
        self,
        document_embedding,
        left_span: Tuple[int],
        right_span: Tuple[int],
        doc: Doc,
        return_scores: bool = False,
    ):
        left_emb = self.encoder.get_span_embedding(document_embedding, left_span)
        right_emb = self.encoder.get_span_embedding(document_embedding, right_span)
        org_emb = None
        if not self.disable_org_feat:
            org_emb = self.org_embed(
                self.parser.get_organization_features(left_span, right_span, doc, self.device)
            ).view(-1)

        nuc_scores = self.out_linear_nucleus(left_emb, right_emb, org_emb)
        rel_scores = self.out_linear_relation(left_emb, right_emb, org_emb)

        if return_scores:
            return nuc_scores, rel_scores

        nuc_scores[self.nuc_vocab["<pad>"]] = -float("inf")
        rel_scores[self.rel_vocab["<pad>"]] = -float("inf")
        nuc_label = self.nuc_vocab.lookup_token(torch.argmax(nuc_scores))
        rel_label = self.rel_vocab.lookup_token(torch.argmax(rel_scores))
        label = ":".join([nuc_label, rel_label])
        return label
