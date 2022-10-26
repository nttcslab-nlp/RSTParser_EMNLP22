from typing import Tuple

import torch
import torch.nn as nn

from data.batch import Batch
from data.doc import Doc
from models.classifier import TopDownClassifierBase
from models.classifier.linear import DeepBiAffine


class TopDownClassifierV2(TopDownClassifierBase):
    def __init__(self, *args, **kwargs):
        super(TopDownClassifierV2, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.ful_vocab = self.DATASET.fully_label_vocab

        embed_dim = self.encoder.get_embed_dim()
        feat_embed_dim = self.get_org_embedding_dim()

        self.out_linear_label = DeepBiAffine(
            embed_dim,
            self.hidden_dim,
            len(self.ful_vocab),
            self.dropout_p,
            feat_embed_dim,
        )

        pad_idx = self.ful_vocab["<pad>"]
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
        label_scores = self.out_linear_label(left_embeddings, right_embeddings, org_embeddings)

        output = {
            "spl_scores": spl_scores,
            "ful_scores": label_scores,
        }
        return output

    def compute_loss(self, output, batch: Batch):
        spl_loss = self.compute_split_loss(output, batch)
        ful_loss = self.compute_label_loss(output, batch)
        loss = (spl_loss + ful_loss) / 2
        return {
            "loss": loss,
            "spl_loss": spl_loss,
            "full_loss": ful_loss,
        }

    def compute_label_loss(self, output, batch: Batch):
        labels = batch.label
        ful_idx = labels["ful"]
        ful_loss = self.xent_label_loss(output["ful_scores"], ful_idx)
        return ful_loss

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

        ful_scores = self.out_linear_label(left_emb, right_emb, org_emb)
        if return_scores:
            return ful_scores

        ful_scores[self.ful_vocab["<pad>"]] = -float("inf")
        label = self.ful_vocab.lookup_token(torch.argmax(ful_scores))
        return label
