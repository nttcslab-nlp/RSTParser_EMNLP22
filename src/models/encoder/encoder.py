import torch.nn as nn

from data.doc import Doc


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def forward(self, doc: Doc):
        raise NotImplementedError

    def get_embed_dim(self):
        raise NotImplementedError

    def get_span_embedding(self, encoder_output, span):
        raise NotImplementedError
