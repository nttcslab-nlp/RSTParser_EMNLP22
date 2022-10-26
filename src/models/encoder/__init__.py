from models.encoder.encoder import Encoder
from models.encoder.bert_encoder import BertEncoder


class Encoders:
    encoder_dict = {
        "bert": BertEncoder,
    }

    @classmethod
    def from_config(cls, config):
        encoder_type = config.encoder_type
        encoder = cls.encoder_dict[encoder_type].from_config(config)
        return encoder


__all__ = ["Encoder", "BertEncoder"]
