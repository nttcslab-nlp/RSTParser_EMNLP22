import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from data.doc import Doc
from models.encoder import Encoder


def complete_model_name(model_name):
    if model_name.startswith("spanbert"):
        return "SpanBERT/" + model_name
    elif model_name.startswith("electra"):
        return "google/" + model_name
    elif model_name.startswith("mpnet"):
        return "microsoft/" + model_name
    elif model_name.startswith("deberta"):
        return "microsoft/" + model_name
    else:
        return model_name


class BertEncoder(Encoder):
    def __init__(self, model_name: str, max_length: int = 512, stride: int = 30):
        super(BertEncoder, self).__init__()
        self.max_length = max_length
        self.stride = stride

        model_name = complete_model_name(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.deprecation_warnings[
            "sequence-length-is-longer-than-the-specified-maximum"
        ] = True
        self.model = AutoModel.from_pretrained(model_name)

    @classmethod
    def from_config(cls, config):
        params = {
            "model_name": config.bert_model_name,
            "max_length": config.bert_max_length,
            "stride": config.bert_stride,
        }
        return cls(**params)

    def get_embed_dim(self):
        return self.config.hidden_size

    def apply_tokenizer(self, doc: Doc):
        edu_strings = doc.get_edu_strings()

        raw_document = " ".join(edu_strings)
        inputs = self.tokenizer(
            raw_document,
            max_length=self.max_length,
            stride=self.stride,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

        flatten_inputs = self.tokenizer(
            raw_document,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        reconst_edu_strings = [
            self.tokenizer.decode(self.tokenizer.encode(edu, add_special_tokens=False))
            for edu in edu_strings
        ]

        input_ids = flatten_inputs.input_ids[0]
        edu_offset = 0
        token_offset = 0
        buf = []
        edu_to_subtokens_mappings = []
        for token_id in input_ids:
            buf.append(token_id)
            edu_from_tokens = self.tokenizer.decode(buf).strip()
            edu = reconst_edu_strings[edu_offset]
            c_edu_from_tokens = edu_from_tokens.replace(" ", "").lower()
            c_edu = edu.replace(" ", "").lower()
            if c_edu_from_tokens == c_edu:  # check a charactor level matching
                edu_to_subtokens_mappings.append([token_offset, token_offset + len(buf)])
                edu_offset += 1
                token_offset = token_offset + len(buf)
                buf = []
            elif len(c_edu_from_tokens) > len(c_edu):
                raise ValueError('"{}" != "{}"'.format(edu_from_tokens, edu))

        # check num of edus and mappings
        assert len(edu_to_subtokens_mappings) == len(edu_strings)
        inputs["edu_to_subtokens_mappings"] = torch.tensor(
            edu_to_subtokens_mappings, dtype=torch.long
        )
        return inputs

    def forward(self, doc: Doc):
        if not hasattr(doc, "inputs") or doc.inputs is None:
            doc.inputs = self.apply_tokenizer(doc).to(self.model.device)

        inputs = doc.inputs

        # run bert model
        outputs = self.model(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # fix a effects of max_length and stride.
        input_ids = []
        embeddings = []
        for idx, (_input_ids, _embeddings, attention_mask, special_tokens_mask,) in enumerate(
            zip(
                inputs["input_ids"],
                outputs.last_hidden_state,
                inputs["attention_mask"],
                inputs["special_tokens_mask"],
            )
        ):

            # at the first, trim special tokens (sep, cls)
            normal_token_indices = torch.where(special_tokens_mask == 0)
            _input_ids = _input_ids[normal_token_indices]
            _embeddings = _embeddings[normal_token_indices]
            if idx == 0:
                input_ids.append(_input_ids)
                embeddings.append(_embeddings)
            else:
                # at the second, trim strided tokens
                input_ids.append(_input_ids[self.stride :])
                embeddings.append(_embeddings[self.stride :])

        input_ids = torch.cat(input_ids, dim=0)
        embeddings = torch.cat(embeddings, dim=0)

        bert_output = {
            "input_ids": input_ids,
            "embeddings": embeddings,
            "edu_to_subtokens_mappings": inputs.edu_to_subtokens_mappings,
            # 'edu_strings': doc.get_edu_strings(),
            # 'subtokens': self.tokenizer.convert_ids_to_tokens(input_ids),
        }
        # self.check_mapping(bert_output, self.tokenizer)
        return bert_output

    def get_span_embedding(self, bert_output, edu_span):
        edu_to_subtokens_mappings = bert_output["edu_to_subtokens_mappings"]
        subtoken_embeddings = bert_output["embeddings"]

        if edu_span == (-1, -1):
            return torch.zeros(self.get_embed_dim(), device=self.model.device)

        i = edu_to_subtokens_mappings[edu_span[0]][0]
        j = edu_to_subtokens_mappings[edu_span[1] - 1][1]
        embedding = (subtoken_embeddings[i] + subtoken_embeddings[j - 1]) / 2
        return embedding

    def batch_get_span_embedding(self, bert_output, edu_spans):
        edu_to_subtokens_mappings = bert_output["edu_to_subtokens_mappings"]
        subtoken_embeddings = bert_output["embeddings"]

        l, r = [], []
        for span in edu_spans:
            l.append(span[0])
            r.append(span[1] - 1)

        i = edu_to_subtokens_mappings[l][:, 0]
        j = edu_to_subtokens_mappings[r][:, 1]
        embedding = (subtoken_embeddings[i] + subtoken_embeddings[j - 1]) / 2
        return embedding

    @staticmethod
    def check_mapping(bert_output, tokenizer=None):
        edus = bert_output["edu_strings"]
        subtokens = bert_output["subtokens"]
        edu_to_subtokens_mappings = bert_output["edu_to_subtokens_mappings"]

        for edu_idx, edu_string in enumerate(edus):
            print("-" * 20)
            print(edu_string)
            subtoken_span = edu_to_subtokens_mappings[edu_idx]
            subtokens_in_edu = []
            for j, subtoken_idx in enumerate(range(*subtoken_span)):
                subtoken = subtokens[subtoken_idx]
                print(subtoken, end=" ")
                subtokens_in_edu.append(subtoken)

            print()

            if tokenizer:
                string = tokenizer.convert_tokens_to_string(subtokens_in_edu)
                print(string)
                assert string.strip() == edu_string

        return
