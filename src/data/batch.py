from typing import List

import torch
import transformers
from torch.utils.data._utils.collate import default_collate

from data.doc import Doc


class Batch(object):
    def __init__(self, samples: List[dict], unit_type: str):
        assert unit_type in ["document", "span", "span_fast"]
        if unit_type != "span_fast":
            assert len(samples) == 1
        self.unit_type = unit_type

        if self.unit_type == "document":
            # If self.unit_type == 'document', a unit of a batch is a document.
            # Span and label contain all of the parsing elements of documents.
            doc: Doc = samples[0]["doc"]  # Doc
            span = samples[0]["span"]  # List[dict]
            label = samples[0]["label"]  # List[dict]
            feat = samples[0]["feat"]  # List[dict]
        elif self.unit_type == "span":
            # If self.unit_type == 'span', a unit of batch is a span.
            # Each sample has one element of parsing proceduer.
            doc: Doc = samples[0]["doc"]  # Doc
            span = [samples[0]["span"]]  # dict -> List[dict]
            label = [samples[0]["label"]]  # dict -> List[dict]
            feat = [samples[0]["feat"]]  # dict -> List[dict]
        elif self.unit_type == "span_fast":
            # If self.unit_type == 'span_fast', a unit of batch is a span.
            # Each sample has multiple element of parsing proceduer.
            doc: Doc = samples[0]["doc"]  # Doc
            span = samples[0]["span"]  # List[dict]
            label = samples[0]["label"]  # List[dict]
            feat = samples[0]["feat"]  # List[dict]
        else:
            raise ValueError("Invalid batch unit_type ({})".format(unit_type))

        self.doc = doc
        self.inputs = doc.inputs
        self.span = span
        self.feat = feat
        self.label = default_collate(label)

    def __len__(self):
        return len(self.span)

    def __repr__(self):
        return "Batch(doc: {}, span: {}, label: {})".format(self.doc, self.span, self.label)

    def pin_memory(self):
        # if self.inputs is not None:
        #     self.inputs = self._pin_memory(self.inputs)

        # self.span = self._pin_memory(self.span)
        self.feat = self._pin_memory(self.feat)
        self.label = self._pin_memory(self.label)
        return self

    def _pin_memory(self, x):
        if isinstance(x, torch.Tensor):
            return x.pin_memory()
        elif isinstance(x, dict):
            return {k: self._pin_memory(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self._pin_memory(_x) for _x in x]
        elif isinstance(x, tuple):
            return tuple(self._pin_memory(_x) for _x in x)
        else:
            raise ValueError

    def to_device(self, device):
        if self.inputs is not None:
            self.inputs = self._to_device(self.inputs, device)

        # self.span = self._to_device(self.span)
        self.feat = self._to_device(self.feat, device)
        self.label = self._to_device(self.label, device)
        return self

    def _to_device(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, dict):
            return {k: self._to_device(v, device) for k, v in x.items()}
        elif isinstance(x, list):
            return [self._to_device(_x, device) for _x in x]
        elif isinstance(x, tuple):
            return tuple(self._to_device(_x, device) for _x in x)
        elif isinstance(x, transformers.tokenization_utils_base.BatchEncoding):
            return x.to(device)
        else:
            raise ValueError
