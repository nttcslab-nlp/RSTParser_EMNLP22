import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

import torch
from torchtext.vocab import vocab

from data.doc import Doc
from data.rstdt_relation import re_categorize as rstdt_re_categorize
from data.tree import AttachTree, RSTTree
from data.utils import is_json_file, is_jsonl_file
from models.encoder import Encoder


class Dataset(torch.utils.data.Dataset):
    nucleus_vocab = vocab(
        Counter(["nucleus-satellite", "satellite-nucleus", "nucleus-nucleus"]),
        specials=["<pad>"],
    )
    action_vocab = vocab(Counter(["shift", "reduce"]), specials=["<pad>"])
    act_nuc_vocab = vocab(
        Counter(
            [
                "shift_<pad>",
                "reduce_nucleus-satellite",
                "reduce_satellite-nucleus",
                "reduce_nucleus-nucleus",
            ]
        ),
        specials=["<pad>"],
    )

    def __init__(self, file_path: Union[Path, str]):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        raw_dataset = self.load(file_path)
        dataset: List[Doc] = self.preprocess(raw_dataset)
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def load(self, file_path):
        if is_json_file(file_path):
            return self.load_from_json(file_path)
        if is_jsonl_file(file_path):
            return self.load_from_jsonl(file_path)

        raise NotImplementedError

    def numericalize_document(self, encoder: Encoder):
        for doc in self.dataset:
            inputs = encoder.apply_tokenizer(doc)
            doc.inputs = inputs

        return

    @staticmethod
    def load_from_json(file_path):
        with open(file_path) as f:
            dataset = json.load(f)
        return dataset

    @staticmethod
    def load_from_jsonl(file_path):
        dataset = []
        with open(file_path) as f:
            for line in f:
                data = json.loads(f)
                dataset.append(data)

        return dataset


class RSTDT(Dataset):
    relation_vocab = vocab(
        Counter(
            [
                "Elaboration",
                "Attribution",
                "Joint",
                "Same-unit",
                "Contrast",
                "Explanation",
                "Background",
                "Cause",
                "Enablement",
                "Evaluation",
                "Temporal",
                "Condition",
                "Comparison",
                "Topic-Change",
                "Summary",
                "Manner-Means",
                "Textual-organization",
                "Topic-Comment",
            ]
        ),
        specials=["<pad>"],
    )
    fully_label_vocab = vocab(  # from TRAINING (TRAIN+DEV)
        Counter(
            [
                # N-S
                "nucleus-satellite:Elaboration",
                "nucleus-satellite:Attribution",
                "nucleus-satellite:Explanation",
                "nucleus-satellite:Enablement",
                "nucleus-satellite:Background",
                "nucleus-satellite:Evaluation",
                "nucleus-satellite:Cause",
                "nucleus-satellite:Contrast",
                "nucleus-satellite:Condition",
                "nucleus-satellite:Comparison",
                "nucleus-satellite:Manner-Means",
                "nucleus-satellite:Summary",
                "nucleus-satellite:Temporal",
                "nucleus-satellite:Topic-Comment",
                "nucleus-satellite:Topic-Change",
                # S-N
                "satellite-nucleus:Attribution",
                "satellite-nucleus:Contrast",
                "satellite-nucleus:Background",
                "satellite-nucleus:Condition",
                "satellite-nucleus:Cause",
                "satellite-nucleus:Evaluation",
                "satellite-nucleus:Temporal",
                "satellite-nucleus:Explanation",
                "satellite-nucleus:Enablement",
                "satellite-nucleus:Comparison",
                "satellite-nucleus:Elaboration",
                "satellite-nucleus:Manner-Means",
                "satellite-nucleus:Summary",
                "satellite-nucleus:Topic-Comment",
                # N-N
                "nucleus-nucleus:Joint",
                "nucleus-nucleus:Same-unit",
                "nucleus-nucleus:Contrast",
                "nucleus-nucleus:Temporal",
                "nucleus-nucleus:Topic-Change",
                "nucleus-nucleus:Textual-organization",
                "nucleus-nucleus:Comparison",
                "nucleus-nucleus:Topic-Comment",
                "nucleus-nucleus:Cause",
                "nucleus-nucleus:Condition",
                "nucleus-nucleus:Explanation",
                "nucleus-nucleus:Evaluation",
            ]
        ),
        specials=["<pad>"],
    )

    def preprocess(self, raw_dataset: List[Dict]):
        dataset = []
        for data in raw_dataset:
            rst_tree = RSTTree.fromstring(data["rst_tree"])
            rst_tree = rstdt_re_categorize(rst_tree)
            assert RSTTree.check_relation(rst_tree, self.relation_vocab)
            bi_rst_tree = RSTTree.binarize(rst_tree)
            attach_tree = RSTTree.convert_to_attach(bi_rst_tree)
            data["attach_tree"] = attach_tree
            # (wsj_1189 has annotateion error)
            if data["doc_id"] != "wsj_1189":  # check conversion
                assert bi_rst_tree == AttachTree.convert_to_rst(attach_tree)

            tokenized_edu_strings = []
            edu_starts_sentence = []

            tokens = data["tokens"]
            edu_start_indices = data["edu_start_indices"]
            sent_id, token_id, _ = edu_start_indices[0]
            for next_sent_id, next_token_id, _ in edu_start_indices[1:] + [(-1, -1, -1)]:
                end_token_id = next_token_id if token_id < next_token_id else None
                tokenized_edu_strings.append(tokens[sent_id][token_id:end_token_id])
                edu_starts_sentence.append(token_id == 0)
                sent_id = next_sent_id
                token_id = next_token_id

            data["tokenized_edu_strings"] = tokenized_edu_strings
            data["edu_starts_sentence"] = edu_starts_sentence

            doc = Doc.from_data(data)
            dataset.append(doc)

        return dataset


class InstrDT(Dataset):
    relation_vocab = vocab(
        Counter(
            [
                "preparation:act",
                "joint",
                "general:specific",
                "criterion:act",
                "goal:act",
                "act:goal",
                "textualorganization",
                "topic-change?",
                "step1:step2",
                "disjunction",
                "contrast1:contrast2",
                "co-temp1:co-temp2",
                "act:reason",
                "act:criterion",
                "cause:effect",
                "comparision",
                "reason:act",
                "act:preparation",
                "situation:circumstance",
                "same-unit",
                "object:attribute",
                "effect:cause",
                "prescribe-act:wrong-act",
                "indeterminate",
                "specific:general",
                "before:after",
                "set:member",
                "situation:obstacle",
                "wrong-act:prescribe-act",
                "act:constraint",
                "circumstance:situation",
                "act:side-effect",
                "obstacle:situation",
                "after:before",
                "side-effect:act",
                "wrong-act:criterion",
                "attribute:object",
                "criterion:wrong-act",
                "constraint:act",
            ]
        ),
        specials=["<pad>"],
    )

    def preprocess(self, raw_dataset: List[Dict]):
        dataset = []
        for data in raw_dataset:
            rst_tree = RSTTree.fromstring(data["rst_tree"])
            if not RSTTree.is_valid_tree(rst_tree):
                continue
            assert RSTTree.check_relation(rst_tree, self.relation_vocab)
            bi_rst_tree = RSTTree.binarize(rst_tree)
            attach_tree = RSTTree.convert_to_attach(bi_rst_tree)
            data["attach_tree"] = attach_tree
            assert bi_rst_tree == AttachTree.convert_to_rst(attach_tree)

            doc = Doc.from_data(data)
            dataset.append(doc)

        return dataset
