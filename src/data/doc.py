from typing import List, Union

from spacy.lang.en import English

from data.edu import EDU
from data.tree import AttachTree, RSTTree

nlp = English()


class Doc(object):
    def __init__(self, edus: List[EDU], tree: Union[RSTTree, AttachTree], doc_id: str):
        self.edus = edus
        self.tree = tree
        self.doc_id = doc_id
        self.inputs = None  # numericalized edus

    def __repr__(self):
        return 'Doc(doc_id: "{}", tree: {}, edus: {})'.format(
            self.doc_id, type(self.tree).__name__, [edu.edu_string for edu in self.edus]
        )

    def get_edu_strings(self):
        return [edu.edu_string for edu in self.edus]

    @classmethod
    def from_data(cls, data: dict):
        assert "doc_id" in data
        doc_id = data["doc_id"]

        assert "attach_tree" in data
        tree = data["attach_tree"]

        assert "edu_strings" in data
        edu_strings = data["edu_strings"]  # non-tokenized

        assert "edu_starts_sentence" in data
        assert "edu_starts_paragraph" in data
        edu_starts_sentence = data["edu_starts_sentence"]
        edu_starts_paragraph = data["edu_starts_paragraph"]
        edu_ends_sentence = edu_starts_sentence[1:] + [True]
        edu_ends_paragraph = edu_starts_paragraph[1:] + [True]

        if "tokenized_edu_strings" in data:
            tokenized_edu_strings = data["tokenized_edu_strings"]
        else:
            tokenized_edu_strings = [[token.text for token in nlp(edu)] for edu in edu_strings]

        edus = []
        sent_idx, para_idx = 0, 0
        for edu_idx, edu_string in enumerate(edu_strings):
            edu_tokens = tokenized_edu_strings[edu_idx]

            is_start_sent = edu_starts_sentence[edu_idx]
            is_end_sent = edu_ends_sentence[edu_idx]

            is_start_para = edu_starts_paragraph[edu_idx]
            is_end_para = edu_ends_paragraph[edu_idx]

            is_start_doc = True if edu_idx == 0 else False
            is_end_doc = True if edu_idx == len(edu_string) - 1 else False

            if is_start_sent:
                sent_idx += 1
            if is_start_para:
                para_idx += 1

            edu = EDU(
                edu_string,
                edu_tokens,
                sent_idx,
                para_idx,
                is_start_sent,
                is_end_sent,
                is_start_para,
                is_end_para,
                is_start_doc,
                is_end_doc,
            )
            edus.append(edu)

        return cls(edus, tree, doc_id)
