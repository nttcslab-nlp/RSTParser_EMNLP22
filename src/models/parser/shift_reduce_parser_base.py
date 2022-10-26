from typing import Tuple

import torch

from data.doc import Doc
from data.tree import AttachTree
from models.classifier import ShiftReduceClassifierBase
from models.parser import ParserBase
from models.parser.organization_feature import OrganizationFeature as OrgFeat
from models.parser.shift_reduce_state import ShiftReduceState


class ShiftReduceParserBase(ParserBase):
    """Basic shift-reduce parser."""

    def __init__(self, classifier: ShiftReduceClassifierBase):
        super(ShiftReduceParserBase, self).__init__(classifier)
        assert isinstance(classifier, ShiftReduceClassifierBase)

    def select_action_and_labels(self, bert_output, span, feat, state, gold_act=None):
        raise NotImplementedError

    def parse(self, doc: Doc):
        bert_output = self.classifier.encoder(doc)
        n_edus = len(doc.edus)

        state = ShiftReduceState(n_edus)
        while not state.is_end():
            s1, s2, q1 = state.get_state()
            span = {"s1": s1, "s2": s2, "q1": q1}
            feat = {"org": self.get_organization_features(s1, s2, q1, doc, self.classifier.device)}

            # predict action and labels
            act, nuc, rel = self.select_action_and_labels(bert_output, span, feat, state)

            # update stack and queue
            state.operate(act, nuc, rel)

        tree = state.get_tree()
        return tree

    def parse_topk(self, doc: Doc, k: int):
        return self.BEAM(doc, k)

    def BEAM(self, doc: Doc, top_k: int):
        raise NotImplementedError

    def parse_with_naked_tree(self, doc: Doc, naked_tree: AttachTree):
        return self.labeling_to_naked_tree(doc, naked_tree)

    def labeling_to_naked_tree(self, doc: Doc, tree: AttachTree):
        bert_output = self.classifier.encoder(doc)
        n_edus = len(doc.edus)

        act_list, _, _ = self.generate_action_sequence(tree)

        state = ShiftReduceState(n_edus)
        for gold_act in act_list:
            s1, s2, q1 = state.get_state()
            span = {"s1": s1, "s2": s2, "q1": q1}
            feat = {"org": self.get_organization_features(s1, s2, q1, doc, self.classifier.device)}

            # predict action and labels
            act, nuc, rel = self.select_action_and_labels(
                bert_output, span, feat, state, gold_act=gold_act
            )

            # update stack and queue
            state.operate(gold_act, nuc, rel)

        tree = state.get_tree()
        return tree

    def generate_action_sequence(self, tree: AttachTree):
        act_list, nuc_list, rel_list = [], [], []
        for tp in tree.treepositions("postorder"):
            node = tree[tp]
            if not isinstance(node, AttachTree):
                continue

            label = node.label()

            if len(node) == 1 and label == "text":
                # terminal node
                act_list.append("shift")
                nuc_list.append("<pad>")
                rel_list.append("<pad>")
            elif len(node) == 2:
                # non-terminal node
                nuc, rel = node.label().split(":", maxsplit=1)
                act_list.append("reduce")
                nuc_list.append(nuc)
                rel_list.append(rel)
            else:
                raise ValueError("Input tree is not binarized.")

        return act_list, nuc_list, rel_list

    def get_organization_features(
        self, s1: Tuple[int], s2: Tuple[int], q1: Tuple[int], doc: Doc, device=None
    ):
        # span == (-1, -1) -> edus = []
        edus = doc.edus
        s1_edus = edus[slice(*s1)]
        s2_edus = edus[slice(*s2)]
        q1_edus = edus[slice(*q1)]

        # init features
        features = []

        if not self.classifier.disable_org_sent:
            # for Stack 1 and Stack2
            features.append(OrgFeat.IsSameSent(s2_edus, s1_edus))
            features.append(OrgFeat.IsContinueSent(s2_edus, s1_edus))

            # for Stack 1 and Queue 1
            features.append(OrgFeat.IsSameSent(s1_edus, q1_edus))
            features.append(OrgFeat.IsContinueSent(s1_edus, q1_edus))

            # for Stack 1, 2 and Queue 1
            features.append(
                OrgFeat.IsSameSent(s2_edus, s1_edus) & OrgFeat.IsSameSent(s1_edus, q1_edus)
            )

            # starts and ends a sentence
            features.append(OrgFeat.IsStartSent(s1_edus))
            features.append(OrgFeat.IsStartSent(s2_edus))
            features.append(OrgFeat.IsStartSent(q1_edus))
            features.append(OrgFeat.IsEndSent(s1_edus))
            features.append(OrgFeat.IsEndSent(s2_edus))
            features.append(OrgFeat.IsEndSent(q1_edus))

            # starts and ends a document
            features.append(OrgFeat.IsStartDoc(s1_edus))
            features.append(OrgFeat.IsStartDoc(s2_edus))
            features.append(OrgFeat.IsStartDoc(q1_edus))
            features.append(OrgFeat.IsEndDoc(s1_edus))
            features.append(OrgFeat.IsEndDoc(s2_edus))
            features.append(OrgFeat.IsEndDoc(q1_edus))

        if not self.classifier.disable_org_para:
            # for Stack 1 and Stack2
            features.append(OrgFeat.IsSamePara(s2_edus, s1_edus))
            features.append(OrgFeat.IsContinuePara(s2_edus, s1_edus))

            # for Stack 1 and Queue 1
            features.append(OrgFeat.IsSamePara(s1_edus, q1_edus))
            features.append(OrgFeat.IsContinuePara(s1_edus, q1_edus))

            # for Stack 1, 2 and Queue 1
            features.append(
                OrgFeat.IsSamePara(s2_edus, s1_edus) & OrgFeat.IsSamePara(s1_edus, q1_edus)
            )

            # starts and ends a paragraph
            features.append(OrgFeat.IsStartPara(s1_edus))
            features.append(OrgFeat.IsStartPara(s2_edus))
            features.append(OrgFeat.IsStartPara(q1_edus))
            features.append(OrgFeat.IsEndPara(s1_edus))
            features.append(OrgFeat.IsEndPara(s2_edus))
            features.append(OrgFeat.IsEndPara(q1_edus))

        # convert to index
        bias = torch.tensor([2 * i for i in range(len(features))], dtype=torch.long, device=device)
        features = torch.tensor(features, dtype=torch.long, device=device)
        return bias + features
