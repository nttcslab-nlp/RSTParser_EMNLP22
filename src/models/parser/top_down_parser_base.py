from heapq import heappop, heappush
from typing import Tuple

import torch

from data.doc import Doc
from data.tree import AttachTree
from models.classifier import TopDownClassifierBase
from models.parser import ParserBase
from models.parser.organization_feature import OrganizationFeature as OrgFeat


class TopDownParserBase(ParserBase):
    """base class for top-down parser."""

    def __init__(self, classifier: TopDownClassifierBase):
        super(TopDownParserBase, self).__init__(classifier)
        assert isinstance(classifier, TopDownClassifierBase)

    def parse(self, doc: Doc):
        bert_output = self.classifier.encoder(doc)
        n_edus = len(doc.edus)
        span = (0, n_edus)

        def build_tree(span):
            if span[0] + 1 == span[1]:
                edu_idx = str(int(span[0]))
                return AttachTree("text", [edu_idx])

            k = self.classifier.predict_split(bert_output, span, doc, return_scores=False)
            # k = torch.argmax(split_scores) + span[0] + 1
            # k = k.item()
            label = self.classifier.predict_label(
                bert_output, (span[0], k), (k, span[1]), doc, return_scores=False
            )

            return AttachTree(label, [build_tree((span[0], k)), build_tree((k, span[1]))])

        tree = build_tree(span)
        return tree

    def parse_topk(self, doc: Doc, k: int):
        naked_trees = self.CKY(doc, k)
        labeled_trees = [self.labeling_to_naked_tree(doc, tree) for tree in naked_trees]
        return labeled_trees

    def CKY(self, doc: Doc, top_k: int):
        bert_output = self.classifier.encoder(doc)
        n_edus = len(doc.edus)

        # prepare org_indices for all spans
        span_to_org_indices = {}
        for i in range(n_edus):
            for j in range(i, n_edus + 1):
                span = (i, j)
                org_indices = []
                for k in range(i + 1, j):
                    org_indices.append(self.get_organization_features((i, k), (k, j), doc))

                span_to_org_indices[span] = org_indices

        # init tables
        CKY_table = [[[-1 for k in range(top_k)] for j in range(n_edus)] for i in range(n_edus)]
        count_table = [[-1 for j in range(n_edus)] for i in range(n_edus)]
        trace_table = [
            [[(-1, -1, -1) for k in range(top_k)] for j in range(n_edus)] for i in range(n_edus)
        ]

        def mult(log_split_scores, left_edu_idx, right_edu_idx):
            buf = []
            for c_idx, k in enumerate(range(left_edu_idx, right_edu_idx)):
                left_scores = CKY_table[left_edu_idx][k][: count_table[left_edu_idx][k]]
                right_scores = CKY_table[k + 1][right_edu_idx][: count_table[k + 1][right_edu_idx]]
                split_score = log_split_scores[k - left_edu_idx].item()
                for l, left_score in enumerate(left_scores):
                    for r, right_score in enumerate(right_scores):
                        score = left_score + right_score + split_score
                        buf.append(
                            {
                                "score": score,
                                "trace": (l, k + 1, r),
                            }
                        )

            sorted_buf = sorted(buf, key=lambda x: x["score"], reverse=True)
            top_k_buf = sorted_buf[:top_k]
            return top_k_buf

        def mult_with_heap(split_scores, left_edu_idx, right_edu_idx):
            i = left_edu_idx
            j = right_edu_idx
            heap = []
            for k in range(i, j):
                # 0. starts with an initial heap of the 1-best derivations
                left_scores = CKY_table[i][k]  # corresponding to (i, k)
                right_scores = CKY_table[k + 1][j]  # corresponding to (k, j)
                split_score = log_split_scores[k - i].item()
                l, r = 0, 0
                score = left_scores[l] + right_scores[r] + split_score
                item = (-score, (k, l, r))
                heappush(heap, item)

            top_k_scores = []
            heap_elm_dict = {}  # remenber heap items to avoid duplication
            for _ in range(top_k):
                if heap == []:
                    # no element in the heap
                    break

                # 1. extract-max from the heap
                elm = heappop(heap)
                best_score = elm[0]
                indices = elm[1]  # (k, l, r)
                k = indices[0]
                l = indices[1]
                r = indices[2]
                top_k_scores.append(
                    {
                        "score": -best_score,
                        "trace": (
                            l,
                            k + 1,
                            r,
                        ),  # k+1 edu index corresponding to split point
                    }
                )

                left_scores = CKY_table[i][k]
                right_scores = CKY_table[k + 1][j]
                split_score = log_split_scores[k - i].item()

                # 2. push the two "shoulders" into the heap
                if l + 1 < count_table[i][k]:
                    score = left_scores[l + 1] + right_scores[r] + split_score
                    item = (-score, (k, l + 1, r))
                    if item not in heap_elm_dict:
                        heappush(heap, item)
                        heap_elm_dict[item] = 0

                if r + 1 < count_table[k + 1][j]:
                    score = left_scores[l] + right_scores[r + 1] + split_score
                    item = (-score, (k, l, r + 1))
                    if item not in heap_elm_dict:
                        heappush(heap, item)
                        heap_elm_dict[item] = 0

            return top_k_scores

        # fill tables with DP
        for length in range(n_edus):
            for offset in range(n_edus):
                left_edu_idx, right_edu_idx = offset, offset + length
                if right_edu_idx > n_edus - 1:
                    # out of range
                    break

                if length == 0:
                    # init leaf node with log(0.0) == 1.0
                    CKY_table[left_edu_idx][right_edu_idx][0] = torch.tensor(1.0).log().item()
                    count_table[left_edu_idx][right_edu_idx] = 1
                    continue

                span = (left_edu_idx, right_edu_idx + 1)
                split_scores = self.classifier.predict_split_fast(
                    bert_output, span, doc, span_to_org_indices, return_scores=True
                )
                log_split_scores = split_scores.log_softmax(dim=0)

                # top_k_buf = mult(log_split_scores, left_edu_idx, right_edu_idx)
                top_k_buf = mult_with_heap(log_split_scores, left_edu_idx, right_edu_idx)
                scores = [(x["score"]) for x in top_k_buf]
                CKY_table[left_edu_idx][right_edu_idx][: len(top_k_buf)] = scores
                traces = [(x["trace"]) for x in top_k_buf]
                trace_table[left_edu_idx][right_edu_idx][: len(top_k_buf)] = traces
                count_table[left_edu_idx][right_edu_idx] = len(top_k_buf)

        def backtrace(table, left, right, index):
            assert right - left >= 1
            label = "nucleus-satellite:Elaboration"  # majority label
            if right - left == 1:
                edu_index = left
                return AttachTree("text", [str(edu_index)])

            l, k, r = table[left][right - 1][index]
            return AttachTree(label, [backtrace(table, left, k, l), backtrace(table, k, right, r)])

        # build tree by back-tracing
        trees = [
            backtrace(trace_table, 0, n_edus, index)
            for index in range(top_k)
            if trace_table[0][n_edus - 1][index] != (-1, -1, -1)
        ]
        return trees

    def parse_with_naked_tree(self, doc: Doc, naked_tree: AttachTree):
        return self.labeling_to_naked_tree(doc, naked_tree)

    def labeling_to_naked_tree(self, doc, naked_tree):
        bert_output = self.classifier.encoder(doc)
        n_edus = len(doc.edus)
        span = (0, n_edus)

        def build_tree(span, given_tree):
            if span[0] + 1 == span[1]:
                edu_idx = str(int(span[0]))
                return AttachTree("text", [edu_idx])

            # get a split point from given tree
            _, leaves = given_tree.label(), given_tree.leaves()
            given_span = (int(leaves[0]), int(leaves[-1]) + 1)
            assert given_span == span
            given_split_idx = int(given_tree[1].leaves()[0]) - span[0] - 1
            k = given_split_idx + span[0] + 1  # == given_tree[1].leves()[0]

            # predict labels
            label = self.classifier.predict_label(
                bert_output, (span[0], k), (k, span[1]), doc, return_scores=False
            )

            return AttachTree(
                label,
                [
                    build_tree((span[0], k), given_tree[0]),
                    build_tree((k, span[1]), given_tree[1]),
                ],
            )

        labeled_tree = build_tree(span, naked_tree)
        return labeled_tree

    def get_organization_features(
        self, left_span: Tuple[int], right_span: Tuple[int], doc: Doc, device=None
    ):
        edus = doc.edus
        left_edus = edus[slice(*left_span)]
        right_edus = edus[slice(*right_span)]

        features = []

        if not self.classifier.disable_org_sent:
            # same or continue sentence
            features.append(OrgFeat.IsSameSent(left_edus, right_edus))
            features.append(OrgFeat.IsContinueSent(left_edus, right_edus))

            # starts and ends a sentence
            features.append(OrgFeat.IsStartSent(left_edus))
            features.append(OrgFeat.IsStartSent(right_edus))
            features.append(OrgFeat.IsEndSent(left_edus))
            features.append(OrgFeat.IsEndSent(right_edus))
            features.append(OrgFeat.IsStartDoc(left_edus))
            features.append(OrgFeat.IsStartDoc(right_edus))
            features.append(OrgFeat.IsEndDoc(left_edus))
            features.append(OrgFeat.IsEndDoc(right_edus))

        if not self.classifier.disable_org_para:
            # same or continue sentence
            features.append(OrgFeat.IsSamePara(left_edus, right_edus))
            features.append(OrgFeat.IsContinuePara(left_edus, right_edus))

            # starts and ends a paragraph
            features.append(OrgFeat.IsStartPara(left_edus))
            features.append(OrgFeat.IsStartPara(right_edus))
            features.append(OrgFeat.IsEndPara(left_edus))
            features.append(OrgFeat.IsEndPara(right_edus))

        # convert to index
        bias = torch.tensor([2 * i for i in range(len(features))], dtype=torch.long, device=device)
        features = torch.tensor(features, dtype=torch.long, device=device)
        return bias + features
