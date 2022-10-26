from typing import List, Union

import torch
from torchmetrics import Metric

from data.tree import AttachTree, RSTTree


class Parseval(Metric):
    def __init__(self):
        super(Parseval, self).__init__(compute_on_step=False)
        self.eval_types = ["span", "nuc", "rel", "full"]
        for eval_type in self.eval_types:
            self.add_state(
                "match_{}".format(eval_type),
                default=torch.tensor(0),
                dist_reduce_fx="sum",
            )

        self.add_state("pred", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("gold", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_trees: List[Union[RSTTree, AttachTree]],
        gold_trees: List[Union[RSTTree, AttachTree]],
    ):
        assert len(pred_trees) == len(gold_trees)

        for pred_tree, gold_tree in zip(pred_trees, gold_trees):
            # convert tree
            pred_tree = self.convert_tree(pred_tree)
            gold_tree = self.convert_tree(gold_tree)
            # get brackets
            pred_brackets = pred_tree.get_brackets(self.eval_types)
            gold_brackets = gold_tree.get_brackets(self.eval_types)
            # count brackets
            pred_cnt = len(pred_brackets["span"])
            gold_cnt = len(gold_brackets["span"])
            assert pred_cnt == gold_cnt
            self.pred += pred_cnt
            self.gold += gold_cnt

            self.match_span += len(
                [bracket for bracket in pred_brackets["span"] if bracket in gold_brackets["span"]]
            )
            self.match_nuc += len(
                [bracket for bracket in pred_brackets["nuc"] if bracket in gold_brackets["nuc"]]
            )
            self.match_rel += len(
                [bracket for bracket in pred_brackets["rel"] if bracket in gold_brackets["rel"]]
            )
            self.match_full += len(
                [bracket for bracket in pred_brackets["full"] if bracket in gold_brackets["full"]]
            )

    def compute(self):
        metric_name = self.__class__.__name__
        return {
            "{}-S".format(metric_name): self.match_span / self.pred,
            "{}-N".format(metric_name): self.match_nuc / self.pred,
            "{}-R".format(metric_name): self.match_rel / self.pred,
            "{}-F".format(metric_name): self.match_full / self.pred,
        }

    def convert_tree(self, tree: Union[RSTTree, AttachTree]):
        raise NotImplementedError
