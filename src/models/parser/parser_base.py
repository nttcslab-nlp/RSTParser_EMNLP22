import time
from typing import Union

from data.dataset import Dataset
from data.doc import Doc
from data.tree import AttachTree, RSTTree
from metrics import OriginalParseval
from models.classifier import ClassifierBase


class ParserBase(object):
    def __init__(self, classifier: ClassifierBase):
        self.classifier = classifier

    def parse_dataset(
        self,
        dataset,
        verbose: bool = False,
    ):
        if verbose:
            print("start parsing a dataset")
            ss = time.time()

        result = {"doc_id": [], "pred_tree": [], "gold_tree": []}
        for batch in dataset:
            assert batch.unit_type == "document"
            doc = batch.doc
            tree = self.parse(doc)

            result["doc_id"].append(doc.doc_id)
            result["pred_tree"].append(tree)
            result["gold_tree"].append(doc.tree)

        if verbose:
            print("elapsed time for all: {:.2f} [sec]".format(time.time() - ss))

        return result

    def parse(self, doc: Doc):
        raise NotImplementedError

    def parse_dataset_with_naked_tree(
        self,
        dataset,
        verbose: bool = False,
    ):
        if verbose:
            print("start parsing a dataset with naked tree")
            ss = time.time()

        result = {"doc_id": [], "pred_tree": [], "gold_tree": []}
        for batch in dataset:
            assert batch.unit_type == "document"
            doc = batch.doc
            tree = self.parse_with_naked_tree(doc, doc.tree)

            result["doc_id"].append(doc.doc_id)
            result["pred_tree"].append(tree)
            result["gold_tree"].append(doc.tree)

        if verbose:
            print("elapsed time for all: {:.2f} [sec]".format(time.time() - ss))

        return result

    def parse_with_naked_tree(
        self,
        doc: Doc,
        tree: Union[RSTTree, AttachTree],
    ):
        raise NotImplementedError

    def parse_dataset_topk(self, dataset, topk: int, verbose: bool = False):
        if verbose:
            print("start parsing a dataset with top-k (k={})".format(topk))
            ss = time.time()

        result = {"doc_id": [], "pred_tree": [], "pred_trees": [], "gold_tree": []}
        metric = OriginalParseval()
        for batch in dataset:
            assert batch.unit_type == "document"
            doc = batch.doc
            # if verbose:
            #     print('document id: ', doc.doc_id)
            #     print('- # of edus   : {}'.format(len(doc.edus)))
            #     s = time.time()

            trees = self.parse_topk(doc, topk)

            best_tree, best_score = None, -1
            for tree in trees:
                metric.update([tree], [doc.tree])
                scores = metric.compute()
                score = scores["OriginalParseval-F"].item()
                if score > best_score:
                    best_tree = tree
                    best_score = score

                metric.reset()

            # if verbose:
            #     print('- best score  : {:.2f}'.format(best_score))
            #     print('- elapsed time: {:.2f} [sec]'.format(time.time() - s))

            result["doc_id"].append(doc.doc_id)
            result["pred_tree"].append(best_tree)
            result["pred_trees"].append(trees)
            result["gold_tree"].append(doc.tree)

        if verbose:
            print("elapsed time for all: {:.2f} [sec]".format(time.time() - ss))

        return result

    def parse_topk(self, doc: Doc):
        raise NotImplementedError

    def generate_training_samples(cls, dataset: Dataset, level: str):
        raise NotImplementedError

    def get_organization_features(self):
        raise NotImplementedError
