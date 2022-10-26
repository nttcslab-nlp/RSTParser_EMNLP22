import random
from typing import Optional

from data.dataset import Dataset
from data.tree import AttachTree, RSTTree
from models.classifier import TopDownClassifierBase
from models.parser import TopDownParserBase
from models.parser.utils import batch_iter


class TopDownParserV1(TopDownParserBase):
    """This basic top-down parser individually predicts span, nucleus and relation."""

    def __init__(self, classifier: TopDownClassifierBase):
        super(TopDownParserV1, self).__init__(classifier)
        assert isinstance(classifier, TopDownClassifierBase)

    def generate_training_samples(
        self,
        dataset: Dataset,
        unit_type: str,
        batch_size: Optional[int] = None,
    ):
        nuc_vocab = self.classifier.nuc_vocab
        rel_vocab = self.classifier.rel_vocab

        samples = []
        for doc in dataset:
            tree = doc.tree
            if isinstance(tree, RSTTree):
                tree = RSTTree.convert_to_attach(tree)

            xs, ys, fs = [], [], []
            for tp in tree.treepositions():
                node = tree[tp]
                if not isinstance(node, AttachTree):
                    continue
                label = node.label()
                if label == "text":
                    continue

                leaves = node.leaves()
                span = (int(leaves[0]), int(leaves[-1]) + 1)
                split_idx = int(node[1].leaves()[0])

                nuc, rel = label.split(":", maxsplit=1)
                nuc_idx = nuc_vocab[nuc]
                rel_idx = rel_vocab[rel]

                org_feat = self.get_organization_features(
                    (span[0], split_idx), (split_idx, span[1]), doc
                )

                xs.append({"i": span[0], "j": span[1], "k": split_idx})
                ys.append({"spl": split_idx - span[0] - 1, "nuc": nuc_idx, "rel": rel_idx})
                fs.append({"org": org_feat})

            if unit_type == "document":
                samples.append({"doc": doc, "span": xs, "label": ys, "feat": fs})
            elif unit_type == "span":
                for x, y, f in zip(xs, ys, fs):
                    samples.append({"doc": doc, "span": x, "label": y, "feat": f})
            elif unit_type == "span_fast":
                assert batch_size > 1
                # should use Trainer.reload_dataloaders_every_n_epochs=1
                indices = list(range(len(xs)))
                random.shuffle(indices)
                xs = [xs[i] for i in indices]
                ys = [ys[i] for i in indices]
                fs = [fs[i] for i in indices]
                for feats in batch_iter(list(zip(xs, ys, fs)), batch_size):
                    b_xs, b_ys, b_fs = list(zip(*feats))
                    samples.append({"doc": doc, "span": b_xs, "label": b_ys, "feat": b_fs})
            else:
                raise ValueError("Invalid batch unit_type ({})".format(unit_type))

        return samples
