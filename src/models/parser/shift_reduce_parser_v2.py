import random
from typing import Optional

import torch

from data.dataset import Dataset
from data.doc import Doc
from data.tree import RSTTree
from models.classifier import ShiftReduceClassifierBase
from models.parser import ShiftReduceParserBase
from models.parser.shift_reduce_state import ShiftReduceState
from models.parser.utils import batch_iter


class ShiftReduceParserV2(ShiftReduceParserBase):
    """This shift-reduce parser predicts nuc and relation labels by one classifier."""

    def __init__(self, classifier: ShiftReduceClassifierBase):
        super(ShiftReduceParserV2, self).__init__(classifier)
        assert isinstance(classifier, ShiftReduceClassifierBase)

    def generate_training_samples(
        self,
        dataset: Dataset,
        unit_type: str,
        batch_size: Optional[int] = None,
    ):
        act_vocab = self.classifier.act_vocab
        ful_vocab = self.classifier.ful_vocab

        samples = []
        for doc in dataset:
            tree = doc.tree
            if isinstance(tree, RSTTree):
                tree = RSTTree.convert_to_attach(tree)

            act_list, nuc_list, rel_list = self.generate_action_sequence(tree)
            xs, ys, fs = [], [], []
            state = ShiftReduceState(len(tree.leaves()))
            for act, nuc, rel in zip(act_list, nuc_list, rel_list):
                s1, s2, q1 = state.get_state()
                label = "<pad>" if nuc == rel == "<pad>" else ":".join([nuc, rel])
                act_idx = act_vocab[act]
                ful_idx = ful_vocab[label]
                org_feat = self.get_organization_features(s1, s2, q1, doc)
                xs.append({"s1": s1, "s2": s2, "q1": q1})
                ys.append({"act": act_idx, "ful": ful_idx})
                fs.append({"org": org_feat})
                state.operate(act, nuc, rel)

            assert tree == state.get_tree()

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

    def select_action_and_labels(self, bert_output, span, feat, state, gold_act=None):
        act_vocab = self.classifier.act_vocab
        ful_vocab = self.classifier.ful_vocab

        act_scores, ful_scores = self.classifier.predict(bert_output, span, feat)

        # select allowed action
        _, act_indices = torch.sort(act_scores, dim=0, descending=True)
        for act_idx in act_indices:
            act = act_vocab.lookup_token(act_idx)
            if act == "<pad>":
                continue

            if gold_act is not None:
                if act == gold_act:
                    break

            else:
                if state.is_allowed_action(act):
                    break

        rel, nuc = None, None
        if act != "shift":
            ful_scores[ful_vocab["<pad>"]] = -float("inf")
            label = ful_vocab.lookup_token(torch.argmax(ful_scores))
            nuc, rel = label.split(":", maxsplit=1)

        return act, nuc, rel

    def parse_topk(self, doc: Doc, k: int):
        return self.BEAM(doc, k)

    def BEAM(self, doc: Doc, top_k: int):
        act_vocab = self.classifier.act_vocab
        ful_vocab = self.classifier.ful_vocab

        bert_output = self.classifier.encoder(doc)
        n_edus = len(doc.edus)

        num_steps = 2 * n_edus - 1
        beams = [None] * (num_steps + 1)
        beams[0] = [ShiftReduceState(n_edus)]  # initial state

        for i in range(num_steps):
            buf = []
            for old_state in beams[i]:
                s1, s2, q1 = old_state.get_state()
                span = {"s1": s1, "s2": s2, "q1": q1}
                feat = {
                    "org": self.get_organization_features(s1, s2, q1, doc, self.classifier.device)
                }
                act_scores, ful_scores = self.classifier.predict(bert_output, span, feat)
                log_act_scores = act_scores.log_softmax(dim=0)

                for act in old_state.allowed_actions():
                    rel, nuc = None, None
                    if act != "shift":
                        ful_scores[ful_vocab["<pad>"]] = -float("inf")
                        label = ful_vocab.lookup_token(torch.argmax(ful_scores))
                        nuc, rel = label.split(":", maxsplit=1)

                    act_idx = act_vocab[act]
                    action_score = log_act_scores[act_idx].item()
                    new_state = old_state.copy()
                    new_state.operate(act, nuc, rel, score=action_score)
                    buf.append(new_state)

            buf = sorted(buf, key=lambda x: x.score, reverse=True)  # descending order

            tmp = {}
            beams[i + 1] = []
            for j, new_state in enumerate(buf):
                _state = new_state.get_state()
                score = new_state.score
                idx = (_state, score)
                if idx not in tmp:
                    tmp[idx] = new_state
                    beams[i + 1].append(new_state)
                else:
                    # print('duplicate')
                    # tmp[_state].merge_state(new_state)
                    pass

                if len(tmp) == top_k:
                    break

        trees = [x.get_tree() for x in beams[-1][:top_k]]
        return trees
