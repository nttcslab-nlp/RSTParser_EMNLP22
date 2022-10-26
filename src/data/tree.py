from typing import List

from nltk import Tree
from torchtext.vocab import Vocab


class RSTTree(Tree):
    def __init__(self, label: str, children: List):
        super(RSTTree, self).__init__(label, children)
        self.nuc = None
        self.rel = None
        if label not in ["ROOT", "text"]:
            nuc, rel = label.split(":", maxsplit=1)

    @classmethod
    def binarize(cls, tree: Tree):
        def helper(node):
            if len(node) == 1:
                # End of recursion
                return node
            elif len(node) == 2:
                # Binary structure
                left_node = helper(node[0])
                right_node = helper(node[1])
            else:
                # Non-Binary structure
                labels = [node[i].label() for i in range(len(node))]
                is_polynuclear = all(map(lambda x: x == labels[0], labels))
                if is_polynuclear:
                    # Polynuclear relation label such as:
                    # same-unit, list, etc...
                    # -> convert to right heavy structure
                    left_node = helper(node[0])
                    right_node = helper(
                        cls(node[0].label(), [node[i] for i in range(1, len(node))])
                    )
                else:
                    # Non Binary structure without Polynuclear label
                    # S/N/S -> left heavy
                    left_node = helper(cls("nucleus:span", [node[0], node[1]]))
                    right_node = helper(node[2])

            return cls(node.label(), [left_node, right_node])

        assert isinstance(tree, RSTTree)
        return helper(tree)

    @classmethod
    def is_binary(cls, tree: Tree):
        def helper(node):
            if not isinstance(node, RSTTree):
                return True
            elif len(node) > 2:
                return False
            else:
                return all([helper(child) for child in node])

        assert isinstance(tree, RSTTree)
        return helper(tree)

    @classmethod
    def convert_to_attach(cls, tree: Tree):
        def helper(node):
            if len(node) == 1:
                edu_idx = node[0][0]
                return AttachTree("text", [edu_idx])

            l_nuc, l_rel = node[0].label().split(":", maxsplit=1)
            r_nuc, r_rel = node[1].label().split(":", maxsplit=1)
            nuc = "-".join([l_nuc, r_nuc])
            rel = l_rel if l_rel != "span" else r_rel
            label = ":".join([nuc, rel])
            return AttachTree(label, [helper(child) for child in node])

        assert RSTTree.is_binary(tree)
        assert isinstance(tree, RSTTree)
        return helper(tree)

    @classmethod
    def is_valid_tree(cls, tree: Tree):
        assert isinstance(tree, RSTTree)
        if len(tree) == 1:
            # (ROOT (text 0))
            return False

        return True

    @classmethod
    def check_relation(cls, tree: Tree, relation_vocab: Vocab):
        for tp in tree.treepositions():
            node = tree[tp]
            if not isinstance(node, RSTTree):
                continue

            label = tree.label()
            if label in ["ROOT", "text"]:
                continue

            nuc, rel = label.split(":", maxsplit=1)
            if rel not in relation_vocab:
                return False

        return True

    def get_brackets(self, eval_types: List[str]):
        brackets = {eval_type: [] for eval_type in eval_types}

        for tp in self.treepositions():
            node = self[tp]
            if not isinstance(node, RSTTree):
                continue  # EDU idx

            label = node.label()
            if label == "ROOT" and tp == ():
                continue  # ROOT node

            if label == "text" and len(node) == 1:
                continue  # leave node

            edu_indices = [int(idx) for idx in node.leaves()]
            span = (edu_indices[0], edu_indices[-1] + 1)
            ns, relation = label.split(":", maxsplit=1)

            if "full" in eval_types:
                brackets["full"].append((span, ns, relation))
            if "rel" in eval_types:
                brackets["rel"].append((span, relation))
            if "nuc" in eval_types:
                brackets["nuc"].append((span, ns))
            if "span" in eval_types:
                brackets["span"].append((span))

        return brackets


class AttachTree(Tree):
    def __init__(self, label: str, children: List):
        super(AttachTree, self).__init__(label, children)
        self.nuc = None
        self.rel = None
        if label == "text":
            # EDU node
            pass
        else:
            nuc, rel = label.split(":", maxsplit=1)

    @classmethod
    def convert_to_rst(cls, tree: Tree):
        def helper(node, label="ROOT"):
            if len(node) == 1:
                edu_idx = node[0]
                return RSTTree(label, [RSTTree("text", [edu_idx])])

            nuc, rel = node.label().split(":", maxsplit=1)
            if len(nuc.split("-")) == 1:
                raise ValueError("Invalid nucleus label: {}".format(nuc))
            l_nuc, r_nuc = nuc.split("-")
            if nuc == "nucleus-satellite":
                l_rel, r_rel = "span", rel
            elif nuc == "satellite-nucleus":
                l_rel, r_rel = rel, "span"
            elif nuc == "nucleus-nucleus":
                l_rel = r_rel = rel
            else:
                raise ValueError("Unkwon Nucleus label: {}".format(nuc))

            l_label = ":".join([l_nuc, l_rel])
            r_label = ":".join([r_nuc, r_rel])
            return RSTTree(label, [helper(node[0], l_label), helper(node[1], r_label)])

        assert isinstance(tree, AttachTree)
        return helper(tree)

    def get_brackets(self, eval_types: List[str]):
        brackets = {eval_type: [] for eval_type in eval_types}

        for tp in self.treepositions():
            node = self[tp]
            if not isinstance(node, AttachTree):
                continue  # EDU idx

            label = node.label()
            if label == "text" and len(node) == 1:
                continue  # leave node

            edu_indices = [int(idx) for idx in node.leaves()]
            span = (edu_indices[0], edu_indices[-1] + 1)
            ns, relation = label.split(":", maxsplit=1)

            if "full" in eval_types:
                brackets["full"].append((span, ns, relation))
            if "rel" in eval_types:
                brackets["rel"].append((span, relation))
            if "nuc" in eval_types:
                brackets["nuc"].append((span, ns))
            if "span" in eval_types:
                brackets["span"].append((span))

        return brackets
