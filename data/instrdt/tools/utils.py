import re
from nltk import Tree

TREE_PRINT_MARGIN = 1000000000


def divide_rst_tree_str(rst_tree_str: str):
    depth = 0
    start, end = 0, None
    for i, c in enumerate(rst_tree_str):
        if c == "\n":
            continue

        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1

        assert depth >= 0

        if depth == 0:
            end = i + 1
            yield rst_tree_str[start:end]
            start = end


def extract_edus_from_rst_tree_str(rst_tree_str: str):
    edus = re.findall(r"\(text (.*?)\)", rst_tree_str)
    return edus


def remove_edu_from_rst_tree_str(rst_tree_str: str):
    rst_tree_str = re.sub(r"<EDU>(.*?)</EDU>", "EDU_TEXT", rst_tree_str)
    return rst_tree_str


def fix_relation_label(tree: Tree):
    for tp in tree.treepositions():
        node = tree[tp]
        if not isinstance(node, Tree):
            continue
        if node.label() == "text":
            continue

        childs = [c for c in node]
        if len(childs) == 1 and childs[0].label() == "text":
            continue

        rel_labels = [c.label().split(":", maxsplit=1)[1] for c in childs]
        nuc_labels = [c.label().split(":", maxsplit=1)[0] for c in childs]
        assert all([l == rel_labels[0] for l in rel_labels])

        if all([n == "nucleus" for n in nuc_labels]):
            # N-N
            continue
        else:
            for n, child in zip(nuc_labels, childs):
                if n == "nucleus":
                    child.set_label("nucleus:span")
    return


def binarize(tree: Tree, top: bool = True):
    if top:
        assert tree.label() == "ROOT"
    if len(tree) == 1:
        # End of recursion
        return tree
    elif len(tree) == 2:
        # Binary structure
        left_tree = binarize(tree[0], top=False)
        right_tree = binarize(tree[1], top=False)
    else:
        # Non-Binary structure
        labels = [tree[i].label() for i in range(len(tree))]
        is_polynuclear = all(map(lambda x: x == labels[0], labels))
        if is_polynuclear:
            # Polynuclear relation label such as:
            # same-unit, list, etc...
            # -> convert to right heavy structure
            left_tree = binarize(tree[0], top=False)
            right_tree = binarize(
                Tree(tree[0].label(), [tree[i] for i in range(1, len(tree))]), top=False
            )
        else:
            # Non Binary structure without Polynuclear label
            # S/N/S -> left heavy
            left_tree = binarize(Tree("nucleus:span", [tree[0], tree[1]]), top=False)
            right_tree = binarize(tree[2], top=False)

    return Tree(tree.label(), [left_tree, right_tree])


def is_binary(tree: Tree):
    if not isinstance(tree, Tree):
        return True
    elif len(tree) > 2:
        return False
    else:
        return all([is_binary(child) for child in tree])


def attach_to_rst(tree: Tree, label: str = "ROOT"):
    if len(tree) == 1:
        return Tree(label, [tree])
    nuc_label, rel_label = tree.label().split(":", maxsplit=1)
    if len(nuc_label.split("-")) == 1:
        raise ValueError("Invalid nucleus label: {}".format(nuc_label))

    l_nuc, r_nuc = nuc_label.split("-")
    if nuc_label == "nucleus-satellite":
        l_rel = "span"
        r_rel = rel_label
    elif nuc_label == "satellite-nucleus":
        l_rel = rel_label
        r_rel = "span"
    elif nuc_label == "nucleus-nucleus":
        l_rel = r_rel = rel_label
    else:
        raise ValueError("Unkwon Nucleus label: {}".format(nuc_label))

    return Tree(
        label,
        [
            attach_to_rst(tree[0], ":".join([l_nuc, l_rel])),
            attach_to_rst(tree[1], ":".join([r_nuc, r_rel])),
        ],
    )


def rst_to_attach(rst_tree: Tree):
    if len(rst_tree) == 1:
        return rst_tree[0]

    l_nuc, l_rel = rst_tree[0].label().split(":", maxsplit=1)
    r_nuc, r_rel = rst_tree[1].label().split(":", maxsplit=1)
    nuc = "-".join([l_nuc, r_nuc])
    rel = l_rel if l_nuc == "satellite" else r_rel
    label = ":".join([nuc, rel])
    return Tree(label, [rst_to_attach(child) for child in rst_tree])


def re_assign_edu_idx(tree: Tree):
    for idx, tp in enumerate(tree.treepositions("leave")):
        lp = tp[:-1]
        tree[lp] = Tree("text", [str(idx)])


def is_nucleus(node: Tree):
    label = node.label()
    if label in ["text", "ROOT"]:
        return False

    nuc = label.split(":")[0]
    assert nuc in ["nucleus", "satellite"]
    return nuc == "nucleus"


def is_satellite(node: Tree):
    return not is_nucleus(node)


def is_multi_nucleus(node: Tree):
    if isinstance(node, str):
        return False
    label = node.label()
    if label in ["text"]:
        return False
    if len(node) <= 2:
        return False

    return True
