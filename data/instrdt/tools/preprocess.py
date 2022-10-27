import argparse
from pathlib import Path
from nltk import ParentedTree, Tree
import json

from rstfinder.reformat_rst_trees import reformat_rst_tree
from utils import (
    extract_edus_from_rst_tree_str,
    remove_edu_from_rst_tree_str,
    divide_rst_tree_str,
    fix_relation_label,
    binarize,
    rst_to_attach,
    attach_to_rst,
    re_assign_edu_idx,
    TREE_PRINT_MARGIN,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="path to original instr-dt dataset (contains .out files)",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="path to output json file",
    )
    parser.add_argument(
        "--joint-with-nn",
        action="store_true",
        help="joint multiple trees in one document to single tree",
    )
    args = parser.parse_args()

    dataset = []
    num_docs = 0
    for file_path in args.input_dir.glob("*.out"):
        data = read_data(file_path, args.joint_with_nn)
        dataset.extend(data)
        num_docs += 1

    # check size
    assert num_docs == 176
    if args.joint_with_nn:
        assert len(dataset) == 176
    else:
        assert len(dataset) == 320

    save_json(args.output_file, dataset)
    return


def read_data(file_path, joint_with_nn=False):
    path_basename = file_path.name
    doc_id = file_path.name.rstrip(".out")
    edu_file = file_path.with_suffix(".out.edus")
    dis_file = file_path.with_suffix(".out.dis")

    edu_strings = read_edus(edu_file)
    rst_trees, _edu_strings = read_dis(dis_file, joint_with_nn)
    assert len(edu_strings) == len(_edu_strings)

    edu_starts_sentence = [True if edu.startswith("<s>") else False for edu in _edu_strings]
    edu_starts_paragraph = [False] * len(edu_strings)  # no annotated

    data = []
    edu_offset = 0
    for idx, rst_tree in enumerate(rst_trees):
        bi_rst_tree = binarize(rst_tree)
        attach_tree = rst_to_attach(bi_rst_tree)
        _bi_rst_tree = attach_to_rst(attach_tree)
        assert _bi_rst_tree == bi_rst_tree
        n_edus = len(rst_tree.leaves())
        assert n_edus == len(edu_strings[edu_offset : edu_offset + n_edus])
        data.append(
            {
                "path_basename": path_basename,
                "doc_id": doc_id + ".{}".format(idx + 1),
                "rst_tree": rst_tree.pformat(margin=TREE_PRINT_MARGIN),
                "binarised_rst_tree": bi_rst_tree.pformat(margin=TREE_PRINT_MARGIN),
                "attach_tree": attach_tree.pformat(margin=TREE_PRINT_MARGIN),
                "edu_strings": edu_strings[edu_offset : edu_offset + n_edus],
                "edu_starts_sentence": edu_starts_sentence[edu_offset : edu_offset + n_edus],
                "edu_starts_paragraph": edu_starts_paragraph[edu_offset : edu_offset + n_edus],
            }
        )
        edu_offset = edu_offset + n_edus

    assert len(edu_strings) == edu_offset

    return data


def read_edus(file_path):
    edus = []
    with open(file_path) as f:
        for line in f:
            edu = line.strip()
            edus.append(edu)

    return edus


def read_dis(file_path, joint_with_NN=False):
    trees = []
    with open(file_path) as f:
        rst_tree_str = f.read().strip()
        edus = extract_edus_from_rst_tree_str(rst_tree_str)
        rst_tree_str = remove_edu_from_rst_tree_str(rst_tree_str)

        for _rst_tree_str in divide_rst_tree_str(rst_tree_str):
            rst_tree = ParentedTree.fromstring(_rst_tree_str)
            reformat_rst_tree(rst_tree)
            rst_tree = Tree.convert(rst_tree)
            trees.append(rst_tree)

    if joint_with_NN and len(trees) != 1:
        for tree in trees:
            tree.set_label("nucleus:topic-change?")

        tree = Tree("ROOT", trees)
        re_assign_edu_idx(tree)
        trees = [tree]

    for tree in trees:
        fix_relation_label(tree)

    return trees, edus


def save_json(file_path, dataset):
    print('save into "{}" (consists of {} trees)'.format(file_path, len(dataset)))
    with open(file_path, "w") as f:
        json.dump(dataset, f)

    return


if __name__ == "__main__":
    main()
