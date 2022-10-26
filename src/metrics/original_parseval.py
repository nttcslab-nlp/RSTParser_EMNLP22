from typing import Union

from data.tree import AttachTree, RSTTree
from metrics import Parseval


class OriginalParseval(Parseval):
    def __init__(self):
        super(OriginalParseval, self).__init__()

    def convert_tree(self, tree: Union[RSTTree, AttachTree]):
        # RSTTree -> AttachTree
        if isinstance(tree, RSTTree):
            tree = RSTTree.convert_to_attach(tree)

        return tree
