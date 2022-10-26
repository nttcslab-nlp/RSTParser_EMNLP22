from typing import Union

from data.tree import AttachTree, RSTTree
from metrics import Parseval


class RSTParseval(Parseval):
    def __init__(self):
        super(RSTParseval, self).__init__()

    def convert_tree(self, tree: Union[RSTTree, AttachTree]):
        # AttachTree -> RSTTree
        if isinstance(tree, AttachTree):
            tree = AttachTree.convert_to_rst(tree)

        return tree
