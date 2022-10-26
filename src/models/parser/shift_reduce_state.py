from typing import List, Tuple, Union

from data.edu import EDU
from data.tree import AttachTree


class ShiftReduceState(object):
    # TODO:
    # https://github.com/lianghuang3/lineardpparser/blob/master/code/newstate.py
    def __init__(self, n_edus: int):
        self.n_edus = n_edus
        self.stack = []
        self.queue = list(map(str, range(n_edus)[::-1]))
        self.score = 0

    def copy(self):
        # make new object
        x = ShiftReduceState(self.n_edus)
        # copy params
        x.stack = self.stack.copy()
        x.queue = self.queue.copy()
        x.score = self.score
        return x

    def operate(self, action: str, nuc: str, rel: str, score: float = 0):
        self.score = self.score + score

        if action == "shift":
            edu_idx = self.queue.pop()
            node = AttachTree("text", [edu_idx])
            self.stack.append(node)
        elif action == "reduce":
            r_node = self.stack.pop()
            l_node = self.stack.pop()
            label = ":".join([nuc, rel])
            new_node = AttachTree(label, [l_node, r_node])
            self.stack.append(new_node)
        else:
            raise ValueError("unexpected action: {}".format(action))

    def is_end(self):
        return len(self.stack) == 1 and len(self.queue) == 0

    def get_tree(self):
        if self.is_end():
            return self.stack[0]
        else:
            raise ValueError

    def get_state(self):
        def get_edu_span(node: Union[AttachTree, str]):
            if isinstance(node, AttachTree):
                leaves = node.leaves()
                span = (int(leaves[0]), int(leaves[-1]) + 1)
            else:
                edu_idx = node
                span = (int(edu_idx), int(edu_idx) + 1)

            return span

        # stack top1, top2
        s1 = get_edu_span(self.stack[-1]) if len(self.stack) > 0 else (-1, -1)
        s2 = get_edu_span(self.stack[-2]) if len(self.stack) > 1 else (-1, -1)
        # queue first
        q1 = get_edu_span(self.queue[-1]) if len(self.queue) > 0 else (-1, -1)
        return s1, s2, q1

    def allowed_actions(self):
        actions = []
        for act in ["shift", "reduce"]:
            if self.is_allowed_action(act):
                actions.append(act)

        return actions

    def is_allowed_action(self, action: str):
        if action == "shift":
            return len(self.queue) >= 1
        elif action == "reduce":
            return len(self.stack) >= 2
        else:
            raise ValueError

    def is_allowed_action_for_sentence_level_parse(self, action: str, edus: List[EDU]):
        s1, s2, q1 = self.get_state()
        if action == "shift":
            if len(self.queue) < 1:
                return False

            if len(self.stack) < 1:
                return True

            # shiftできない場合:
            # - s1とq1が異なる文に属しており，
            #   s1が不完全な文ノードであるならshiftできない．
            # * 判定のためにはs1, q1が存在する必要がある．
            if not self.in_same_sentence(s1, q1, edus):
                if self.is_part_of_sentence_tree(s1, edus):
                    return False

            return True

        elif action == "reduce":
            if len(self.stack) < 2:
                return False

            if len(self.queue) < 1:
                return True

            # reduceできない場合:
            # - s1とq1が同じ文に属しており，
            #   s1とs2が異なる文に属するならreduceできない．
            # * 判定のためにはs1,s2,q1が存在する必要がある．
            if self.in_same_sentence(s1, q1, edus):
                if not self.in_same_sentence(s1, s2, edus):
                    return False

            return True
        else:
            raise ValueError('Invalid action "{}"'.format(action))

    def in_same_sentence(self, q1: Tuple[int], s1: Tuple[int], edus: List[EDU]):
        if q1 == (-1, -1) or s1 == (-1, -1):
            raise ValueError
        # q1 is edu idx
        # s1 is edu span
        q1_edu = edus[q1[0]]
        s1_edu = edus[s1[1] - 1]  # most right edu of s1
        return q1_edu.sent_idx == s1_edu.sent_idx

    def is_part_of_sentence_tree(self, s: Tuple[int], edus: List[EDU]):
        if s == (-1, -1):
            raise ValueError
        # s is edu span
        s_left_edu = edus[s[0]]  # most left edu of s
        s_right_edu = edus[s[1] - 1]  # most right edu of s
        if s_left_edu.sent_idx != s_right_edu.sent_idx:
            # larger than sentence
            return False
        if s_left_edu.start_sent and s_right_edu.end_sent:
            # complete sentence
            return False

        # smaller than sentence
        return True
