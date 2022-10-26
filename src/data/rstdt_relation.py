from data.tree import RSTTree


def re_categorize(tree: RSTTree):
    def helper(node):
        if not isinstance(node, RSTTree):
            return node

        label = node.label()
        if label not in ["ROOT", "text"]:
            nuc, rel = node.label().split(":", maxsplit=1)
            while rel[-2:] in ["-s", "-e", "-n"]:
                rel = rel[:-2]

            rel = RELATION_TABLE[rel]
            label = ":".join([nuc, rel])

        return RSTTree(label, [helper(child) for child in node])

    assert isinstance(tree, RSTTree)
    return helper(tree)


RELATION_TABLE = {
    "ROOT": "ROOT",
    "span": "span",
    "attribution": "Attribution",
    "attribution-negative": "Attribution",
    "background": "Background",
    "circumstance": "Background",
    "cause": "Cause",
    "result": "Cause",
    "cause-result": "Cause",
    "consequence": "Cause",
    "comparison": "Comparison",
    "preference": "Comparison",
    "analogy": "Comparison",
    "proportion": "Comparison",
    "condition": "Condition",
    "hypothetical": "Condition",
    "contingency": "Condition",
    "otherwise": "Condition",
    "contrast": "Contrast",
    "concession": "Contrast",
    "antithesis": "Contrast",
    "elaboration-additional": "Elaboration",
    "elaboration-general-specific": "Elaboration",
    "elaboration-part-whole": "Elaboration",
    "elaboration-process-step": "Elaboration",
    "elaboration-object-attribute": "Elaboration",
    "elaboration-set-member": "Elaboration",
    "example": "Elaboration",
    "definition": "Elaboration",
    "enablement": "Enablement",
    "purpose": "Enablement",
    "evaluation": "Evaluation",
    "interpretation": "Evaluation",
    "conclusion": "Evaluation",
    "comment": "Evaluation",
    "evidence": "Explanation",
    "explanation-argumentative": "Explanation",
    "reason": "Explanation",
    "list": "Joint",
    "disjunction": "Joint",
    "manner": "Manner-Means",
    "means": "Manner-Means",
    "problem-solution": "Topic-Comment",
    "question-answer": "Topic-Comment",
    "statement-response": "Topic-Comment",
    "topic-comment": "Topic-Comment",
    "comment-topic": "Topic-Comment",
    "rhetorical-question": "Topic-Comment",
    "summary": "Summary",
    "restatement": "Summary",
    "temporal-before": "Temporal",
    "temporal-after": "Temporal",
    "temporal-same-time": "Temporal",
    "sequence": "Temporal",
    "inverted-sequence": "Temporal",
    "topic-shift": "Topic-Change",
    "topic-drift": "Topic-Change",
    "textualorganization": "Textual-organization",
    "same-unit": "Same-unit",
}
