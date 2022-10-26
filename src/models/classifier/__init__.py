from models.classifier.classifier_base import ClassifierBase
from models.classifier.shift_reduce_classifier_base import ShiftReduceClassifierBase
from models.classifier.shift_reduce_classifier_v1 import ShiftReduceClassifierV1
from models.classifier.shift_reduce_classifier_v2 import ShiftReduceClassifierV2
from models.classifier.shift_reduce_classifier_v3 import ShiftReduceClassifierV3
from models.classifier.top_down_classifier_base import TopDownClassifierBase
from models.classifier.top_down_classifier_v1 import TopDownClassifierV1
from models.classifier.top_down_classifier_v2 import TopDownClassifierV2


class Classifiers:
    classifier_dict = {
        "top_down_v1": TopDownClassifierV1,
        "top_down_v2": TopDownClassifierV2,
        "shift_reduce_v1": ShiftReduceClassifierV1,
        "shift_reduce_v2": ShiftReduceClassifierV2,
        "shift_reduce_v3": ShiftReduceClassifierV3,
    }

    @classmethod
    def from_config(cls, config):
        classifier_type = config.model_type
        classifier = cls.classifier_dict[classifier_type].from_config(config)
        return classifier


__all__ = [
    "ClassifierBase",
    "TopDownClassifierBase",
    "ShiftReduceClassifierBase",
    "TopDownClassifierV1",
    "TopDownClassifierV2",
    "ShiftReduceClassifierV1",
    "ShiftReduceClassifierV2",
    "ShiftReduceClassifierV3",
]
