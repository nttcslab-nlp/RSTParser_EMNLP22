from models.parser.parser_base import ParserBase
from models.parser.shift_reduce_parser_base import ShiftReduceParserBase
from models.parser.shift_reduce_parser_v1 import ShiftReduceParserV1
from models.parser.shift_reduce_parser_v2 import ShiftReduceParserV2
from models.parser.shift_reduce_parser_v3 import ShiftReduceParserV3
from models.parser.top_down_parser_base import TopDownParserBase
from models.parser.top_down_parser_v1 import TopDownParserV1
from models.parser.top_down_parser_v2 import TopDownParserV2


class Parsers:
    parser_dict = {
        "top_down_v1": TopDownParserV1,
        "top_down_v2": TopDownParserV2,
        "shift_reduce_v1": ShiftReduceParserV1,
        "shift_reduce_v2": ShiftReduceParserV2,
        "shift_reduce_v3": ShiftReduceParserV3,
    }

    @classmethod
    def from_config(cls, config, classifier):
        parser_type = config.model_type
        parser = cls.parser_dict[parser_type](classifier)
        return parser


__all__ = [
    "ParserBase",
    "TopDownParserBase",
    "TopDownParserV1",
    "TopDownParserV2",
    "ShiftReduceParserBase",
    "ShiftReduceParserV1",
    "ShiftReduceParserV2",
    "ShiftReduceParserV3",
]
