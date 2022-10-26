from typing import List


class EDU(object):
    def __init__(
        self,
        edu_string: str,
        tokens: List[str],
        sent_idx: int,
        para_idx: int,
        is_start_sent: bool,
        is_end_sent: bool,
        is_start_para: bool,
        is_end_para: bool,
        is_start_doc: bool,
        is_end_doc: bool,
    ):
        self.edu_string = edu_string
        self.tokens = tokens
        self.sent_idx = sent_idx
        self.para_idx = para_idx
        self.is_start_sent = is_start_sent
        self.is_end_sent = is_end_sent
        self.is_start_para = is_start_para
        self.is_end_para = is_end_para
        self.is_start_doc = is_start_doc
        self.is_end_doc = is_end_doc

    def __repr__(self):
        return 'EDU(edu_string: "{}", tokens: {}, flags: [{}, {}, {}, {}, {}, {}])'.format(
            self.edu_string,
            self.tokens,
            self.is_start_sent,
            self.is_end_sent,
            self.is_start_para,
            self.is_end_para,
            self.is_start_doc,
            self.is_end_doc,
        )
