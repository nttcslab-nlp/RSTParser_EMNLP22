class OrganizationFeature:
    @staticmethod
    def IsSameSent(edus_a, edus_b):
        if edus_a == [] or edus_b == []:
            return False
        return edus_a[0].sent_idx == edus_b[-1].sent_idx

    @staticmethod
    def IsContinueSent(edus_a, edus_b):
        if edus_a == [] or edus_b == []:
            return False
        return edus_a[-1].sent_idx == edus_b[0].sent_idx

    @staticmethod
    def IsSamePara(edus_a, edus_b):
        if edus_a == [] or edus_b == []:
            return False
        return edus_a[0].para_idx == edus_b[-1].para_idx

    @staticmethod
    def IsContinuePara(edus_a, edus_b):
        if edus_a == [] or edus_b == []:
            return False
        return edus_a[-1].para_idx == edus_b[0].para_idx

    @staticmethod
    def IsStartSent(edus):
        if edus == []:
            return False
        return edus[0].is_start_sent

    @staticmethod
    def IsStartPara(edus):
        if edus == []:
            return False
        return edus[0].is_start_para

    @staticmethod
    def IsStartDoc(edus):
        if edus == []:
            return False
        return edus[0].is_start_doc

    @staticmethod
    def IsEndSent(edus):
        if edus == []:
            return False
        return edus[-1].is_end_sent

    @staticmethod
    def IsEndPara(edus):
        if edus == []:
            return False
        return edus[-1].is_end_para

    @staticmethod
    def IsEndDoc(edus):
        if edus == []:
            return False
        return edus[-1].is_end_doc
