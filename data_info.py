class DataInfo:
    def __init__(self, args):
        self.__dict__ = args

    def __contains__(self, item):
        return item in self.__dict__