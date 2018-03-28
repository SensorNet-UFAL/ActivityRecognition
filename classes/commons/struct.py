class Struct(object):
    def __init__(self, entries):
        self.__dict__.update(entries)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__.__getitem__(key)
