import tensorflow as tf

class NBitsClassifier(object):
    """
    only init will be used by factory registration
    """
    def __init__(self, bits=8):
        self.bits = bits

