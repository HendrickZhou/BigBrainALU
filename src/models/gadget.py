import tensorflow as tf


class GadgetAnd:
    def __init__(self, trainable = False):
        if not trainable:
            #self.W1 = [[1, 1]]
            #self.b1 = [[]]
            #self.W2 = 
            #self.b2 = 
            pass
        else:
            # init to random weights
            pass

    def __call__(self, X):
        """
        0---0
          X   >0
        0---0
        """
        return self._cell(X)

    @tf.function
    def _cell(self, X):
        H = X*self.W1 + self.b1
        return H*self.W2 + self.b2

        

class GadgetOr:
    def __init__(self):
        pass


class GadgetXor:
    def __init__(self):
        pass

if __name__ == "__main__":
    and_layer = GadgetAnd()
    print(and_layer([[1,0]]))

