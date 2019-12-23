import tensorflow as tf
from tensorflow.keras.activations import relu

class GadgetAnd:
    def __init__(self, trainable = False):
        if not trainable:
            self.W = tf.constant([[1.0], [1.0]], dtype=tf.float32)
            self.b = tf.constant([-1.0], dtype=tf.float32)
        else:
            initializer = tf.initializers.GlorotNormal()
            self.W = tf.Variable(initializer([2, 1]))
            self.b = tf.Variable(tf.zeros([1], dtype = tf.float32))

    def __call__(self, X):
        """
        0---0
          X   >0
        0---0
        """
        return self._cell(X)

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32),)
    )
    def _cell(self, X):
        H = X@self.W + self.b
        return relu(H)
    
class GadgetOr:
    def __init__(self, trainable = False):
        if not trainable:
            self.W = tf.constant([[1.0],[1.0]], dtype=tf.float32)
            self.b = tf.constant([0.0], dtype=tf.float32)
        else:
            initializer = tf.initializers.GlorotNormal()
            self.W = tf.Variable(initializer([2, 1]))
            self.b = tf.Variable(tf.zeros([1], dtype = tf.float32))

    def __call__(self, X):
        """
        """
        return self._cell(X)

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32),)
    )
    def _cell(self, X):
        H = X@self.W + self.b
        return relu(H)    

class GadgetXor:
    def __init__(self, trainable = False):
        if not trainable:
            self.W_1 = tf.constant([[1.], [1.]], dtype=tf.float32)
            self.b_1 = tf.constant([-1.5], dtype=tf.float32)
            self.W_skip = tf.constant([[1.], [1.]], dtype=tf.float32)
            self.b_2 = tf.constant([-0.5], dtype=tf.float32)
            self.W_2 = tf.constant([[-3.0]], dtype = tf.float32)
        else:
            initializer = tf.initializers.GlorotNormal()
            self.W_1 = tf.Variable(initializer([2, 1]))
            self.b_1 = tf.Variable(tf.zeros([1], dtype = tf.float32))
            self.W_skip = tf.Variable(initializer([2,1]))
            self.b_2 = tf.Variable(tf.zeros([1], dtype = tf.float32))
            self.W_2 = tf.Variable(initializer([1, 1]))

    def __call__(self, X):
        return self._cell(X)
    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32),)
    )
    def _cell(self, X):
        H = relu(X@self.W_1 + self.b_1)
        return relu(X@self.W_skip + H@self.W_2 + self.b_2)

if __name__ == "__main__":
    and_layer = GadgetXor(True)
    input = tf.Variable([[1,0]], dtype=tf.float32)
    print(and_layer(input))

