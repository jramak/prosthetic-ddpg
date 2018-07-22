import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


def apply_activation(x, activation):
    if activation == 'selu':
        x = tf.nn.selu(x)
    elif activation == 'elu':
        x = tf.nn.elu(x)
    elif activation == 'relu':
        x = tf.nn.relu(x)
    elif activation == 'leaky_relu':
        x = tf.nn.leaky_relu(x)
    return x


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, activation='selu', layer_sizes=[64,64]):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.activation = activation
        self.layer_sizes = layer_sizes

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, self.layer_sizes[0])
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = apply_activation(x, self.activation)

            x = tf.layers.dense(x, self.layer_sizes[1])
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = apply_activation(x, self.activation)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True, activation='selu', layer_sizes=[64,64]):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.activation = activation
        self.layer_sizes = layer_sizes

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, self.layer_sizes[0])
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = apply_activation(x, self.activation)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, self.layer_sizes[1])
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = apply_activation(x, self.activation)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
