"""Utils for the TensorFlow model."""

import warnings

import numpy as np
import tensorflow as tf

import utils


def check_built(f):
    """Simple wrapper to make sure the model is built."""
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_built') or not self._built:
            raise RuntimeError('You must build the model before calling '
                               '"%s".' % f.__name__)
        return f(self, *args, **kwargs)
    return wrapper


def get_scope_variables(scope):
    """Returns all the variables in scope.
    Args:
        scope: str, the scope to use.
    Returns:
        list of variables.
    """

    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


class SeqGAN(object):
    """The SeqGAN model.
    Args:
        sess: an active TF session to use for the model.
        num_classes: int, number of output classes (i.e. characters).
        only_cpu: bool (default: False), if set, only build weights on CPU
            (useful for deploying a trained model to a production server).
        logdir: str, where to save each model epoch.
        num_latent: int, number of latent dimensions.
    """

    def __init__(self, sess, num_classes,
                 only_cpu=False, logdir='model/', num_latent=100):
        self._num_latent = num_latent
        self._logdir = logdir
        self._sess = sess
        self._num_classes = num_classes
        self._only_cpu = only_cpu
        self._weights = []

        # Builds the various placeholders.
        self._text_len_pl = tf.placeholder(
            dtype='int32', shape=(), name='text_len_pl')
        self._text_pl = tf.placeholder(
            dtype='int32', shape=(None, None), name='text_pl')
        self._latent_pl = tf.placeholder(
            dtype='float32', shape=(None, num_latent), name='latent_pl')
        self._time = tf.Variable(0, name='time')

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def text_pl(self):
        return self._text_pl

    @property
    def text_len_pl(self):
        return self._text_len_pl

    @property
    def latent_pl(self):
        return self._latent_pl

    @property
    def time(self):
        return self._time

    @property
    def num_latent(self):
        return self._num_latent

    @property
    def logdir(self):
        return self._logdir

    def _generate_latent_variable(self, batch_size):
        return np.random.normal(size=(batch_size, self.num_latent))

    def get_weight(self, name, shape,
                   init='glorot',
                   device='gpu',
                   weight_val=None,
                   trainable=True):
        """Creates a new weight.
        Args:
            name: str, the name of the variable.
            shape: tuple of ints, the shape of the variable.
            init: str, the type of initialize to use.
            device: str, 'cpu' or 'gpu'.
            weight_val: Numpy array to use as the initial weights.
            trainable: bool, whether or not this weight is trainable.
        Returns:
            a trainable TF variable with shape `shape`.
        """

        if weight_val is None:
            init = init.lower()
            if init == 'normal':
                weight_val = tf.random_normal(shape, stddev=0.05)
            elif init == 'uniform':
                weight_val = tf.random_uniform(shape, maxval=0.05)
            elif init == 'glorot':
                stddev = np.sqrt(6. / sum(shape))
                weight_val = tf.random_normal(shape, stddev=stddev)
            elif init == 'eye':
                assert all(i == shape[0] for i in shape)
                weight_val = tf.eye(shape[0])
            elif init == 'zero':
                weight_val = tf.zeros(shape)
            else:
                raise ValueError('Invalid init: "%s"' % init)
        else:
            weight_val = weight_val.astype('float32')

        device = device.lower()
        if device == 'gpu':
            on_gpu = True
        elif device == 'cpu':
            on_gpu = False
        else:
            raise ValueError('Invalid device: "%s"' % device)

        if self._only_cpu:
            on_gpu = False

        with tf.device('/gpu:0' if on_gpu else '/cpu:0'):
            weight = tf.Variable(weight_val, name=name, trainable=trainable)
        self._weights.append(weight)

        return weight

    def build_generator(self, num_rnns=3, rnn_dims=128, scope='generator'):
        """Builds the generator part of the model.
        Args:
            num_rnns: int, number of RNNs to stack.
            rnn_dims: int, number of outputs of the RNN.
            scope: str (default: "generator"), the scope of this part
                of the model.
        Returns:
            a tensor representing the generated question.
        """

        with tf.variable_scope(scope):

            # Creates the RNN output -> model output function.
            output_W = self.get_weight('output_W', (rnn_dims, self.num_classes))
            output_fn = lambda x: tf.matmul(x, output_W)

            # Creates the RNN cell.
            cells = [tf.contrib.rnn.GRUCell(rnn_dims) for _ in range(num_rnns)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _add_proj(i, activation=tf.tanh):
                W = self.get_weight('rnn_proj_%d_W' % i,
                                    (self.num_latent, rnn_dims))
                b = self.get_weight('rnn_proj_%d_b' % i, (rnn_dims,))
                proj = activation(tf.matmul(self.latent_pl, W) + b)
                return proj

            # Creates the initial encoder state by mapping from latent dim.
            encoder_state = tuple(_add_proj(i) for i in range(num_rnns))

            # Creates the embeddings (just one-hot encodings).
            embeddings = tf.eye(self.num_classes)

            # Builds the inferrence functions.
            infer_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=encoder_state,
                embeddings=embeddings,
                start_of_sequence_id=utils.START_IDX,
                end_of_sequence_id=-1,
                maximum_length=self.text_len_pl - 1,
                num_decoder_symbols=self.num_classes,
                name='decoder_inference_fn')

            generated_sequence, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=cell,
                decoder_fn=infer_fn)

            generatd_sequence = tf.nn.softmax(generated_sequence)

            generator_weights = get_scope_variables(scope)

        return generated_sequence, generator_weights

    def build_discriminator(self, input_tensor, reuse=False, num_rnns=3,
                            rnn_dims=128, scope='discriminator'):
        """Builds the discriminator part of the model.
        Args:
            input_tensor: Tensor with shape (batch_size, num_timesteps), where
                each value is an integer token index.
            reuse: bool (default: False), if set, reuse variable weights.
            num_rnns: int, number of RNNs to stack.
            rnn_dims: int, number of outputs of the RNN.
            scope: str (default: "discriminator"), the scope of this part
                of the model.
        Returns:
            a tensor with shape (batch_size) that predicts whether the input
                tensor is real or fake.
        """

        with tf.variable_scope(scope):

            if reuse:
                tf.get_variable_scope().reuse_variables()

            # Encodes the tensors as one-hot.
            input_ohe = tf.one_hot(input_tensor, self.num_classes)

            # Creates the RNN cell.
            cells = [tf.contrib.rnn.GRUCell(rnn_dims) for _ in range(num_rnns)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell, True)

            # Calls the cell, doing the necessary transpose op.
            input_ohe = tf.transpose(input_ohe, (1, 0, 2))
            rnn_output, _ = cell(input_ohe, dtype='float32')
            rnn_output = tf.transpose(rnn_output, (1, 0, 2))

            # Reduces to binary prediction.
            pred_W = self.get_weight('pred_W', (rnn_dims, 1))
            preds = tf.einsum('ijk,kl->ijl', rnn_output, pred_W)
            preds = tf.squeeze(tf.sigmoid(preds), axis=-1)

            discriminator_weights = get_scope_variables(scope)

        return preds, discriminator_weights

    def build(self):
        """Builds the model."""

        g_sequence, g_weights = self.build_generator()
        g_argmax = tf.argmax(g_sequence, axis=-1)
        r_preds, d_weights = self.build_discriminator(self.text_pl)
        g_preds, _ = self.build_discriminator(g_argmax, reuse=True)

        # Captures the generated sequence to use later.
        self.generated_sequence = g_argmax

        # Creates the discriminator loss function.
        eps = 1e-12
        dis_loss = -tf.log(r_preds + eps) - tf.log(1 - g_preds)

        # Creates separate generator and discriminator optimizers.
        generator_opt = tf.train.AdamOptimizer(learning_rate=1e-4)
        discriminator_opt = tf.train.AdamOptimizer(learning_rate=3e-4)

        # Creates the discriminator minimization op.
        dis_op = discriminator_opt.minimize(dis_loss)

        # Creates the generator policy.
        policy = tf.expand_dims(1 - 2 * g_preds, axis=-1)
        ohe = tf.one_hot(g_argmax, self.num_classes)
        generator_grad_loss = policy * ohe

        # Creates the generator minimization op.
        gvs = generator_opt.compute_gradients(
            g_sequence, g_weights, grad_loss=generator_grad_loss)
        gen_op = generator_opt.apply_gradients(gvs)

        # Creates op to update time.
        step_op = self.time.assign(self.time + 1)

        self.train_op = tf.group(gen_op, dis_op, step_op)

        # Creates the log directory and saving objects.
        if self.logdir is None:
            self.logdir = tempfile.mkdtemp()
            sys.stdout.write('logdir: "%s"\n' % self.logdir)
        self.summary_writer = tf.summary.FileWriter(
            self.logdir, self._sess.graph)
        self.summary_op = tf.summary.merge_all()

        self._saver = tf.train.Saver()
        self._sess.run(tf.global_variables_initializer())
        self._built = True


    @check_built
    def load(self, ignore_missing=False):
        """Loads the model from the logdir.
        Args:
            ignore_missing: bool, if set, ignore when no save_dir exists,
                otherwise raises an error.
        """

        ckpt = tf.train.get_checkpoint_state(self.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
        elif ignore_missing:
            return
        elif not ckpt:
            raise ValueError('No checkpoint found: "%s"' % self.logdir)
        else:
            raise ValueError('Checkpoint found, but no model checkpoint path '
                             'in "%s"' % self.logdir)

    @check_built
    def save(self):
        """Saves the model to the logdir."""

        self._saver.save(self._sess, self.logdir + 'model.ckpt')

    @check_built
    def train_batch(self, batch):
        """Trains on a single batch of data.
        Args:
            batch: numpy array with shape (batch_size, num_timesteps), where
                values are encoded tokens.
        """

        batch_size, seq_len = batch.shape
        latent = self._generate_latent_variable(batch_size)

        feed_dict = {
            self.text_pl: batch,
            self.text_len_pl: seq_len,
            self.latent_pl: latent,
        }

        _ = self._sess.run([self.train_op], feed_dict=feed_dict)

    @check_built
    def generate(self, sample_len):
        """Generates a sample from the model.
        Args:
            sample_len: int, length of the sample to generate.
        """

        latent = self._generate_latent_variable(1)
        sequence, = self._sess.run([self.generated_sequence],
                                   feed_dict={self.latent_pl: latent,
                                              self.text_len_pl: sample_len})
        return sequence[0]
