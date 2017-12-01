"""Utils for the TensorFlow model."""

from collections import OrderedDict
import warnings

import numpy as np
import tensorflow as tf

import utils

# Defines the maximum length of a training sequence.
SEQUENCE_MAXLEN = 1000


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
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def multinomial_3d(x):
    """Samples from a multinomial distribution from 3D Tensor.
    Args:
        x: Tensor with shape (batch_size, timesteps, classes)
    Returns:
        Tensor with shape (batch_size, timesteps), sampled from `classes`.
    """
    a, b = tf.shape(x)[0], tf.shape(x)[1]
    x = tf.reshape(x, (a * b, -1))
    m = tf.multinomial(x, 1)
    return tf.reshape(m, (a, b))


def multinomial_2d(x):
    """Samples from a multinomial distribution from 2D Tensor.
    Args:
        x: Tensor with shape (batch_size, classes)
    Returns:
        Tensor with shape (batch_size), sampled from `classes`.
    """
    a = tf.shape(x)[0]
    m = tf.multinomial(x, 1)
    return tf.reshape(m, (a,))


class SeqGAN(object):
    """The SeqGAN model.
    Args:
        sess: an active TF session to use for the model.
        num_classes: int, number of output classes (i.e. characters).
        learn_phase: int (default: None), the phase (i.e. one generator step
            per cycle, the rest discriminator steps). If None, the
            discriminator and generator are updated simultaneously.
        log_every: int (default: 50), how often to save tensor summaries.
        only_cpu: bool (default: False), if set, only build weights on CPU
            (useful for deploying a trained model to a production server).
        logdir: str, where to save each model epoch.
        num_latent: int, number of latent dimensions.
    """

    def __init__(self, sess, num_classes, learn_phase=None, log_every=50,
                 only_cpu=False, logdir='model/', num_latent=100):
        self._num_latent = num_latent
        self._logdir = logdir
        self._sess = sess
        self._num_classes = num_classes
        self._only_cpu = only_cpu
        self._weights = []
        self._learn_phase = learn_phase
        self.log_every = log_every

        # Builds the various placeholders.
        self._text_len_pl = tf.placeholder(
            dtype='int32', shape=(), name='text_len_pl')
        self._text_pl = tf.placeholder(
            dtype='int32', shape=(None, None), name='text_pl')
        self._latent_pl = tf.placeholder(
            dtype='float32', shape=(None, num_latent), name='latent_pl')
        self._time = tf.Variable(0, name='time')
        self._sample_pl = tf.placeholder(
            dtype='bool', shape=(), name='sample_pl')

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
    def sample_pl(self):
        return self._sample_pl

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

    @check_built
    @property
    def current_time(self):
        return self._sess.run(self.time)

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
                initializer = (lambda shape, dtype, partition_info:
                               tf.random_normal(shape, stddev=0.05))
            elif init == 'uniform':
                initializer = (lambda shape, dtype, partition_info:
                               tf.random_uniform(shape, stddev=0.05))
            elif init == 'glorot':
                initializer = (lambda shape, dtype, partition_info:
                               tf.random_normal(
                                   shape, stddev=np.sqrt(6. / sum(shape))))
            elif init == 'eye':
                assert all(i == shape[0] for i in shape)
                initializer = (lambda shape, dtype, partition_info:
                               tf.eye(shape[0]))
            elif init == 'zero':
                initializer = (lambda shape, dtype, partition_info:
                               tf.zeros(shape))
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
            weight = tf.get_variable(name=name,
                                     shape=shape,
                                     initializer=initializer,
                                     trainable=trainable)
        self._weights.append(weight)

        return weight

    def build_generator(self, use_multinomial=False, num_rnns=3, rnn_dims=128):
        """Builds the generator part of the model.
        Args:
            use_multinomial: bool (default: True), whether or not to sample
                from a multinomial distribution for each consecutive step of
                the RNN.
            num_rnns: int (default: 3), number of RNNs to stack.
            rnn_dims: int (default: 128), number of outputs of the RNN.
        Returns:
            a tensor representing the generated question.
        """

        with tf.variable_scope('generator'):

            # Creates the RNN output -> model output function.
            output_W = self.get_weight('output_W', (rnn_dims, self.num_classes))
            output_fn = lambda x: tf.matmul(x, output_W)

            # Creates the RNN cell.
            cells = [tf.contrib.rnn.GRUCell(rnn_dims) for _ in range(num_rnns)]
            cells = [tf.contrib.rnn.DropoutWrapper(cell, 0.7) for cell in cells]
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _add_proj(i, activation=tf.tanh):
                W = self.get_weight('rnn_proj_%d_W' % i,
                                    (self.num_latent, rnn_dims))
                b = self.get_weight('rnn_proj_%d_b' % i, (rnn_dims,))
                proj = activation(tf.matmul(self.latent_pl, W) + b)
                return proj

            # Creates the initial encoder state by mapping from latent dim.
            encoder_state = tuple(_add_proj(i) for i in range(num_rnns))

            # Gets the batch size from the latent pl.
            batch_size = tf.shape(self.latent_pl)[0]

            # Creates the teacher forcing op.
            teacher_inp = tf.concat([tf.zeros_like(self.text_pl[:, :1]),
                                     self.text_pl[:, :-1]], axis=1)
            teacher_inp = tf.one_hot(teacher_inp, self.num_classes)
            teacher_fn = tf.contrib.seq2seq.simple_decoder_fn_train(
                encoder_state)
            seq_len = tf.ones((batch_size,), 'int32') * self.text_len_pl
            teacher_preds, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=cell,
                inputs=teacher_inp,
                decoder_fn=teacher_fn,
                sequence_length=seq_len)
            teacher_preds = tf.einsum('ijk,kl->ijl', teacher_preds, output_W)
            teach_loss = tf.contrib.seq2seq.sequence_loss(
                logits=teacher_preds,
                targets=self.text_pl,
                weights=tf.ones((batch_size, self.text_len_pl)))
            teach_loss = tf.reduce_mean(teach_loss)

            # Reuses generator variables for the inference part.
            tf.get_variable_scope().reuse_variables()

            if use_multinomial:
                def infer_fn(time, state, input_var, output_var, ctx):
                    if output_var is None:
                        next_id = tf.zeros((batch_size,), 'int32')
                        state = encoder_state
                        output_var = tf.zeros((self.num_classes,))

                    else:
                        output_var = output_fn(output_var)
                        next_id = tf.cond(
                            self.sample_pl,
                            lambda: tf.argmax(output_var, axis=-1),
                            lambda: multinomial_2d(output_var))

                    next_input = tf.one_hot(next_id, self.num_classes)
                    done = tf.cond(tf.greater_equal(time, self.text_len_pl),
                                   lambda: tf.ones((batch_size,), 'bool'),
                                   lambda: tf.zeros((batch_size,), 'bool'))

                    return done, state, next_input, output_var, ctx

            else:
                embeddings = tf.eye(self.num_classes)
                infer_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=encoder_state,
                    embeddings=embeddings,
                    start_of_sequence_id=0,
                    end_of_sequence_id=-1,
                    maximum_length=self.text_len_pl - 1,
                    num_decoder_symbols=self.num_classes,
                    name='decoder_inference_fn')

            generated_sequence, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=cell,
                decoder_fn=infer_fn)

            class_scores = tf.nn.softmax(generated_sequence)
            generated_sequence = tf.argmax(generated_sequence, axis=-1)
            # generated_sequence = multinomial_3d(generated_sequence)

        tf.summary.scalar('loss/teacher', teach_loss)

        return class_scores, generated_sequence, teach_loss

    def build_discriminator(self, input_tensor, reuse=False, num_rnns=3,
                            rnn_dims=128):
        """Builds the discriminator part of the model.
        Args:
            input_tensor: Tensor with shape (batch_size, num_timesteps), where
                each value is an integer token index.
            reuse: bool (default: False), if set, reuse variable weights.
            num_rnns: int, number of RNNs to stack.
                of the model.
            rnn_dims: int, number of dimensions in each RNN.
        Returns:
            a tensor with shape (batch_size) that predicts whether the input
                tensor is real or fake.
        """

        with tf.variable_scope('discriminator'):

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
            preds = tf.sigmoid(preds)

        return preds

    def get_discriminator_op(self, r_preds, g_preds, d_weights):
        """Returns an op that updates the discriminator weights correctly.
        Args:
            r_preds: Tensor with shape (batch_size, num_timesteps, 1), the
                discriminator predictions for real data.
            g_preds: Tensor with shape (batch_size, num_timesteps, 1), the
                discriminator predictions for generated data.
            d_weights: a list of trainable tensors representing the weights
                associated with the discriminator model.
        Returns:
            dis_op, the op to run to train the discriminator.
        """

        with tf.variable_scope('loss/discriminator'):

            # Creates the optimizer.
            discriminator_opt = tf.train.AdamOptimizer(1e-4)

            # Computes log loss on real and generated sequences.
            eps = 1e-12
            r_loss = -tf.reduce_mean(tf.log(r_preds + eps))  # r_preds -> 1.
            f_loss = -tf.reduce_mean(tf.log(1 - g_preds + eps))  # g_preds -> 0.
            dis_loss = r_loss + f_loss

            # Adds summaries.
            tf.summary.scalar('real', r_loss)
            tf.summary.scalar('generated', f_loss)

            # Adds discriminator regularization loss.
            with tf.variable_scope('regularization'):
                dis_reg_loss = sum([tf.nn.l2_loss(w) for w in d_weights]) * 1e-4
            tf.summary.scalar('regularization', dis_reg_loss)

            # Minimizes the discriminator loss
            total_loss = dis_loss + dis_reg_loss
            dis_op = discriminator_opt.minimize(total_loss, var_list=d_weights)
            tf.summary.scalar('total', total_loss)

        return dis_op

    def get_generator_op(self, g_sequence, d_preds, g_preds, g_weights):
        """Returns an op that updates the generator weights correctly.
        Args:
            g_sequence: Tensor with shape (batch_size, num_timesteps) where
                each value is the token predicted by the generator.
            d_preds: Tensor with shape (batch_size, num_timesteps, 1)
                representing the output of the discriminator on the generated
                sequence.
            g_preds: Tensor with shape (batch_size, num_timesteps, num_classes)
                representing the softmax distribution over generator classes.
            g_weights: a list of trainable tensors representing the weights
                associated with the generator model.
        Returns:
            gen_op, the op to run to train the generator.
        """

        with tf.variable_scope('loss/generator'):

            # Creates the optimizer.
            generator_opt = tf.train.AdamOptimizer(1e-4)
            reward_opt = tf.train.GradientDescentOptimizer(1e-3)

            # Masks the predictions.
            g_sequence = tf.one_hot(g_sequence, self.num_classes)
            g_preds = tf.clip_by_value(g_preds * g_sequence, 1e-20, 1)

            # Keeps track of the "expected reward" at each timestep.
            expected_reward = tf.Variable(tf.zeros((SEQUENCE_MAXLEN,)))
            reward = d_preds - expected_reward[:tf.shape(d_preds)[1]]
            mean_reward = tf.reduce_mean(reward)

            # This variable is updated to know the "expected reward". This means
            # that only results that do surprisingly well are "kept" and used
            # to update the generator.
            exp_reward_loss = tf.reduce_mean(tf.abs(reward))
            exp_op = reward_opt.minimize(
                exp_reward_loss, var_list=[expected_reward])

            # The generator tries to maximize the outputs that lead to a high
            # reward value. Any timesteps before the reward happened should
            # recieve that reward (since it helped cause that reward).
            reward = tf.expand_dims(tf.cumsum(reward, axis=1, reverse=True), -1)
            gen_reward = tf.log(g_preds) * reward
            gen_reward = tf.reduce_mean(gen_reward)

            # Maximize the reward signal.
            gen_loss = -gen_reward

            # Adds generator regularization loss.
            with tf.variable_scope('regularization'):
                gen_reg_loss = sum([tf.nn.l2_loss(w) for w in g_weights]) * 1e-5
            tf.summary.scalar('regularization', gen_reg_loss)

            # Minimizes the generator loss.
            total_loss = gen_loss + gen_reg_loss
            gen_op = generator_opt.minimize(total_loss, var_list=g_weights)
            tf.summary.scalar('total', total_loss)

            gen_op = tf.group(gen_op, exp_op)

        tf.summary.scalar('loss/expected_reward', exp_reward_loss)
        tf.summary.scalar('reward/mean', mean_reward)
        tf.summary.scalar('reward/generator', gen_reward)

        return gen_op

    def build(self, reg_loss=1e-4):
        """Builds the model.
        Args:
            reg_loss: float, how much to weight regularization loss.
        """

        if hasattr(self, '_built') and self._built:
            raise RuntimeError('The model is already built.')

        g_classes, g_seq, teach_loss = self.build_generator()
        r_preds = self.build_discriminator(self.text_pl)
        g_preds = self.build_discriminator(g_seq, reuse=True)

        g_weights = get_scope_variables('generator')
        d_weights = get_scope_variables('discriminator')

        # Adds summaries of the real and fake predictions.
        tf.summary.histogram('predictions/fake', g_preds)
        tf.summary.histogram('predictions/real', r_preds)

        # Saves predictions for analysis later.
        self.g_preds, self.r_preds = g_preds, r_preds

        # Captures the generated sequence to use later.
        self.generated_sequence = g_seq

        # Computes the weight updates for the discriminator and generator.
        dis_op = self.get_discriminator_op(r_preds, g_preds, d_weights)
        gen_op = self.get_generator_op(g_seq, g_preds, g_classes, g_weights)

        # Adds the teacher forcing part, decaying at some rate.
        teach_lr = 10000. / (10000. + tf.cast(self.time, 'float32'))
        teach_lr *= 1e-3
        teach_opt = tf.train.AdamOptimizer(teach_lr)
        teach_op = teach_opt.minimize(teach_loss)
        gen_op = tf.group(gen_op, teach_op)
        tf.summary.scalar('teacher_lr', teach_lr)

        # Creates op to update time.
        step_op = self.time.assign(self.time + 1)

        # Allows the user to specify sequential vs. simultaneous updates.
        if self._learn_phase is None:
            gan_train_op = tf.group(gen_op, dis_op)
        else:
            gan_train_op = tf.cond(
                tf.equal(tf.mod(self.time, self._learn_phase), 0),
                lambda: gen_op,
                lambda: dis_op)

        # Updates time every step.
        self.train_op = tf.group(gan_train_op, step_op)

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
            self.sample_pl: False,
        }

        t = self._sess.run(self.time)
        if t % self.log_every:
            self._sess.run(self.train_op, feed_dict=feed_dict)
        else:
            _, summary = self._sess.run([self.train_op, self.summary_op],
                                        feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, t)

    @check_built
    def generate(self, sample_len):
        """Generates a sample from the model.
        Args:
            sample_len: int, length of the sample to generate.
        """

        latent = self._generate_latent_variable(1)
        sequence, = self._sess.run([self.generated_sequence],
                                   feed_dict={self.latent_pl: latent,
                                              self.text_len_pl: sample_len,
                                              self.sample_pl: True})
        return sequence[0]
