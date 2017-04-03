#!/usr/bin/env python
"""Trains the model on some data, and saves it locally.

The default training text is the `lorem.txt` file, which consists of some
lorem ipsum text. The character frequencies in this text are as follows:

';' -> 2		'L' -> 8		'O' -> 12		'E' -> 34
'F' -> 39		'Q' -> 39		'U' -> 46		'I' -> 68
'j' -> 71		'A' -> 75		'V' -> 77		'C' -> 95
'S' -> 100		'x' -> 103		'D' -> 106		'M' -> 116
'P' -> 149		'N' -> 178		'\n' -> 200		'h' -> 271
'f' -> 431		'q' -> 593		'g' -> 620		'b' -> 625
'v' -> 771		',' -> 861		'p' -> 1085		'.' -> 1138
'd' -> 1416		'c' -> 2050		'o' -> 2184		'm' -> 2319
'r' -> 2833		'n' -> 2916		'l' -> 3017		'a' -> 3969
't' -> 4194		's' -> 4246		'u' -> 4514		'i' -> 5111
'e' -> 5643		' ' -> 8998
"""

from __future__ import print_function

import utils
from model import SeqGAN

import argparse

import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a SeqGAN model on some text.')
    parser.add_argument('-t', '--text', default='lorem.txt', type=str,
                        help='path to the text to use')
    parser.add_argument('-l', '--seq_len', default=100, type=int,
                        help='the length of each training sequence')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='size of each training batch')
    parser.add_argument('-n', '--num_steps', default=100, type=int,
                        help='number of steps per epoch')
    parser.add_argument('-e', '--num_epochs', default=1000, type=int,
                        help='number of training epochs')
    parser.add_argument('-c', '--only_cpu', default=False, action='store_true',
                        help='if set, only build weights on cpu')
    parser.add_argument('-p', '--learn_phase', default=None, type=int,
                        help='learning phase (None for synchronized)')
    parser.add_argument('-d', '--logdir', default='model/', type=str,
                        help='where to store the trained model')

    args = parser.parse_args()

    # Turns on logging.
    import logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    dictionary, rev_dict = utils.get_dictionary(args.text)
    num_classes = len(dictionary)

    iterator = utils.tokenize(args.text,
                              dictionary,
                              batch_size=args.batch_size,
                              seq_len=args.seq_len)

    sess = tf.Session()
    model = SeqGAN(sess,
                   num_classes,
                   logdir=args.logdir,
                   learn_phase=args.learn_phase,
                   only_cpu=args.only_cpu)
    model.build()
    model.load(ignore_missing=True)

    for epoch in xrange(1, args.num_epochs + 1):
        for step in xrange(1, args.num_steps + 1):
            logging.info('epoch %d, step %d', epoch, step)
            model.train_batch(iterator.next())

        # Generates a sample from the model.
        g = model.generate(1000)
        print(utils.detokenize(g, rev_dict))

        # Saves the model to the logdir.
        model.save()
