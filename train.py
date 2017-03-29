#!/usr/bin/env python
"""Trains the model on some data, and saves it locally."""

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
    parser.add_argument('-l', '--seq_len', default=50, type=int,
                        help='the length of each training sequence')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='size of each training batch')
    parser.add_argument('-n', '--num_steps', default=1000, type=int,
                        help='number of steps per epoch')
    parser.add_argument('-e', '--num_epochs', default=100, type=int,
                        help='number of training epochs')

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
    model = SeqGAN(sess, num_classes, only_cpu=True)
    model.build()

    for epoch in xrange(args.num_epochs):
        for step in xrange(args.num_steps):
            logging.info('epoch %d, step %d', epoch, step)
            model.train_batch(iterator.next())

        # Generates a sample from the model.
        g = model.generate(100)
        logging.info('Epoch %d: "%s"', epoch, utils.detokenize(g, rev_dict))

        # Saves the model to the logdir.
        model.save()
