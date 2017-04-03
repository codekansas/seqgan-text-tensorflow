#!/usr/bin/env python
"""Samples from a trained model.

usage: sample.py [-h] [-t DICTIONARY] [-d LOGDIR] [-c] N

Sample from a trained SeqGAN model.

positional arguments:
  N                     length of sample to generate

optional arguments:
  -h, --help            show this help message and exit
  -t DICTIONARY, --dictionary DICTIONARY
                        path to dictionary file
  -d LOGDIR, --logdir LOGDIR
                        directory of the trained model
  -c, --only_cpu        if set, only build weights on cpu
"""

from __future__ import print_function

import utils
from model import SeqGAN

import argparse
import os

import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample from a trained SeqGAN model.')
    parser.add_argument('sample_len', metavar='N', type=int,
                        help='length of sample to generate')
    parser.add_argument('-t', '--dictionary', default='dictionary.pkl',
                        type=str, help='path to dictionary file')
    parser.add_argument('-d', '--logdir', default='model/', type=str,
                        help='directory of the trained model')
    parser.add_argument('-c', '--only_cpu', default=True, action='store_true',
                        help='if set, only build weights on cpu')
    args = parser.parse_args()

    if not os.path.exists(args.dictionary):
        raise ValueError('No dictionary file found: "%s". To build it, '
                         'run train.py' % args.dictionary)

    _, rev_dict = utils.get_dictionary(None, dfile=args.dictionary)
    num_classes = len(rev_dict)

    sess = tf.Session()
    model = SeqGAN(sess,
                   num_classes,
                   logdir=args.logdir,
                   only_cpu=args.only_cpu)
    model.build()
    model.load(ignore_missing=True)

    g = model.generate(args.sample_len)
    print('Generated text:', utils.detokenize(g, rev_dict))
