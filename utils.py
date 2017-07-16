"""Some utils to help with text parsing / feeding to the model."""

from __future__ import division

import os
import cPickle as pkl
import logging

import numpy as np

DICTIONARY_NAME = 'dictionary.pkl'
START_TOKEN, END_TOKEN = '$', '^'
START_IDX, END_IDX = 0, 1  # Start and end indices.


def get_dictionary(fpath, dfile=DICTIONARY_NAME, rebuild=False):
    """Creates the dictionary from a file, or loads cached dictionary.
    Args:
        fpath: str, the path to the file to build a dictionary from.
        dfile: str (default: DICTIONARY_NAME), the name of the pickled
            dictionary to use if none is provided.
        rebuild: bool (default: Fales), if set, removes an old dictionary
            and rebuilds a new one.
    Returns:
        a dict of char->idx mappings used to convert tokens to indices, and
        a reverse dictionary of idx->char mappings.
    """

    if not os.path.exists(dfile) or rebuild:
        logging.info('Dictionary not found; creating it in "%s"', dfile)
        with open(fpath, 'r') as f:
            dictionary = set(f.read())
        if START_TOKEN in dictionary:
            dictionary.remove(START_TOKEN)
        if END_TOKEN in dictionary:
            dictionary.remove(END_TOKEN)
        dictionary = [START_TOKEN, END_TOKEN] + list(dictionary)
        dictionary = dict((c, i) for i, c in enumerate(dictionary))
        with open(dfile, 'wb') as f:
            pkl.dump(dictionary, f)
    else:
        with open(dfile, 'rb') as f:
            dictionary = pkl.load(f)

    rev_dict = dict((i, c) for c, i in dictionary.items())

    return dictionary, rev_dict


def tokenize(fpath, dictionary, batch_size=32, seq_len=50):
    """Yields lines of indices, creating a dictionary file if none exists.
    Args:
        fpath: str, the path to the file to convert.
        dictionary: dict (default: None), the token->index mappings.
        batch_size: int (default: 32), the size of each training batch.
        seq_len: int (default: 50), the length of each training sequence.
    Yields:
        tokenized batches with shape (batch_size, seq_len) consisting of
            tokens as integers.
    """

    if not os.path.exists(fpath):
        raise IOError('File not found to tokenize: "%s"' % fpath)

    with open(fpath, 'r') as f:
        all_text = f.read()

    idxs = np.arange(0, len(all_text) - seq_len)

    while True:
        batch_idxs = np.random.choice(idxs, size=(batch_size,), replace=False)
        texts = [all_text[i:i + seq_len] for i in batch_idxs]
        tokens = [[dictionary[i] for i in text] for text in texts]
        yield np.asarray(tokens)


def detokenize(indices, rev_dict, argmax=False):
    """Converts text from indices back to a string.
    Args:
        indices: list of ints or Numpy array, the indices to detokenize. If
            argmax is True, this array has shape (text_length, num_classes)
            and consists of distributions over class probabilities,
            otherwise it has shape (text_length) and consists of integers
            representing the classes.
        rev_dict: dict (default: None), the idx->token mapping.
        argmax: bool (default: False), if set, takes the argmax over the class
            dimension to get the actual tokens.
    Returns:
        the detokenized string.
    """

    if argmax:
        indices = np.argmax(indices, axis=-1)

    chars = [rev_dict.get(i, None) for i in indices]
    text = ''.join(c for c in chars if c)  # Removes all None elements.

    return text
