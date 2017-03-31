# seqgan-text-tensorflow

SeqGAN implementation for generating text using an RNN.

## Links

 - [Paper](https://arxiv.org/abs/1609.05473)
 - [Canonical Implementation](https://github.com/LantaoYu/SeqGAN)
 - [Another Implementation](https://github.com/ofirnachum/sequence_gan)

## Usage

To train the default model to generate lines from `lorem.txt`, use:

```bash
>>> ./train
```

To specify your own text, use:

```bash
>>> ./train -t /path/to/your/file.txt
```

To see all the training options, use:

```bash
>>> ./train --help
```

This gives

```
usage: train.py [-h] [-t TEXT] [-l SEQ_LEN] [-b BATCH_SIZE] [-n NUM_STEPS]
                [-e NUM_EPOCHS] [-c] [-p LEARN_PHASE]

Train a SeqGAN model on some text.

optional arguments:
  -h, --help            show this help message and exit
  -t TEXT, --text TEXT  path to the text to use
  -l SEQ_LEN, --seq_len SEQ_LEN
                        the length of each training sequence
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        size of each training batch
  -n NUM_STEPS, --num_steps NUM_STEPS
                        number of steps per epoch
  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        number of training epochs
  -c, --only_cpu        if set, only build weights on cpu
  -p LEARN_PHASE, --learn_phase LEARN_PHASE
                        learning phase (None for synchronized)
```
