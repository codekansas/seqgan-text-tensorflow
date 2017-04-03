# seqgan-text-tensorflow

SeqGAN implementation for generating text using an RNN.

## Links

 - [Paper](https://arxiv.org/abs/1609.05473)
 - [Canonical Implementation](https://github.com/LantaoYu/SeqGAN)
 - [Another Implementation](https://github.com/ofirnachum/sequence_gan)

## Samples

By default the model is trained on some Lorem Ipsum text. After just 1000 minibatches, the model produces lorem-looking outputs:

```
en  eros  ells  at malesuada portere eget  Prlesent eellus egem ntum eget eel nec eel us. Nullam eu t mattis  on nisl eed euem eget vellus et malesuada  Num eu melles eg meliuenium el malesuada au eel uentum et  Ned erit est s erat  euismod eu malesuada  Nulla  velis  velitelis eget libero  Nullam sec us eros eel s egett s vusci  eel pem lacinia sed eem s vel massa vel  vel eu sapien, ante  el tempor  Sed eget tempur  eras eu lacinia ligula. Nullam posuere eills ar er malesuada eelit. Nullam sed  aancidunt lacus. Nulla vacien euis od luruus eul e et melus. Nulla eu sem aelierae et maleis eu eu musr sed ac erem. Ienc pretium necue elit  Nulla ac eget semper eget  Nullam ornare egim  Nullam eget lacinia nisl  elterdum  Nunc egettaugue velit llum eget. Nunec fermentum loctus  elis at mattis eel mellentesque  Nuscipit eros euementum egim nella  Nullam vel t lacus eel ee lacis eu m euet  mellentesque erat vetae eret lactus eells elet. Prasellus eget a faucibus eisus vel erit. Nullam velit a
```

## Usage

To train the default model to generate lines from `lorem.txt`, use:

```bash
>>> ./train
```

To sample from a trained model, use:

```bash
>>> ./sample <sample_len>
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
