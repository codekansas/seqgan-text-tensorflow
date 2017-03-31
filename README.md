# seqgan-text-tensorflow

SeqGAN implementation for generating text using an RNN.

## Some thoughts

This didn't work as well as I was expecting. After adding a vanilla teacher forcing rule, the model still didn't get any better at fooling the discriminator. This may have been due to a bug in the implementation, but if so, I can't figure out what it is.
