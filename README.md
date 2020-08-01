# Lewis_Signalling_RVED
We present a Lewis signaling-based collaborative ''guessing game'' that is capable of learning the joint distribution of labeled images. We adopt a VAE-like formalism to design this Lewis Signaling Recurrent Variational Encoder-Decoder Network (LS-RVED). The LS-RVED is a recurrent probabilistic encoder-decoder network, where at each time step, the encoder observes a portion from the labeled image and encodes that onto a latent variable (a signal). Given this signal, the decoder network tries to generate/''guess'' the labeled image. This process over time is used to train the encoder and decoder to generate effective labeled images.<br>

### Dataset 
The present version is for MNIST dataset. Shortly, we will provide for other datasets like: PCAM, Chest-Xray-14, FIRE, HAM10000 from the medical domain, and CIFAR 10, LSUN, ImageNet

### How to run the code
run the train.ipython notebook
