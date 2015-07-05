# ICML-2015
Code for models in the ICML 2015 paper: 
    "Variational Generative Stochastic Networks with Collaborative Shaping"

I'm in the process of working back from my more recent work -- which involved
substantial changes to the basic model implementations, optimization methods,
etc. -- to match this repo's state to what I was doing when I wrote the ICML
paper. The newer models in the Sequential-Generation repo are more powerful,
FYI. The main problem with a "unimodal" reconstruction distribution is that
forcing more overlap in the corruption distributions q(z|x1) and q(z|x2) --
via KL regularization -- becomes unreasonable when the required reconstruction
distribution p(x|z) contains multiple significant modes. Using a sequential
construction for the corruption process and reconstruction distribution allows
truncation of the corruption process at varying degrees of overlap in
q(z|x1)/q(z|x2). It also readily permits multi-modal reconstruction via a chain
of composed unimodal reconstruction distributions.

The file "MnistWalkoutTests.py" shows the basic incantations for initializing
and training the models described in the paper. The general VAE implementation
is in "OneStageModel.py" and the collaboratively guided Markov chain stuff is
in "VCGLoop.py". To perform more unrolling steps, the VCGLoop code would need
to be modified to use Theano's scan op, rather than manual loop unrolling.
