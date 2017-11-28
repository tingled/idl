# Autoencoders

IDL November 27, 2017

## Previous class recap
- Review Exam Q tips
- Kaggle competition

## Review of Problem Set
RNNs part 2

## Discussion

### Where are we
So far we have focused primarily on architecture.

Next we'll discuss:
- more complicated architectures & introspection
- optimization
- regularization
- best practices

### Autoencoders
1) trained to reconstruct their input (unsupervised)
2) common structure (often symmetrical)
    - encoder
    - decoder
3) representational bottleneck
    - structure (under-complete)
    - regularization (denoising, sparse, contractive)

#### Convolutional autoencoders
At first, learned simple encoders and all the interesting info was in activations.
After regularize activations to use "winner take all", which will force more information into the filters.

#### Cross trial encoder
Minimize reconstruction error bw x2 and f(g(x1)) where x1 and x2 are instances of same class

#### Similarity-Constraint Encoder (Triplet Networks)
Motivated by relative constraints

For trial triplets (x1, x2, z) where x1 and x2 are same class, z from another class

#### Siamese Network
Energy based loss function
- Using analogy of springs
- Pair from same class => low energy
- Pair from diff classes => high energy

#### Simple Autoencoder
Gray code as neighborhood preserving code

### Course Project
Add ideas in mattermos channel
