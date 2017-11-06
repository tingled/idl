# CNNs part 2

## Short Recap

Relationship b/w filter shape, stride, padding and output shape
Pooling

## Assignment

Two important pieces from assignment:
-   visualizing filters
-   visualizing feature map

## Paper discussion

### You Only Look Once
Find objects in an image and classify them
One stage network, rather than a sliding window approach for classifying windows.

Object recognition within regions of the picture

22 conv layers and 2 fully connected layers

### Sentence Modelling Using CNNs

Evaluate on sentiment analysis datasets.

Input is word2vec embeddings for ach word in the sentence.

Wide convolution, then sum pooling, then K max pooling

### Super Resolution

3 hidden layers

First step is up sampling with bicubic interpolation

### Style Transfer

Content image and style image

Attempt to convert style image to content image
Pass each image through a pre-trained image classification network

Define some "style score" based on inner product of feature maps on style image

Define a loss function based on some combination of matching style score from style image and weights of hidden layer of content image

## Misc

With CNNs, semantics come from the shape of the data.

LeNet

## Questions

Is there any advantage to initializing filters with Gabor-style filters in addition to / instead of random weights?
No, but not a ton of background
