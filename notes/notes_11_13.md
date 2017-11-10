# RNNs

## Reading notes
Specialized for processing sequences of values
Parameter sharing makes it possible to extend and apply model to examples of differing lengths, and allows us to generalize. zB, rather than having to learn a new set of weights for each time-step in a sequence, we simply reuse the same weights for each time-step.
You can build a computational graph by "unrolling" a recursive circuit-style diagram. The benefits of doing so are:
1) the model has the same input size regardless of length of signal
2) you can share parameters

The unrolled diagram shows how to build a computational graph, and thus how to propagate loss through network using *bptt* or back propagation through time.

When the network has hidden-to-hidden this can be computationally expensive, and not easily parallelizable (unlike CNNs) b/c each state depends on previous state.

When the network has no hidden-to-hidden connections, (say it only has output-to-hidden connections,) the loss at an individual time step can be computed without knowing the entire history of the sequence. You only need to know the predicted output at time `t-1`. During training, you can feed the ground truth label for time-step `t-1` as an input for calculating the output at time `t`. This is called *teacher forcing*.

Teacher forcing can be used when a network has hidden-to-hidden connections, as long as there are also output-to-hidden connections. In this scenario you also would use bptt.

One issue with teacher forcing is that the predicted outputs when running in *open-loop* mode, may differ from the ground truth outputs, which could then bias the model. There are different approaches to mitigate this problem that involve feeding some combination of generated or ground truth data into the model while training.

## In Course notes

### Assignment overview with Jens

#### LeNet
Some people were having trouble using ReLUs in their LeNet. It's possible that small
networks may fail when using ReLU, as they could have difficulty
accounting for sparse outputs of ReLU. Weight initialization can also impact the of
ReLU networks.

#### Tensorboard
We discussed some issues with TensorBoard. Visualizing sprites can help you
understand what outliers may look like.
Using the pane on the right, you can search for and visualize specific classes.

projections into lower dimensional spaces can measured by their "trustworthiness" and "continuity"

trustworthiness - neighbors in low dimensional space are correct in the high dimensional (original) space
continuity - neighbors in high dimensional (original) space are close in the low dimensional space

Distance metrics:
- images: generally use Euclidean
- text: generally use Cosine
- Variational Autoencoders can learn a similarity measure at the same time as your network

### Group work

We then broke out into small groups to discuss the properties of different RNN architectures
described in the text

See photos
