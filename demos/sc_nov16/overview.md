# SC Journcal Club
Nov 16, 2017

## Overview
### Structure

- Introduction to the course
- Overview of MLPs
- Tensorflow and Tensorboard
- Q / A
    + Is this a useful format?
    + Would people be interested in shorter, more frequent sessions?

### Goals
- Share my SAT learnings
- Introduce topics at a high level
- Pass along tips / tricks you may not have encountered before

### Non-goals
- Convince you that deep learning will suddenly give your life meaning
- Convince you that deep learning is snake oil
- Convince you that we should (or shouldn't) be using deep learning at SC
- Derive the Jacobian of your error function for some layer of some architecture

## The Course
Course website: https://mlcogup.github.io/idl2017/
Book: __Deep Learning__ Goodfellow, Bengio, Courville

Follow along if you'd like.

## Multi Layer Perceptrons (MLPs)

These are the simplest networks, and perhaps relatedly, they aren't used all that often.
However, they are very instructive for understanding foundational concepts that are used
by all networks.
Plus, you'll find fully connected, MLP-like structures embedded in many coplicated networks.

### Perceptrons
- Scalar input(s)
    - it's a number
- Weight value(s)
    - some other number
- Bias value
    - offset value

Looks a lot like mx + b
In fact, this is how it works. A single perceptron can be used as a very, very dumb linear classifier:
sign(wx + b)
You can even line a bunch of them up: H(X) = WX + B, where W is a matrix and X, B, and H(X) are vectors

### Multilayer Perceptrons
Like a perceptron, but more than one layer!
Also has an activation function to add some nonlinearity.
so instead of h(x) = wx + b, we now have h(x) = g(wx + b)

### Activation functions
Here are some:
- sigmoid
- tanh
- ReLU (Rectified Linear Unit)
    - Leaky ReLU (LReLU)
    - Exponential Linear Unit (ELU)
    - Shifted ReLU (SReLU)

### XOR problem
Straight from the book: http://www.deeplearningbook.org/contents/mlp.html

### Supervised Learning
As opposed to just getting the ideal network from a website, maybe we want to *learn* something

This looks like any other supervised learning problem. You have some training examples. You predict some output.
You can use the error of your output to update your model to perform better on this output.
(I'll let someone else give a better definition of supervised learning).

We just need a cost function and an optimization strategy.

I'm not going to talk much about the choice of cost function, but
most people use the cross entropy between the true
distribution and the predicted distribution. This is the
negative log-likelihood.

B/c of nonlinearities, most cost functions
become non-convex. So we use a gradient-descent based optimization.

Ok, now we've gotten that unpleasantness out of the way, we're good.
Just take the derivative of this thing and.. AHHH

### Back propagation

Exercise for the reader.

Revisit activation functions. Why do they matter? Where do they saturate?
