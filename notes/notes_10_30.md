# CNNs
### Oct 30, 2017

## Recap last class
Back-prop, not optimzation alg, used for updating params
Batch size: refers to number of training examples before propogating derivs
Activation functions: sigmoid, hyperbolic tangent, ReLU, Leaky Relu, Maxout

### Programming Assignment

Tips:
- When you run into NaNs in your weights, this is often an overflow problem, as a result of exploding gradient
- When you see tiny gradients, this could be vanishing gradients, when the gradient
    is unable to reach the first layers of the network. Sigmoid activation functions may saturate
- If ReLU weights are initialized < 0, you can starve them, where they will output zero and cannot
    propogate any gradients

Before ADAM and other modern optimizers, people used to choose the highest learning that wouldn't lead to
exploding gradients.


