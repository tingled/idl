dynamic programming for backprop?

batch: calculate error on entire dataset before backprop

log comes from cross entropy loss function

maximum likelihood estimate

back propagation:
    tool for efficiently computing derivatives, but it's not the same thing as gradiant decent
    not an optization function

gradient decent:
    stoichastic -> just one example
    mini-batch -> some subset
    batch -> stable, but you won't get stuck in a local minima

activation functions:
    ReLUs very popular at time of writing, but they can end up with many dead units
    ELUs (exponential linear units)

linear layers in the middle can be used for compressing
    ie, 100 inputs fully connected to 100 outputs is 1e4 params, however you can go 100 -> 10 -> 100 which would be 2e3 params
