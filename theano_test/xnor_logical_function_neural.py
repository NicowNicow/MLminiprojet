# Two-layers neural network implementation of the XNOR logical function

# Imports
import theano
import theano.tensor
from theano import function

from random import random
import numpy


# Variables Definition
inputMatrix = theano.tensor.matrix('inputMatrix')

weight1 = theano.shared(numpy.array([random(), random()]))
weight2 = theano.shared(numpy.array([random(), random()]))
weight3 = theano.shared(numpy.array([random(), random()]))

bias1 = theano.shared(1.)
bias2 = theano.shared(1.)

learningRate = 0.01
iterationNumber = 100000


# Neurons Definition
neuron1 = 1/(1 + theano.tensor.exp( -theano.tensor.dot(inputMatrix, weight1) - bias1))
neuron2 = 1/(1 + theano.tensor.exp( -theano.tensor.dot(inputMatrix, weight2) - bias1))
firstLayerResulMatrix = theano.tensor.stack([neuron1, neuron2], axis=1)

neuron3 = 1/(1 + theano.tensor.exp( -theano.tensor.dot(firstLayerResulMatrix, weight3) - bias2))


# Gradient Definition
realOutput = theano.tensor.vector('realOutput')
cost = -(realOutput*theano.tensor.log(neuron3) + (1 - realOutput)*theano.tensor.log(1-neuron3)).sum()
gradWeight1, gradWeight2, gradWeight3, gradBias1, gradBias2 = theano.tensor.grad(cost, [weight1, weight2, weight3, bias1, bias2])


# Weight and Bias update
TrainingFunction = function(
    inputs = [inputMatrix, realOutput],
    outputs = [neuron3, cost],
    updates = [
        [weight1, weight1-learningRate*gradWeight1],
        [weight2, weight2-learningRate*gradWeight2],
        [weight3, weight3-learningRate*gradWeight3],
        [bias1, bias1-learningRate*gradBias1],
        [bias2, bias2-learningRate*gradBias2]
    ]
)


# Inputs and Outputs Definition
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

outputs = [1, 0, 0, 1]


# Model Training
costArray = []
for index in range(iterationNumber):
    predictionArray, iterationCost = TrainingFunction(inputs, outputs)
    costArray.append(iterationCost)

# Output Printing
print('Here are the outputs of the Neural Network:')
for index in range (len(inputs)):
    print('The output for the input [%d, %d] is %.2f' % (inputs[index][0], inputs[index][1], predictionArray[index])) 