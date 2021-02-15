import theano
import numpy

#Trying to approach a target with minimal training material, because lazyness

variable = theano.tensor.fvector('variable')
target = theano.tensor.fscalar('target')
weight = theano.shared(numpy.asarray([0.2, 0.7]), 'weight')

weightedSum = (variable * weight).sum()
cost = theano.tensor.sqr(target - weightedSum) #Cost functions are used to train models
gradient = theano.tensor.grad(cost, [weight])
updatedWeight = weight - (0.1 * gradient[0])
updates = [(weight, updatedWeight)]

targetApproach = theano.function([variable, target], weightedSum, updates=updates)
for index in range(25):
    print(targetApproach([1.0, 1.0], 20.0))