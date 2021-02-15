import theano
from theano import tensor
from theano import pp

variable = tensor.dmatrix('variable')
formula = tensor.sum(1/(1 + tensor.exp(-variable)))
gradient = tensor.grad(formula, variable)
toExecute = theano.function([variable], gradient)

print(toExecute([[0, 1], [-1, -2]])) #Expected results: [[0.25, 0.196], [0.20, 0.1]]

