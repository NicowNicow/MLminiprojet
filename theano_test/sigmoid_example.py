import theano
from theano import tensor

variable = tensor.dmatrix('x')
formula = 1/(1 + tensor.exp(-variable))

sigmoid = theano.function([variable], formula)

print(sigmoid([[0, 1], [-1, -2]])) #Expected result: [[0.5, 0.731], [0.27, 0.12]]