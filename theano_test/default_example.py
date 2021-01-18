import theano
from theano import tensor

number1 = tensor.dscalar()
number2 = tensor.dscalar()
formula = number1 + number2
toExecute = theano.function([number1,number2], formula)
result = toExecute(1.5, 2.5)
print (result)
