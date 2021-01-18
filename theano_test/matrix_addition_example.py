import numpy 
import theano.tensor

from theano import function 

matrix1 = tensor.dmatrix('x') 
matrix2 = tensor.dmatrix('y') 
formula = matrix1 + matrix2 
toExecute = function([matrix1, matrix2], formula) 
  
result = toExecute([[30, 50], [2, 3]], [[60, 70], [3, 4]]) 
print(result)