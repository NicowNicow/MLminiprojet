import numpy 
import theano.tensor

from theano import function 

matrix1 = tensor.dmatrix('matrix1') 
matrix2 = tensor.dmatrix('matrix2') 
formula = matrix1 + matrix2 
toExecute = function([matrix1, matrix2], formula) 
  
print(toExecute([[30, 50], [2, 3]], [[60, 70], [3, 4]]) )