#!/usr/bin/env python
import numpy as np
from scipy import linalg
from scipy.sparse import diags

def jacobi(A, b,error):
	iteration=0
	[m, n] = np.shape(A)
	D = np.diag(A)
	R = A - np.diagflat(D)
	x = np.ones((m,1))
	err = np.ones((m,1))*100
	while np.max(err) > error:
		iteration=iteration+1
		xn = (b - np.dot(R,x))/ D
		err = abs((xn - x)/xn)*100
		x = xn
	
	print("jacobi:",x)	
	print(iteration)

def Gauss_Seidel(A, b, error_s):
	iteration=0
	[m, n] = np.shape(A)

	U = np.triu(A, 1)
	L = np.tril(A)

	x = np.ones((m,1))
	err = np.ones((m,1))*100

	while np.max(err) > error_s:
		iteration=iteration+1
		xn = np.dot(np.linalg.inv(L), (b - np.dot(U, x)))
		err = abs((xn - x)/xn)*100
		x = xn

	print("gauss_seidel:",x)
	print(iteration)


def main():
	n = 100
	k = -1*np.array([np.ones(n-1),-2*np.ones(n),1*np.ones(n-1)])
	offset = [-1,0,1]
	A = diags(k,offset).toarray()	
	print(A)
	
	s=(n,1)
	b=np.ones(s)

	
	x=b

	x2=linalg.solve(A, b)
	print("the aswer is:",x2)

	x_jacobi=jacobi(A,b,0)

	Gauss_Seidel(A, b, 0)
if __name__ == '__main__':
	main()
