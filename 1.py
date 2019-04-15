#!/usr/bin/env python
import numpy as np
from scipy import linalg
from scipy.sparse import diags
from numpy.linalg import inv

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

def SOR(A, b, error_s):
	iteration=0
	[m, n] = np.shape(A)
	w=1.5
	U = np.triu(A, 1)
	L = np.tril(A)-np.diag(np.diag(A))
	D=np.diag(np.diag(A))
	print("L",L)
	s1=np.linalg.inv(D+np.dot(w,L))
	s2=np.dot(w-1,D)+np.dot(w,U)
	s3=np.dot(w,s1)
	x = np.ones((m,1))
	err = np.ones((m,1))*100

	while np.max(err) > error_s:
		iteration=iteration+1
		xn=np.dot(np.dot(np.dot(-1,s1),s2),x)+np.dot(s3,b)
		err = abs((xn - x)/(xn+0.0001))*100
		x = xn

	print("SOR:",x)
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
	SOR(A, b, 0)
if __name__ == '__main__':
	main()
