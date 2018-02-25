# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:24:07 2015

@author: gil
"""
import numpy as np
import cvxopt
import collections
import pdb
#from pdb import set_trace as bp
def FindBestNuclearApprox(X, Lambda):
#    pdb.set_trace()
    (u,s,v)=np.linalg.svd(X,0)
    if sum(s)<Lambda:
        return X    
    n=s.size;
    H=np.eye(n);
    G=np.vstack([np.ones((1,n)), -np.eye(n)])
    h=np.hstack([np.array([[Lambda]]), np.zeros((1,n))]).T
    sol = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(-s.T), cvxopt.matrix(G), cvxopt.matrix(h));
    x=sol['x'];
    sol=u.dot(np.diagflat(x)).dot(v);
    return sol;

def FindBestSpectralApprox(X, Lambda):
    (u,s,v)=np.linalg.svd(X,0)
    if s[0]<Lambda:
        return X
    s[s>Lambda]=Lambda
    return u.dot(np.diagflat(s)).dot(v)
    
def MatrixApproximationNuclear(A, Init, Mask, Lambda, N, tol, display):
    Approximated = collections.namedtuple('ApproxMatrix',['NewMAT', 'Error'])
    NewMAT = Init
    S = sum(sum(Mask))
    ii = 0
    ErrFrob = float('inf')
    ErrPrev = 0
    while ((ii<=N) and (ErrFrob>tol)):
        NewMAT = (1-Mask)*NewMAT+Mask*A
        NewMAT = FindBestNuclearApprox(np.array(NewMAT), Lambda);
        ErrFrob = np.sqrt(sum(sum(((NewMAT-A)**2)*Mask)))/S
        if display:
            print("ii=%d, error=%g" %(ii, ErrFrob))
        ii+=1
        if ii>2:
            Diff = abs(ErrFrob-ErrPrev)
            if Diff<tol/1000:
                if display:
                    print( "Diff too small")
                return Approximated(NewMAT, ErrFrob)
        ErrPrev = ErrFrob
    return Approximated(NewMAT, ErrFrob)     
 
   
def MatrixApproximationSpectral(A, Init, Mask, Lambda, N, tol, display):
    Approximated = collections.namedtuple('ApproxMatrix',['NewMAT', 'Error'])
    NewMAT = Init
    S = sum(sum(Mask))
    ii = 0
    ErrFrob = float('inf')
    ErrPrev = 0
    while ((ii<=N) and (ErrFrob>tol)):
        NewMAT = (1-Mask)*NewMAT+Mask*A
        NewMAT = FindBestSpectralApprox(np.array(NewMAT), Lambda);
        ErrFrob = np.sqrt(sum(sum(((NewMAT-A)**2)*Mask)))/S
        if display:
            print("ii=%d, error=%g" %(ii, ErrFrob))
        ii+=1
        if ii>2:
            Diff = abs(ErrFrob-ErrPrev)
            if Diff<tol/1000:
                if display:
                    print("Diff too small")
                return Approximated(NewMAT, ErrFrob)
        ErrPrev = ErrFrob
    return Approximated(NewMAT, ErrFrob)     
    

def MatrixCompletion(A, Mask, N, Mode, lambda_tol, tol, display):
    cvxopt.solvers.options['show_progress'] = False
    Completed = collections.namedtuple('CompletedMatrix',['NewMAT', 'ier'])
    ier = 0
    min_lambda = 0
    if Mode=='nuclear':
        max_lambda = sum(np.linalg.svd(A,0,0))*1.1
    elif Mode=='spectral':
        max_lambda = max(np.linalg.svd(A,0,0))*1.1
    else:
        print("Wrong mode. Aborting.")
        return
    NewMAT = A;
    Lambda = float('inf')
    Lambda_prev = 0
    err = float('inf')
    Counter=0
    Converge=0
    while ((err>tol) or (abs(Lambda-Lambda_prev)>lambda_tol) ):
        Counter+=1
        print("Counter=%d" %Counter)
        Lambda_prev = Lambda
        Lambda = (min_lambda+max_lambda)/2
        if Mode=='nuclear':
            Out = MatrixApproximationNuclear(A*Mask, NewMAT, Mask, Lambda, N, tol, display)
        elif Mode=='spectral':
            Out = MatrixApproximationSpectral(A*Mask, NewMAT, Mask, Lambda, N, tol, display)
        NewMAT = Out.NewMAT
        err = Out.Error
        if err>tol:
            min_lambda=Lambda
        else:
            max_lambda=Lambda
        if abs(Lambda-Lambda_prev)<lambda_tol:
            Converge+=1
        if Converge>4:
            print("Convergence failed, try different parameters")
            ier=1       
            return Completed(Out.NewMAT, ier)
    return Completed(Out.NewMAT, ier)
   
    
