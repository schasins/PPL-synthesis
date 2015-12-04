import numpy as np

def ND(mat,beta=0.99,alpha=1,control=0):
    '''
    This is a python implementation/translation of network deconvolution by MIT-KELLIS LAB
    
    
     LICENSE: MIT-KELLIS LAB
    
    
     AUTHORS:
        Algorithm was programmed by Soheil Feizi.
    Python implementation: Gideon Rosenthal
    
    REFERENCES:
       For more details, see the following paper:
        Network Deconvolution as a General Method to Distinguish
        Direct Dependencies over Networks
    
    --------------------------------------------------------------------------
     ND.m: network deconvolution
    --------------------------------------------------------------------------
    
    DESCRIPTION:
    
     USAGE:
        mat_nd = ND(mat)
        mat_nd = ND(mat,beta)
        mat_nd = ND(mat,beta,alpha,control)
    
    
     INPUT ARGUMENTS:
     mat           Input matrix, if it is a square matrix, the program assumes
                   it is a relevance matrix where mat(i,j) represents the similarity content
                   between nodes i and j. Elements of matrix should be
                   non-negative.
     optional parameters:
     beta          Scaling parameter, the program maps the largest absolute eigenvalue
                   of the direct dependency matrix to beta. It should be
                   between 0 and 1.
     alpha         fraction of edges of the observed dependency matrix to be kept in
                   deconvolution process.
     control       if 0, displaying direct weights for observed
                   interactions, if 1, displaying direct weights for both observed and
                   non-observed interactions.
    
     OUTPUT ARGUMENTS:
    
     mat_nd        Output deconvolved matrix (direct dependency matrix). Its components
                   represent direct edge weights of observed interactions.
                   Choosing top direct interactions (a cut-off) depends on the application and
                   is not implemented in this code.
    
     To apply ND on regulatory networks, follow steps explained in Supplementary notes
     1.4.1 and 2.1 and 2.3 of the paper.
     In this implementation, input matrices are made symmetric.
    
    **************************************************************************
     loading scaling and thresholding parameters
    '''
    import scipy.stats.mstats as stat
    from numpy import linalg as LA


    if beta>=1 or beta<=0:
        print 'error: beta should be in (0,1)'
      
    if alpha>1 or alpha<=0:
            print 'error: alpha should be in (0,1)';
     
    
    '''
    ***********************************
     Processing the inut matrix
     diagonal values are filtered
    '''
    
    n = mat.shape[0]
    np.fill_diagonal(mat, 0)
    
    '''
    Thresholding the input matrix
    '''
    y =stat.mquantiles(mat[:],prob=[1-alpha])
    th = mat>=y
    mat_th=mat*th;

    '''
    making the matrix symetric if already not
    '''
    mat_th = (mat_th+mat_th.T)/2

    
    '''
    ***********************************
    eigen decomposition
    '''
    print 'Decomposition and deconvolution...'

    Dv,U = LA.eigh(mat_th) 
    D = np.diag((Dv))
    lam_n=np.abs(np.min(np.min(np.diag(D)),0))
    lam_p=np.abs(np.max(np.max(np.diag(D)),0))

    
    m1=lam_p*(1-beta)/beta
    m2=lam_n*(1+beta)/beta
    m=max(m1,m2)
    
    #network deconvolution
    for i in range(D.shape[0]):
        D[i,i] = (D[i,i])/(m+D[i,i])
    
    mat_new1 = np.dot(U,np.dot(D,LA.inv(U)))
    
                    
    '''
    
    ***********************************
     displying direct weights
    '''
    if control==0:
        ind_edges = (mat_th>0)*1.0;
        ind_nonedges = (mat_th==0)*1.0;
        m1 = np.max(np.max(mat*ind_nonedges));
        m2 = np.min(np.min(mat_new1));
        mat_new2 = (mat_new1+np.max(m1-m2,0))*ind_edges+(mat*ind_nonedges);
    else:
        m2 = np.min(np.min(mat_new1));
        mat_new2 = (mat_new1+np.max(-m2,0));
    
    
    '''
    ***********************************
     linearly mapping the deconvolved matrix to be between 0 and 1
    '''
    m1 = np.min(np.min(mat_new2));
    m2 = np.max(np.max(mat_new2));
    mat_nd = (mat_new2-m1)/(m2-m1);


    return mat_nd

# Example Usage
'''
x = np.array([[1,.8,.2,0],\
              [.8,1,.5,.1],\
              [.2,.5,1,-.1],\
              [0,.1,-.1,1]])
print x
print ND(x)
'''
