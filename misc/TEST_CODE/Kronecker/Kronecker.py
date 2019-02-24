import numpy as np
import scipy
import scipy.linalg
import time

# Set grid sizes
N1 = 30
N2 = 12
N3 = 10
N4 = 4
N = N1*N2*N3*N4

# Fix random seed
#np.random.seed(seed=0)

col1 = np.linspace(0.0,1.0,N1)
A = scipy.linalg.toeplitz(col1) 

col2 = np.linspace(0.0,1.0,N2)
B = scipy.linalg.toeplitz(col2) 

col3 = np.linspace(0.0,1.0,N3)
C = scipy.linalg.toeplitz(col3) 

col4 = np.linspace(0.0,1.0,N4)
D = scipy.linalg.toeplitz(col4) 

start_time = time.time()
T = np.kron(np.kron(np.kron(A,B),C),D)
end_time = time.time()
print('\nDirect Formation Time:  {:.5f} s'.format(end_time-start_time)) 



def kron_product(A,B,y):
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    y = y.astype(np.float64)
    input_dim = y.size
    output_dim = A.shape[0]*B.shape[0]
    vals = np.zeros([output_dim])

    A_dim_out = A.shape[0]
    A_dim_in = A.shape[1]
    B_dim_out = B.shape[0]
    B_dim_in = B.shape[1]

    y = np.reshape(y, [B_dim_in, -1], order='F')
    B_times_y = np.matmul(B,y)
    
    for i in range(0, A_dim_out):
        vals[i*B_dim_out : (i+1)*B_dim_out] = np.sum(A[i,:] * B_times_y[:,:], axis=1)
        #for j in range(0,A_dim_in):
        #    vals[i*B_dim_out : (i+1)*B_dim_out] += A[i,j] * B_times_y[:,j]
    return vals


###
###  REFERENCE: https://arxiv.org/pdf/1601.01507.pdf
###
def kron_vec_trick_2(A,B,y):
    A_dim_out = A.shape[0]
    B_dim_in = B.shape[1]
    y = np.reshape(y, [B_dim_in, A_dim_out], order='F')
    vals = np.matmul(B, np.matmul(y, A.transpose()))
    vals = np.reshape(vals, [-1], order='F')
    return vals


### Three Matrix Implementation
def kron_vec_trick_3(A,B,C,y):

    y = np.reshape(y, [-1, A.shape[0]], order='F')
    
    intermediate_list = []
    for i in range(0, y.shape[1]):
        intermediate_list.append(kron_vec_trick_2(B,C,y[:,i]))
    intermediate_list = np.array(intermediate_list).transpose()
    
    vals = np.matmul(intermediate_list, A.transpose())
    vals = np.reshape(vals, [-1], order='F')
    return vals

### Four Matrix Implementation
def kron_vec_trick_4(A,B,C,D,y):

    y = np.reshape(y, [-1, A.shape[0]], order='F')
    
    intermediate_list = []
    for i in range(0, y.shape[1]):
        intermediate_list.append(kron_vec_trick_3(B,C,D,y[:,i]))
    intermediate_list = np.array(intermediate_list).transpose()
    
    vals = np.matmul(intermediate_list, A.transpose())
    vals = np.reshape(vals, [-1], order='F')
    return vals


### General Implementation
def kron_vec_trick(matrix_list,y):

    # Remove first matrix from copy of original list
    matrix_list = matrix_list.copy()
    A = matrix_list.pop(0)

    if len(matrix_list) == 1:
        B = matrix_list.pop(0)
        y = np.reshape(y, [B.shape[1], A.shape[0]], order='F')
        vals = np.matmul(B, np.matmul(y, A.transpose()))
        vals = np.reshape(vals, [-1], order='F')
        return vals
    else:
        y = np.reshape(y, [-1, A.shape[0]], order='F')
        intermediate_list = []
        for i in range(0, y.shape[1]):
            intermediate_list.append(kron_vec_trick(matrix_list,y[:,i]))
        intermediate_list = np.array(intermediate_list).transpose()
        vals = np.matmul(intermediate_list, A.transpose())
        vals = np.reshape(vals, [-1], order='F')
        return vals


y = np.random.uniform(0.0,1.0,N)

start_time = time.time()
Ty = np.matmul(T,y)
end_time = time.time()
print('\nDirect Time:  {:.5f} s'.format(end_time-start_time)) 

start_time = time.time()
#Ty_approx = kron_vec_trick_3(A,B,C,y)
#Ty_approx = kron_vec_trick_4(A,B,C,D,y)
Ty_approx = kron_vec_trick([A,B,C,D],y)
end_time = time.time()
print('\nKernel Trick Time:  {:.5f} s \n'.format(end_time-start_time)) 

print(np.linalg.norm(Ty_approx-Ty))
