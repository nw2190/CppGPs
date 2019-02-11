mport numpy as np
import scipy
import time

# Import matrix reconstruction function
from matvec import construct_mat

# Define function for computing low-rank approximation
def lowrank_approx(T, jitter, threshold=None, verbose=False, use_operator=False):


    # Compute full SVD for target matrix
    if use_operator:
        T_mat = construct_mat(T)
        colspace, singular_vals, rowspace = np.linalg.svd(T_mat)
    else:
        colspace, singular_vals, rowspace = np.linalg.svd(T)

    ## May have to use SciPy for this with matvec apparently...
    #R = 10
    #ncv = T.shape[0] - 1
    #colspace, singular_vals, rowspace = scipy.sparse.linalg.svds(T, k=R, ncv=ncv, tol=0)

    # Remove singular values below specified threshold
    if threshold is None:
        threshold = 2/3*np.square(jitter)
    singular_vals[singular_vals<threshold] = np.zeros((singular_vals[singular_vals<threshold]).shape)

    # Index of first zero entry coincides with rank
    R = np.argmin(singular_vals)

    # Reduce dimensions of low-rank approximation factors
    singular_vals = singular_vals[:R]
    colspace = colspace[:,:R]
    rowspace = rowspace[:R,:]

    # Display low-rank approximation information
    if verbose:

        # Show dimensions of approximation factors
        print("\n  - Rank:  {} ".format(R))
        print("\n  - T = U C V   ({} x {}) ({} x {}) ({} x {})"\
              .format(colspace.shape[0], colspace.shape[1],
                      singular_vals.shape[0], singular_vals.shape[0],
                      rowspace.shape[0], rowspace.shape[1]))


    return colspace, singular_vals, rowspace, R


# Define function for approximating log determinant from low-rank approximation
def lowrank_approx_ld(singular_vals, N, jitter):
    R = singular_vals.size
    return np.sum(np.log(singular_vals+jitter)) + (N-R)*np.log(jitter)


# Define function for computing inverse using Sherman-Morrison-Woodbury formula
def SMW_inverse(U, C, V, jitter):
    C_inv = np.reciprocal(C)
    VU = np.matmul(V,U)
    C_inv_plus_VU = np.diag(C_inv) + 1/jitter*VU
    offdiag = 1/np.square(jitter) * np.matmul( U, np.matmul( np.linalg.inv(C_inv_plus_VU), V))
    return 1/jitter*np.eye(U.shape[0]) - offdiag






# Define function for converting time to "h m s" string
def convert_time(t):
    hours = np.floor(t/3600.0)
    minutes = np.floor((t/3600.0 - hours) * 60)
    seconds = np.ceil(((t/3600.0 - hours) * 60 - minutes) * 60)
    if hours > 0:
        minutes = np.floor((t/3600.0 - hours) * 60)
        seconds = np.ceil(((t/3600.0 - hours) * 60 - minutes) * 60)
        t_str = str(int(hours)) + 'h  ' + \
                str(int(minutes)).rjust(2) + 'm  ' + \
                str(int(seconds)).rjust(2) + 's'
    elif (hours == 0) and (minutes >= 1):
        minutes = np.floor(t/60.0)
        seconds = np.ceil((t/60.0 - minutes) * 60)
        t_str = str(int(minutes)).rjust(2) + 'm  ' + \
                str(int(seconds)).rjust(2) + 's'
    else:
        seconds = (t/60.0 - minutes) * 60
        t_str = "{:.5}".format(seconds) + 's'
    return t_str

# Define class for timing subroutines
class Timer:
    def __init__(self):
        self._start = 0.0
        self._end = 0.0
    def start(self):
        self._start = time.time()
    def end(self):
        self._end = time.time()
    def show_time(self):
        print(convert_time(self._end-self._start))
    def get_time_string(self):
        return convert_time(self._end-self._start)
    def get_time(self):
        return self._end-self._start

