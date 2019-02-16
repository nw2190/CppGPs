import numpy as np
import csv
import time

from scipy.linalg import cholesky, cho_solve, solve_triangular

# Function for converting time to formatted string
def convert_time(t):
    minutes = np.floor((t/3600.0) * 60)
    seconds = np.ceil(((t/3600.0) * 60 - minutes) * 60)
    if (minutes >= 1):
        minutes = np.floor(t/60.0)
        seconds = np.ceil((t/60.0 - minutes) * 60)
        t_str = str(int(minutes)).rjust(2) + 'm  ' + \
                str(int(seconds)).rjust(2) + 's'
    else:
        seconds = (t/60.0 - minutes) * 60
        t_str = str(seconds) + 's'
    return t_str



# Evaluate SciKit Learn Gaussian Process Regressor and Plot Results
def main():


    # Get K matrix
    filename = "K.csv"
    K_vals = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            K_vals.append(np.array(row))
    obsCount = K_vals[0].size
    K_vals = np.array(K_vals).astype(np.float64)

    # Get term matrix
    filename = "term.csv"
    term_vals = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            term_vals.append(np.array(row))
    term_vals = np.array(term_vals).astype(np.float64)

    # Get alpha matrix
    filename = "alpha.csv"
    alpha_vals = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            alpha_vals.append(np.array(row))
    alpha_vals = np.array(alpha_vals).astype(np.float64)
    
    ### SCIKIT LEARN IMPLEMENTATION
    K = np.reshape(K_vals, [obsCount, obsCount])
    term = np.reshape(term_vals, [obsCount, obsCount])
    alpha = np.reshape(alpha_vals, [obsCount, 1])

    
    # Compute Cholesky factor
    start_time = time.time()
    L = cholesky(K, lower=True)
    end_time = time.time()
    time_elapsed = convert_time(end_time-start_time)
    print('\n Cholesky Time: \t'  + time_elapsed + '\n') 

    # May be worth comparing this as well...
    #alpha = cho_solve((L, True), y_train)

    start_time = time.time()
    scipy_term = -np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
    scipy_term += cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
    end_time = time.time()
    time_elapsed = convert_time(end_time-start_time)
    print(' Computation Time: \t'  + time_elapsed + '\n') 
    
    scipy_term = scipy_term[:,:,0]
    scipy_norm = np.linalg.norm(scipy_term)
    scipy_term -= term

    print('\n Relative Error: \t {:.5e}\n'.format( np.linalg.norm(scipy_term)/scipy_norm ) )

# Run main() function when called directly
if __name__ == '__main__':
    main()
