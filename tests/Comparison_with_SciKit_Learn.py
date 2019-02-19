import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import os
import csv
import time


# Evaluate SciKit Learn Gaussian Process Regressor on CppGPs training data
def main():

    # Get training data from CppGPs setup
    inputDim, obsX, obsY, inVals = getTrainingData()
    if inputDim == 1:
        sampleCount = getSampleCount()
        
    ### SCIKIT LEARN IMPLEMENTATION
    X = np.reshape(obsX, [-1, inputDim])
    Y = np.reshape(obsY, [-1])
    Xtest = np.reshape(inVals, [-1, inputDim])

    # Model parameters
    n_restarts = 0
    normalize_y = False
    use_white_noise = True
    RBF_bounds = [0.01, 100.0]
    Noise_bounds = [0.00001, 20.0]
    jitter = 1e-7

    # Define kernel for SciKit Learn Gaussian process regression model
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(RBF_bounds[0],RBF_bounds[1])) + \
        WhiteKernel(noise_level=1, noise_level_bounds=(Noise_bounds[0], Noise_bounds[1]))


    # Fit model to data
    start_time = time.time()
    model = GaussianProcessRegressor(kernel=kernel, alpha=jitter, optimizer='fmin_l_bfgs_b',
                                     normalize_y=normalize_y, n_restarts_optimizer=n_restarts).fit(X, Y)
    end_time = time.time()

    # Display computation time 
    print('\nComputation Time:  {:.5f} s \n'.format(end_time-start_time)) 

    print("Optimized Kernel Parameters:")
    print(model.kernel_)
    print(" ")
    mean, std = model.predict(Xtest, return_std=True)

    if inputDim == 1:
        model_samples = model.sample_y(Xtest, sampleCount)

    NLML = -model.log_marginal_likelihood()
    print("NLML:  {:.4f}\n".format(NLML))

    
    # Save results to file
    SciKit_Learn_results_dir = "./SciKit_Learn_Results/"

    if not os.path.exists(SciKit_Learn_results_dir):
        os.makedirs(SciKit_Learn_results_dir)

    # Save prediction data
    filename = os.path.join(SciKit_Learn_results_dir, "predMean.npy")
    np.save(filename, mean)

    filename = os.path.join(SciKit_Learn_results_dir, "predStd.npy")
    np.save(filename, std)

    if inputDim == 1:
        # Save posterior samples
        filename = os.path.join(SciKit_Learn_results_dir, "samples.npy")
        np.save(filename, model_samples)

    # Save NLML
    filename = os.path.join(SciKit_Learn_results_dir, "NLML.npy")
    np.save(filename, NLML)
    


# Get training data from CppGPs setup
def getTrainingData():

    # First determine the dimension of the input values
    filename = "predictions.csv"
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row = next(csvreader)
        nonInputLength = 3
        inputDim = len(row) - nonInputLength

    # Get prediction data
    filename = "predictions.csv"
    inVals = []; trueVals = []; predMean = []; predStd = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:

            if inputDim == 2:
                i1, i2, t, m, v = row
                inVals.append([i1,i2])
            else:
                i, t, m, v = row
                inVals.append(i)
                
            trueVals.append(t)
            predMean.append(m)
            predStd.append(v)
    inVals = np.array(inVals).astype(np.float64)
    trueVals = np.array(trueVals).astype(np.float64)
    predMean = np.array(predMean).astype(np.float64)
    predStd = np.array(predStd).astype(np.float64)

    ## Get observation data
    filename = "observations.csv"
    obsX = []; obsY = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            if inputDim == 2:
                x1, x2, y = row
                obsX.append([x1,x2])
            else:
                x, y = row
                obsX.append(x)
            obsY.append(y)
    obsX = np.array(obsX).astype(np.float64)
    obsY = np.array(obsY).astype(np.float64)
    return inputDim, obsX, obsY, inVals

def getSampleCount():
    # Get posterior samples
    filename = "samples.csv"
    samples = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            vals = np.array(row)
            samples.append(vals)
    samples = np.array(samples).astype(np.float64)
    return samples.shape[0]


# Run main() function when called directly
if __name__ == '__main__':
    main()
