import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import csv
import time

# Function for converting time to formatted string
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
        t_str = str(seconds) + 's'
    return t_str


def main():

    # Get prediction data
    filename = "predictions.csv"
    inVals = []
    trueVals = []
    predMean = []
    predStd = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            i, t, m, v = row
            inVals.append(i)
            trueVals.append(t)
            predMean.append(m)
            predStd.append(v)
    inVals = np.array(inVals).astype(np.float32)
    trueVals = np.array(trueVals).astype(np.float32)
    predMean = np.array(predMean).astype(np.float32)
    predStd = np.array(predStd).astype(np.float32)

    ## Get observation data
    filename = "observations.csv"
    obsX = []
    obsY = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            x, y = row
            obsX.append(x)
            obsY.append(y)
    obsX = np.array(obsX).astype(np.float32)
    obsY = np.array(obsY).astype(np.float32)

    # Get posterior samples
    filename = "samples.csv"
    samples = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            vals = np.array(row)
            samples.append(vals)
    samples = np.array(samples).astype(np.float32)



    ### SCIKIT LEARN IMPLEMENTATION
    X = np.reshape(obsX, [-1, 1])
    Y = np.reshape(obsY, [-1])
    Xtest = np.reshape(inVals, [-1, 1])

    # Model parameters
    n_restarts = 0
    normalize_y = False
    use_white_noise = True
    #use_white_noise = False  ### Completely overfit...
    #use_white_noise = False
    #RBF_bounds = [0.01, 500.0]
    #Noise_bounds = [0.00001, 10.0]
    RBF_bounds = [0.01, 100.0]
    Noise_bounds = [0.00001, 5.0]
    jitter = 1e-7
    if use_white_noise:
        #kernel = RBF(length_scale=1.0, length_scale_bounds=(0.001,500.0)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        kernel = RBF(length_scale=1.0, length_scale_bounds=(RBF_bounds[0],RBF_bounds[1])) + WhiteKernel(noise_level=1, noise_level_bounds=(Noise_bounds[0], Noise_bounds[1]))
    else:
        #kernel = RBF(length_scale=1.0, length_scale_bounds=(0.001,500.0))
        kernel = RBF(length_scale=1.0, length_scale_bounds=(RBF_bounds[0],RBF_bounds[1]))

    # Fit model to data
    start_time = time.time()
    model = GaussianProcessRegressor(kernel=kernel, alpha=jitter, optimizer='fmin_l_bfgs_b',
                                     normalize_y=normalize_y, n_restarts_optimizer=n_restarts).fit(X, Y)
    end_time = time.time()

    # Display computation time 
    time_elapsed = convert_time(end_time-start_time)
    print('\nComputation Time:  '  + time_elapsed + '\n') 

    print("Optimized Kernel Parameters:")
    print(model.kernel_)
    print(" ")
    mean, std = model.predict(Xtest, return_std=True)
    model_samples = model.sample_y(Xtest, samples.shape[0])

    #print("\nSample Path Count:")
    #print(samples.shape[0])

    #print("\nCpp Standard Deviations:")
    #print(predStd[:10])
    
    #print("\nSciKit Standard Deviations:")
    #print(std[:10])

    NLML = -model.log_marginal_likelihood()
    print("NLML:   {:.4f}\n".format(NLML))

    
    # Plot Scikit Learn results
    plt.figure()
    plt.plot(inVals, mean, 'C0', linewidth=2.0)

    alpha = 0.1
    for k in [1,2,3]:
        plt.fill_between(inVals, mean - k*std, mean + k*std, where=1 >= 0, facecolor="C0", alpha=alpha, interpolate=True, label=None)

    plt.plot(inVals, trueVals, 'C1', linewidth=2.0, linestyle="dashed")

    plt.scatter(obsX, obsY)

    for i in range(0,model_samples.shape[1]):
        plt.plot(inVals, model_samples[:,i], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")
    plt.suptitle("Scikit Learn Implementation")


    ### C++ IMPLEMENTATION
    plt.figure()    
    plt.plot(inVals, predMean, 'C0', linewidth=2.0)

    alpha = 0.1
    for k in [1,2,3]:
        plt.fill_between(inVals, predMean - k*predStd, predMean + k*predStd, where=1 >= 0, facecolor="C0", alpha=alpha, interpolate=True, label=None)

    plt.plot(inVals, trueVals, 'C1', linewidth=2.0, linestyle="dashed")

    plt.scatter(obsX, obsY)

    for i in range(0,samples.shape[0]):
        plt.plot(inVals, samples[i,:], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")

    plt.suptitle("C++ Implementation")    
    plt.show()


# Run main() function when called directly
if __name__ == '__main__':
    main()
