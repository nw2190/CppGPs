import numpy as np
import matplotlib.pyplot as plt
import csv

# Plot results from CppGP code
def main():

    # Get prediction data
    filename = "predictions.csv"
    inVals = []; trueVals = []; predMean = []; predStd = []
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
    obsX = []; obsY = []
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


    ### Plot CppGPs results
    plt.figure()    
    plt.plot(inVals, predMean, 'C0', linewidth=2.0)

    # Add standard deviation regions
    alpha = 0.1
    for k in [1,2,3]:
        plt.fill_between(inVals, predMean - k*predStd, predMean + k*predStd, where=1 >= 0, facecolor="C0", alpha=alpha, interpolate=True, label=None)

    # Plot true solution values
    plt.plot(inVals, trueVals, 'C1', linewidth=2.0, linestyle="dashed")

    # Plot observation points
    plt.scatter(obsX, obsY)

    # Plot posterior sample paths
    for i in range(0,samples.shape[0]):
        plt.plot(inVals, samples[i,:], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")

    # Add title to plot
    plt.suptitle("CppGPs Posterior Prediction")    
    plt.show()


# Run main() function when called directly
if __name__ == '__main__':
    main()
