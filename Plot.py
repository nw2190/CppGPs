import numpy as np
import matplotlib.pyplot as plt
import csv



def main():

    filename = "predictions.csv"

    inVals = []
    trueVals = []
    predMean = []
    predVar = []
    
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            i, t, m, v = row
            inVals.append(i)
            trueVals.append(t)
            predMean.append(m)
            predVar.append(v)
            
    inVals = np.array(inVals).astype(np.float32)
    trueVals = np.array(trueVals).astype(np.float32)
    predMean = np.array(predMean).astype(np.float32)
    predVar = np.array(predVar).astype(np.float32)



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

    filename = "samples.csv"
    samples = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            vals = np.array(row)
            samples.append(vals)
            
    samples = np.array(samples).astype(np.float32)

    
    
    plt.plot(inVals, predMean, 'C0', linewidth=2.0)

    alpha = 0.1
    for k in [1,2,3]:
        plt.fill_between(inVals, predMean - k*predVar, predMean + k*predVar, where=1 >= 0, facecolor="C0", alpha=alpha, interpolate=True, label=None)

    plt.plot(inVals, trueVals, 'C1', linewidth=2.0, linestyle="dashed")

    plt.scatter(obsX, obsY)

    for i in range(0,samples.shape[0]):
        plt.plot(inVals, samples[i,:], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")
    
    plt.show()


# Run main() function when called directly
if __name__ == '__main__':
    main()
