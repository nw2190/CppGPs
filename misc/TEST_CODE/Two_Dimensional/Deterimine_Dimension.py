import numpy as np
import csv
import time

# Evaluate SciKit Learn Gaussian Process Regressor and Plot Results
def main():

    # Get prediction data
    filename = "predictions.csv"
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row = next(csvreader)
        print(len(row))

    
# Run main() function when called directly
if __name__ == '__main__':
    main()
