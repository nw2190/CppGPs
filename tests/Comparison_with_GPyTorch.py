import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import time
import csv
import os

from Comparison_with_SciKit_Learn import getTrainingData, getSampleCount


# Evaluate GPyTorch Exact Gaussian Process Model on CppGPs training data
def main():

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    ##
    ##  REFERENCE PAPER: https://arxiv.org/pdf/1809.11165.pdf
    ##

    # Specify number of training iterations and learning rate for optimizer
    training_iter = 250
    learning_rate = 0.1

    # Option to display training information for each iteration
    VERBOSE = False


    # Get training data from CppGPs setup
    inputDim, obsX, obsY, inVals = getTrainingData()
    if inputDim == 1:
        sampleCount = getSampleCount()

    # Convert to float32
    obsX = obsX.astype(np.float32)
    obsY = obsY.astype(np.float32)
    inVals = inVals.astype(np.float32)


    # Define exact inference Gaussian process regression model
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Define the target function to be an oscillatory, non-periodic function
    def targetFunc(X):
        oscillation = 30.0
        xshifted = 0.5*(X[0] + 1.0)
        return np.sin(oscillation*(xshifted-0.1))*(0.5-(xshifted-0.1))*15.0

    # Define utility function for applying Python function to PyTorch tensor
    def applyFunc(func, Tensor):
        tList = [func(m) for m in torch.unbind(Tensor, dim=0) ]
        result = torch.stack(tList, dim=0)
        return result

    # Specify observation count and input dimension
    obsCount = obsX.shape[0]

    # Construct noise for observation data
    noiseLevel = 1.0
    normal_dist =  torch.distributions.Normal(0.0,noiseLevel)
    noise = normal_dist.rsample(sample_shape=torch.Size([obsCount]))

    # Generate random input observation data
    uniform_dist = torch.distributions.Uniform(-1.0, 1.0)
    #train_x = uniform_dist.rsample(sample_shape=torch.Size([obsCount, inputDim]))
    train_x = torch.from_numpy(obsX)

    # Generate noisy observation target data
    #train_y = applyFunc(targetFunc, train_x)
    #train_y = train_y + noise
    train_y = torch.from_numpy(obsY)

    # Initialize the likelihood function and GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)    

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the Adam optimizer to tune hyperparameters
    optimizer = torch.optim.Adam([ {'params': model.parameters()}, ], lr=learning_rate)

    # Define (negative) loss function for GP model to be the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Define the training loop
    start_time = time.time()
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        # Calculate loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()    
        if VERBOSE:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item()*obsCount,
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()



    end_time = time.time()

    # Display computation time 
    print('\nComputation Time:  {:.5f} s \n'.format(end_time-start_time))


    ##
    #   [ Posterior Predictions and Samples ]
    ##

    with torch.no_grad():
        # They appear to scale the NLML calculation by 1/obsCount
        # (also this must be run before issuing the ".eval()" statements)
        NLML = -mll.forward(model(train_x), train_y)
        NLML = obsCount * NLML 


    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    print("Optimized Hyperparameters:")
    kernel_outputscale = model.covar_module.outputscale.item()
    length_scale = model.covar_module.base_kernel.lengthscale.item()
    predictive_noise = model.likelihood.noise.item()
    print(" {:.4f}**2 * {:.4f}   (Noise = {:.4f})\n".format(np.sqrt(kernel_outputscale), length_scale, predictive_noise))

    print("NLML = {:.4f}\n".format(NLML))


    # Specify the input mesh for testing
    #predCount = inVals.shape[0]
    #testMesh = torch.linspace(-1.0, 1.0, steps=predCount)
    #testMesh = testMesh.unsqueeze(0)
    #testMesh = torch.transpose(testMesh, 0, 1)
    testMesh = torch.from_numpy(inVals)

    # Define true solution values on test mesh
    #trueSoln = applyFunc(targetFunc, testMesh)

    # Get posterior distributions for the mean and output values
    mean_predictive_distribution = model(testMesh)
    posterior_distribution = likelihood(model(testMesh))

    # Get posterior means and variances
    pred_mean = posterior_distribution.mean
    pred_var = posterior_distribution.variance
    pred_covar = posterior_distribution.covariance_matrix

    if inputDim == 1:
        # Get sample paths from posterior
        samples = posterior_distribution.rsample(sample_shape=torch.Size([sampleCount]))



    GPyTorch_results_dir = "./GPyTorch_Results/"

    if not os.path.exists(GPyTorch_results_dir):
        os.makedirs(GPyTorch_results_dir)

    # Save results to file
    with torch.no_grad():

        # Save prediction data
        filename = os.path.join(GPyTorch_results_dir, "predMean.npy")
        predMean = pred_mean.detach().numpy()
        np.save(filename, predMean)

        filename = os.path.join(GPyTorch_results_dir, "predStd.npy")
        predStd = np.sqrt(pred_var.detach().numpy())
        np.save(filename, predStd)


        if inputDim == 1:
            # Save posterior samples
            filename = os.path.join(GPyTorch_results_dir, "samples.npy")
            samples = samples.detach().numpy()
            np.save(filename, samples)

        # Save NLML
        filename = os.path.join(GPyTorch_results_dir, "NLML.npy")
        np.save(filename, NLML)
    

# Run main() function when called directly
if __name__ == '__main__':
    main()
    
