import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import time
import csv
import os

from Comparison_with_SciKit_Learn import getTrainingData, getSampleCount

import matplotlib.pyplot as plt

USE_Standard = True
USE_SKI = True

# Option to display training information for each iteration
VERBOSE = False
PLOT_LOSS = False

# Evaluate GPyTorch Exact Gaussian Process Model on CppGPs training data
def standard():

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    ##
    ##  REFERENCE PAPER: https://arxiv.org/pdf/1809.11165.pdf
    ##


    # Get training data from CppGPs setup
    inputDim, obsX, obsY, inVals = getTrainingData()
    if inputDim == 1:
        sampleCount = getSampleCount()
    
    # Specify number of training iterations and learning rate for optimizer
    if inputDim == 1:
        training_iter = 250
    else:
        training_iter = 75
    learning_rate = 0.1

    # Specify precision level for stopping criteria
    eps = 2.220446049250313e-16
    factr = 1e11
    ftol = factr*eps

    if USE_SKI:
        print("\n[ Standard Implementation ]")

    # Convert to float32
    obsX = obsX.astype(np.float32)
    obsY = obsY.astype(np.float32)
    inVals = inVals.astype(np.float32)

    # Define exact inference Gaussian process regression model
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
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
    model = GPRegressionModel(train_x, train_y, likelihood)    

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the Adam optimizer to tune hyperparameters
    optimizer = torch.optim.Adam([ {'params': model.parameters()}, ], lr=learning_rate)

    # Define (negative) loss function for GP model to be the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Define the training loop
    start_time = time.time()
    f_prev = 1e9
    loss_steps = []
    loss_list = []
    for i in range(training_iter):

        #learning_decay_step = 5
        #if np.mod(i,learning_decay_step) == 0:
        #    learning_decay = 0.9
        #    learning_rate = learning_rate*learning_decay
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = learning_rate

        #for param_group in optimizer.param_groups:
        #    print("{:.4e}".format(param_group['lr']))
        
        
        optimizer.zero_grad()
        output = model(train_x)
        # Calculate loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if VERBOSE:
            print('Iter %d/%d - Loss: %.3f   outputscale: %.3f lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item()*obsCount,
                np.sqrt(model.covar_module.outputscale.item()),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))

        ### Try enforcing stopping criteria; NLML values seem to be too noisy...
        """
        f_current = loss.item()*obsCount
        stopping_crit = np.abs((f_current - f_prev)/(f_current))
        if VERBOSE:
            print("[ Stopping Criteria: {:.4e} < {:.4e} ]   Moving Average = {:.4f} ".format(stopping_crit,ftol,f_prev))
        if stopping_crit < ftol:
            break;
        # Define exponential moving average to account for noise
        alpha = 0.25
        f_prev = alpha*f_current + (1-alpha)*f_prev
        """
        if PLOT_LOSS:
            loss_steps.append(i)
            loss_list.append(loss.item()*obsCount)
        optimizer.step()

    end_time = time.time()

    if VERBOSE:
        print("\n[*] Function Evaluations: {}".format(i+1))
    
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


    # Plot loss values from training procedure
    if PLOT_LOSS:
        plt.plot(np.array(loss_steps),np.array(loss_list))
        plt.show()


    GPyTorch_results_dir = "./gpytorch_results/"

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


# Evaluate GPyTorch Exact Gaussian Process Model on CppGPs training data
def SKI():

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    ##
    ##  REFERENCE PAPER: https://arxiv.org/pdf/1809.11165.pdf
    ##

    # Get training data from CppGPs setup
    inputDim, obsX, obsY, inVals = getTrainingData()
    if inputDim == 1:
        sampleCount = getSampleCount()

    # Specify number of training iterations and learning rate for optimizer
    if inputDim == 1:
        training_iter = 250
    else:
        training_iter = 75
    learning_rate = 0.1

    # Specify precision level for stopping criteria
    eps = 2.220446049250313e-16
    factr = 1e10
    ftol = factr*eps


    if USE_Standard:
        print("\n[ SKI Implementation ]")

    # Convert to float32
    obsX = obsX.astype(np.float32)
    obsY = obsY.astype(np.float32)
    inVals = inVals.astype(np.float32)


    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x,1.0)

            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                grid_size=grid_size, num_dims=inputDim,
            )

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
    model = GPRegressionModel(train_x, train_y, likelihood)    

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the Adam optimizer to tune hyperparameters
    optimizer = torch.optim.Adam([ {'params': model.parameters()}, ], lr=learning_rate)

    # Define (negative) loss function for GP model to be the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Define the training loop
    start_time = time.time()
    f_prev = 1e9
    for i in range(training_iter):

        # Try using learning rate decay
        """
        learning_decay_step = 5
        if np.mod(i,learning_decay_step) == 0:
            learning_decay = 0.9
            learning_rate = learning_rate*learning_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        #for param_group in optimizer.param_groups:
        #    print("{:.4e}".format(param_group['lr']))
        """
        
        optimizer.zero_grad()
        output = model(train_x)
        # Calculate loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()    
        if VERBOSE:
            print('Iter %d/%d - Loss: %.3f   outputscale: %.3f lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item()*obsCount,
                np.sqrt(model.covar_module.base_kernel.outputscale.item()),
                model.covar_module.base_kernel.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        ### Try enforcing stopping criteria; NLML values seem to be too noisy...
        """
        f_current = loss.item()*obsCount
        stopping_crit = np.abs((f_current - f_prev)/(f_current))
        if VERBOSE:
            print("[ Stopping Criteria: {:.4e} < {:.4e} ] ".format(stopping_crit,ftol))
        if stopping_crit < ftol:
            break;
        f_prev = f_current
        """
        optimizer.step()



    end_time = time.time()

    if VERBOSE:
        print("\n[*] Function Evaluations: {}".format(i+1))
    
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


    # Unwrap the GridInterpolationKernel to obtain underlying ScaleKernel
    scale_kernel = model.covar_module.base_kernel

    print("Optimized Hyperparameters:")
    kernel_outputscale = scale_kernel.outputscale.item()
    length_scale = scale_kernel.base_kernel.lengthscale.item()
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

    with torch.no_grad(), gpytorch.settings.fast_pred_var():    
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



    GPyTorch_results_dir = "./gpytorch_results/"

    if not os.path.exists(GPyTorch_results_dir):
        os.makedirs(GPyTorch_results_dir)

    # Save results to file
    with torch.no_grad():

        # Save prediction data
        filename = os.path.join(GPyTorch_results_dir, "SKI_predMean.npy")
        predMean = pred_mean.detach().numpy()
        np.save(filename, predMean)

        filename = os.path.join(GPyTorch_results_dir, "SKI_predStd.npy")
        predStd = np.sqrt(pred_var.detach().numpy())
        np.save(filename, predStd)


        if inputDim == 1:
            # Save posterior samples
            filename = os.path.join(GPyTorch_results_dir, "SKI_samples.npy")
            samples = samples.detach().numpy()
            np.save(filename, samples)

        # Save NLML
        filename = os.path.join(GPyTorch_results_dir, "SKI_NLML.npy")
        np.save(filename, NLML)
        

# Run main() function when called directly
if __name__ == '__main__':
    if USE_Standard:
        standard()
    if USE_SKI:
        SKI()
    
