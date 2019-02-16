import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import csv
import time

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

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

# Define function for removing axes from MatPlotLib plots
def remove_axes(ax):
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # remove axes
    ax._axis3don = False


# Evaluate SciKit Learn Gaussian Process Regressor and Plot Results
def main():


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
            if inputDim == 2:
                x1, x2, y = row
                obsX.append([x1,x2])
            else:
                x, y = row
                obsX.append(x)
            obsY.append(y)
    obsX = np.array(obsX).astype(np.float32)
    obsY = np.array(obsY).astype(np.float32)



    if inputDim == 1:
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
    X = np.reshape(obsX, [-1, inputDim])
    Y = np.reshape(obsY, [-1])
    Xtest = np.reshape(inVals, [-1, inputDim])


    # Model parameters
    n_restarts = 0
    normalize_y = False
    use_white_noise = True
    RBF_bounds = [0.01, 100.0]
    Noise_bounds = [0.00001, 10.0]
    jitter = 1e-7

    # Define kernel for SciKit Learn Gaussian process regression model
    if use_white_noise:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(RBF_bounds[0],RBF_bounds[1])) + \
            WhiteKernel(noise_level=1, noise_level_bounds=(Noise_bounds[0], Noise_bounds[1]))
    else:
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

    if inputDim == 1:
        model_samples = model.sample_y(Xtest, samples.shape[0])

    NLML = -model.log_marginal_likelihood()
    print("NLML:   {:.4f}\n".format(NLML))




    ###                          ###
    ###       PLOT RESULTS       ###
    ###                          ###

    if inputDim == 1:

        """ ONE-DIMENSIONAL PLOTS """
        
        # Plot Scikit Learn results
        plt.figure()
        plt.plot(inVals, mean, 'C0', linewidth=2.0)
        alpha = 0.075
        for k in [1,2,3]:
            plt.fill_between(inVals, mean-k*std, mean+k*std, where=1 >= 0, facecolor="C0", alpha=alpha, interpolate=True, label=None)
        plt.plot(inVals, trueVals, 'C1', linewidth=1.0, linestyle="dashed")
        alpha_scatter = 0.5
        plt.scatter(obsX, obsY, alpha=alpha_scatter)
        for i in range(0,model_samples.shape[1]):
            plt.plot(inVals, model_samples[:,i], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")
        plt.suptitle("Scikit Learn Implementation")


        ### C++ IMPLEMENTATION
        plt.figure()    
        plt.plot(inVals, predMean, 'C0', linewidth=2.0)
        for k in [1,2,3]:
            plt.fill_between(inVals, predMean-k*predStd, predMean+k*predStd, where=1>=0, facecolor="C0", alpha=alpha, interpolate=True)
        plt.plot(inVals, trueVals, 'C1', linewidth=1.0, linestyle="dashed")
        plt.scatter(obsX, obsY, alpha=alpha_scatter)
        for i in range(0,samples.shape[0]):
            plt.plot(inVals, samples[i,:], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")
        plt.suptitle("C++ Implementation")    
        plt.show()




    
    
    elif inputDim == 2:

        """ TWO-DIMENSIONAL PLOTS """
        
        # Flatten input values for compatibility with MatPlotLib's tri_surf
        plot_X_flat = []; plot_Y_flat = []
        R = inVals.shape[0]
        for n in range(0,R):
            plot_X_flat.append(inVals[n,0])
            plot_Y_flat.append(inVals[n,1])

        tri_fig = plt.figure()
        tri_ax1 = tri_fig.add_subplot(121, projection='3d')
        linewidth = 0.1
        cmap = "Blues"

        # Plot CppGPs results
        tri_ax1.plot_trisurf(plot_X_flat,plot_Y_flat, predMean, cmap=cmap, linewidth=linewidth, antialiased=True)
        pred_title = "CppGPs"
        tri_ax1.set_title(pred_title, fontsize=24)

        # Plot SciKit Learn results    
        tri_ax2 = tri_fig.add_subplot(122, projection='3d')
        tri_ax2.plot_trisurf(plot_X_flat,plot_Y_flat, mean, cmap=cmap, linewidth=linewidth, antialiased=True)
        soln_title = "SciKit Learn"
        tri_ax2.set_title(soln_title, fontsize=24)

        # Remove axes from plots
        remove_axes(tri_ax1) 
        remove_axes(tri_ax2) 

        # Bind axes for comparison
        def tri_on_move(event):
            if event.inaxes == tri_ax1:
                if tri_ax1.button_pressed in tri_ax1._rotate_btn:
                    tri_ax2.view_init(elev=tri_ax1.elev, azim=tri_ax1.azim)
                elif tri_ax1.button_pressed in tri_ax1._zoom_btn:
                    tri_ax2.set_xlim3d(tri_ax1.get_xlim3d())
                    tri_ax2.set_ylim3d(tri_ax1.get_ylim3d())
                    tri_ax2.set_zlim3d(tri_ax1.get_zlim3d())
            elif event.inaxes == tri_ax2:
                if tri_ax2.button_pressed in tri_ax2._rotate_btn:
                    tri_ax1.view_init(elev=tri_ax2.elev, azim=tri_ax2.azim)
                elif tri_ax2.button_pressed in tri_ax2._zoom_btn:
                    tri_ax1.set_xlim3d(tri_ax2.get_xlim3d())
                    tri_ax1.set_ylim3d(tri_ax2.get_ylim3d())
                    tri_ax1.set_zlim3d(tri_ax2.get_zlim3d())
            else:
                return
            tri_fig.canvas.draw_idle()
        tri_c1 = tri_fig.canvas.mpl_connect('motion_notify_event', tri_on_move)



        """ Zoom in to view predictive uncertainty """
        plot_radius = 0.5
        plot_x_min = -0.125
        plot_y_min = -0.125
        plot_x_max = 0.25

        # Define conditions for including values associated with an input point (x,y)
        def include_conditions(x,y,delta=0.0):
            rad = np.sqrt(np.power(x,2) + np.power(y,2))
            return (x-delta<=plot_x_max) and (x+delta>=plot_x_min) and (y+delta>=plot_y_min) and (rad-delta<=plot_radius)


        # Restrict plots to values corresponding to valid input points
        R = inVals.shape[0]    
        plot_X_zoom = []; plot_Y_zoom = []; predMean_zoom = []
        predMean_plus_std = []; predMean_minus_std = []
        predMean_plus_std2 = []; predMean_minus_std2 = []
        predMean_plus_std3 = []; predMean_minus_std3 = []
        trueVals_zoom = []
        for n in range(0,R):
            x = plot_X_flat[n]
            y = plot_Y_flat[n]
            if include_conditions(x,y,delta=0.025):
                plot_X_zoom.append(x)
                plot_Y_zoom.append(y)
                predMean_zoom.append(predMean[n])
                predMean_plus_std.append( predMean[n] + 1 * predStd[n] )
                predMean_minus_std.append( predMean[n] - 1 * predStd[n] )
                predMean_plus_std2.append( predMean[n] + 2 * predStd[n] )
                predMean_minus_std2.append( predMean[n] - 2 * predStd[n] )
                predMean_plus_std3.append( predMean[n] + 3 * predStd[n] )
                predMean_minus_std3.append( predMean[n] - 3 * predStd[n] )
                trueVals_zoom.append(trueVals[n])


        # Restrict observations to valid input points
        obsX_x_zoom = []; obsX_y_zoom = []; obsY_zoom = []
        for n in range(0, obsX.shape[0]):
            x = obsX[n,0]
            y = obsX[n,1]
            if include_conditions(x,y):
                obsX_x_zoom.append(x)
                obsX_y_zoom.append(y)
                obsY_zoom.append(obsY[n])


        # Initialize plot for assessing predictive uncertainty 
        tri_fig2 = plt.figure()
        tri2_ax1 = tri_fig2.add_subplot(111, projection='3d')


        ## Plot Predictive Mean
        linewidth = 0.1; alpha = 0.85
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_zoom, cmap=cmap, linewidth=linewidth, antialiased=True, alpha=alpha)


        ## One Standard Deviation
        linewidth = 0.075; alpha = 0.2
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_plus_std, cmap=cmap, linewidth=linewidth, antialiased=True,alpha=alpha)
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_minus_std, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)

        ## Two Standard Deviations
        linewidth = 0.05; alpha = 0.1
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_plus_std2, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_minus_std2, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)

        ## Three Standard Deviations
        linewidth = 0.01; alpha = 0.01
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_plus_std3, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_minus_std3, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)

        ## Scatter plot of training observations
        alpha = 0.4
        tri2_ax1.scatter(obsX_x_zoom, obsX_y_zoom, obsY_zoom, c='k', marker='o', s=15.0, alpha=alpha)

        # Remove axes from plot
        remove_axes(tri2_ax1)

        # Display plots
        plt.show()
    


# Run main() function when called directly
if __name__ == '__main__':
    main()
