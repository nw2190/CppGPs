import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import os
import csv
import time

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


# Plot results of CppGPs, SciKit Learn, and GPyTorch
def main():

    # Specify whether or not to compare SciKit Learn results
    USE_SciKit_Learn = True

    # Specify whether or not to compare GPyTorch / SKI results
    USE_GPyTorch = False
    USE_GPyTorch_SKI = False

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



    if inputDim == 1:
        # Get posterior samples
        filename = "samples.csv"
        samples = []
        with open(filename, "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                vals = np.array(row)
                samples.append(vals)
        samples = np.array(samples).astype(np.float64)
    
    #filename = "NLML.csv"
    #NLML_tmp = []
    #with open(filename, "r") as csvfile:
    #    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #    for row in csvreader:
    #        vals = np.array(row, dtype=np.float32)
    #        NLML_tmp.append(vals)
    #NLML = NLML_tmp[0][0]

    
            
    if USE_SciKit_Learn:

        SciKit_Learn_results_dir = "./scikit_learn_results/"
        
        # Get SciKit Learn prediction data
        filename = os.path.join(SciKit_Learn_results_dir, "predMean.npy")
        skl_predMean = np.load(filename)

        filename = os.path.join(SciKit_Learn_results_dir, "predStd.npy")
        skl_predStd = np.load(filename)
        
        if inputDim == 1:
            # Get posterior samples
            filename = os.path.join(SciKit_Learn_results_dir, "samples.npy")
            skl_samples = np.load(filename)

        #filename = os.path.join(SciKit_Learn_results_dir, "NLML.npy")
        #skl_NLML = np.load(filename)


    if USE_GPyTorch:

        GPyTorch_results_dir = "./gpytorch_results/"
        
        # Get GPyTorch prediction data
        filename = os.path.join(GPyTorch_results_dir, "predMean.npy")
        gpy_predMean = np.load(filename)

        filename = os.path.join(GPyTorch_results_dir, "predStd.npy")
        gpy_predStd = np.load(filename)
        
        if inputDim == 1:
            # Get posterior samples
            filename = os.path.join(GPyTorch_results_dir, "samples.npy")
            gpy_samples = np.load(filename)
            gpy_samples = np.transpose(gpy_samples,[1,0])

        #filename = os.path.join(GPyTorch_results_dir, "NLML.npy")
        #gpy_NLML = np.load(filename)

        
    if USE_GPyTorch_SKI:

        GPyTorch_results_dir = "./gpytorch_results/"
        
        # Get GPyTorch prediction data
        filename = os.path.join(GPyTorch_results_dir, "SKI_predMean.npy")
        gpy_SKI_predMean = np.load(filename)

        filename = os.path.join(GPyTorch_results_dir, "SKI_predStd.npy")
        gpy_SKI_predStd = np.load(filename)
        
        if inputDim == 1:
            # Get posterior samples
            filename = os.path.join(GPyTorch_results_dir, "SKI_samples.npy")
            gpy_SKI_samples = np.load(filename)
            gpy_SKI_samples = np.transpose(gpy_SKI_samples,[1,0])

        #filename = os.path.join(GPyTorch_results_dir, "SKI_NLML.npy")
        #gpy_SKI_NLML = np.load(filename)
        
        
            

    ###                          ###
    ###       DISPLAY NLML       ###
    ###                          ###

    """
    print("\n")
    print("CppGPs NLML:        {:.4f}".format(NLML))
    if USE_SciKit_Learn:
        print("\nSciKit Learn NLML:  {:.4f}".format(skl_NLML))
    if USE_GPyTorch:
        print("\nGPyTorch NLML:      {:.4f}".format(gpy_NLML))
    print("\n")
    """
    
        
    ###                          ###
    ###       PLOT RESULTS       ###
    ###                          ###

    if inputDim == 1:

        """ ONE-DIMENSIONAL PLOTS """
        
        # CppGP results
        plt.figure()    
        plt.plot(inVals, predMean, 'C0', linewidth=2.0)
        alpha = 0.075
        for k in [1,2,3]:
            plt.fill_between(inVals, predMean-k*predStd, predMean+k*predStd, where=1>=0, facecolor="C0", alpha=alpha, interpolate=True)
        plt.plot(inVals, trueVals, 'C1', linewidth=1.0, linestyle="dashed")
        alpha_scatter = 0.5
        plt.scatter(obsX, obsY, alpha=alpha_scatter)
        for i in range(0,samples.shape[0]):
            plt.plot(inVals, samples[i,:], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")
        plt.suptitle("CppGP Implementation")    
        
        # Plot SciKit Learn results
        if USE_SciKit_Learn:
            plt.figure()
            plt.plot(inVals, skl_predMean, 'C0', linewidth=2.0)            
            for k in [1,2,3]:
                plt.fill_between(inVals, skl_predMean-k*skl_predStd, skl_predMean+k*skl_predStd, where=1 >= 0, facecolor="C0", alpha=alpha, interpolate=True, label=None)
            plt.plot(inVals, trueVals, 'C1', linewidth=1.0, linestyle="dashed")            
            plt.scatter(obsX, obsY, alpha=alpha_scatter)
            for i in range(0,skl_samples.shape[1]):
                plt.plot(inVals, skl_samples[:,i], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")
            plt.suptitle("SciKit Learn Implementation")

        
        # Plot GPyTorch results
        if USE_GPyTorch:
            plt.figure()
            plt.plot(inVals, gpy_predMean, 'C0', linewidth=2.0)
            alpha = 0.075
            for k in [1,2,3]:
                plt.fill_between(inVals, gpy_predMean-k*gpy_predStd, gpy_predMean+k*gpy_predStd, where=1 >= 0, facecolor="C0", alpha=alpha, interpolate=True, label=None)
            plt.plot(inVals, trueVals, 'C1', linewidth=1.0, linestyle="dashed")
            alpha_scatter = 0.5
            plt.scatter(obsX, obsY, alpha=alpha_scatter)
            for i in range(0,gpy_samples.shape[1]):
                plt.plot(inVals, gpy_samples[:,i], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")
            plt.suptitle("GPyTorch Implementation")

        # Plot GPyTorch SKI results
        if USE_GPyTorch_SKI:
            plt.figure()
            plt.plot(inVals, gpy_SKI_predMean, 'C0', linewidth=2.0)
            alpha = 0.075
            for k in [1,2,3]:
                plt.fill_between(inVals, gpy_SKI_predMean-k*gpy_SKI_predStd, gpy_SKI_predMean+k*gpy_SKI_predStd, where=1 >= 0, facecolor="C0", alpha=alpha, interpolate=True, label=None)
            plt.plot(inVals, trueVals, 'C1', linewidth=1.0, linestyle="dashed")
            alpha_scatter = 0.5
            plt.scatter(obsX, obsY, alpha=alpha_scatter)
            for i in range(0,gpy_SKI_samples.shape[1]):
                plt.plot(inVals, gpy_SKI_samples[:,i], 'C0', alpha=0.2, linewidth=1.0, linestyle="dashed")
            plt.suptitle("GPyTorch SKI Implementation")
            
        plt.show()    
    
    elif inputDim == 2:

        """ TWO-DIMENSIONAL PLOTS """
        
        # Flatten input values for compatibility with MatPlotLib's tri_surf
        plot_X_flat = []; plot_Y_flat = []
        R = inVals.shape[0]
        for n in range(0,R):
            plot_X_flat.append(inVals[n,0])
            plot_Y_flat.append(inVals[n,1])


        if (not USE_SciKit_Learn) and (not USE_GPyTorch):
            plot_count = 1
        elif (not USE_GPyTorch):
            plot_count = 2
        elif (not USE_SciKit_Learn):
            plot_count = 2
        else:
            plot_count = 3
            
        tri_fig = plt.figure()
        if plot_count == 1:
            tri_ax1 = tri_fig.add_subplot(111, projection='3d')
        elif plot_count == 2:
            tri_ax1 = tri_fig.add_subplot(121, projection='3d')
        elif plot_count == 3:
            tri_ax1 = tri_fig.add_subplot(131, projection='3d')
        linewidth = 0.1
        cmap = "Blues"

        # Plot CppGPs results
        tri_ax1.plot_trisurf(plot_X_flat,plot_Y_flat, predMean, cmap=cmap, linewidth=linewidth, antialiased=True)
        pred_title = "CppGPs"
        tri_ax1.set_title(pred_title, fontsize=24)

        if USE_SciKit_Learn:
            # Plot SciKit Learn results
            if plot_count == 2:
                tri_ax2 = tri_fig.add_subplot(122, projection='3d')
            elif plot_count == 3:
                tri_ax2 = tri_fig.add_subplot(132, projection='3d')
            tri_ax2.plot_trisurf(plot_X_flat,plot_Y_flat, skl_predMean, cmap=cmap, linewidth=linewidth, antialiased=True)
            soln_title = "SciKit Learn"
            tri_ax2.set_title(soln_title, fontsize=24)

        if USE_GPyTorch:
            # Plot GPyTorch Learn results
            if plot_count == 2:
                tri_ax3 = tri_fig.add_subplot(122, projection='3d')
            elif plot_count == 3:
                tri_ax3 = tri_fig.add_subplot(133, projection='3d')
            tri_ax3.plot_trisurf(plot_X_flat,plot_Y_flat, gpy_predMean, cmap=cmap, linewidth=linewidth, antialiased=True)
            soln_title = "GPyTorch Learn"
            tri_ax3.set_title(soln_title, fontsize=24)
        
        # Remove axes from plots
        remove_axes(tri_ax1)
        if USE_SciKit_Learn:
            remove_axes(tri_ax2)
        if USE_GPyTorch:
            remove_axes(tri_ax3)

        if plot_count == 2:
            if USE_SciKit_Learn:
                canvas = bind_axes(tri_fig, tri_ax1, tri_ax2)
            else:
                canvas = bind_axes(tri_fig, tri_ax1, tri_ax3)
        elif plot_count == 3:
            canvas = bind_axes_3(tri_fig, tri_ax1, tri_ax2, tri_ax3)
            


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


        # Plot Predictive Mean
        linewidth = 0.1; alpha = 0.85
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_zoom, cmap=cmap, linewidth=linewidth, antialiased=True, alpha=alpha)


        # One Standard Deviation
        linewidth = 0.075; alpha = 0.2
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_plus_std, cmap=cmap, linewidth=linewidth, antialiased=True,alpha=alpha)
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_minus_std, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)

        # Two Standard Deviations
        linewidth = 0.05; alpha = 0.1
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_plus_std2, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_minus_std2, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)

        # Three Standard Deviations
        linewidth = 0.01; alpha = 0.01
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_plus_std3, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)
        tri2_ax1.plot_trisurf(plot_X_zoom,plot_Y_zoom, predMean_minus_std3, cmap=cmap, linewidth=linewidth,antialiased=True,alpha=alpha)

        # Scatter plot of training observations
        alpha = 0.4
        tri2_ax1.scatter(obsX_x_zoom, obsX_y_zoom, obsY_zoom, c='k', marker='o', s=15.0, alpha=alpha)

        # Add title to plot
        plt.suptitle("CppGPs Predictive Uncertainty", fontsize=24)
        
        # Remove axes from plot
        remove_axes(tri2_ax1)

        # Display plots
        plt.show()
    

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

        
# Update rotation
def update_rotation(target_ax, source_ax):
    target_ax.view_init(elev=source_ax.elev, azim=source_ax.azim)

# Update zoom
def update_zoom(target_ax, source_ax):
    target_ax.set_xlim3d(source_ax.get_xlim3d())
    target_ax.set_ylim3d(source_ax.get_ylim3d())
    target_ax.set_zlim3d(source_ax.get_zlim3d())

# Bind MatPlotLib Axes [2 Plots]
def bind_axes(fig, ax1, ax2):
    
    # Bind axes for comparison
    def bind_on_move(event):
        if event.inaxes == ax1:
            if ax1.button_pressed in ax1._rotate_btn:
                update_rotation(ax2,ax1)
            elif ax1.button_pressed in ax1._zoom_btn:
                update_zoom(ax2,ax1)
        elif event.inaxes == ax2:
            if ax2.button_pressed in ax2._rotate_btn:
                update_rotation(ax1,ax2)
            elif ax2.button_pressed in ax2._zoom_btn:
                update_zoom(ax1,ax2)
        else:
            return
        fig.canvas.draw_idle()
    canvas = fig.canvas.mpl_connect('motion_notify_event', bind_on_move)
    return canvas

# Bind MatPlotLib Axes [3 Plots]
def bind_axes_3(fig, ax1, ax2, ax3):
    
    # Bind axes for comparison
    def bind_on_move(event):
        if event.inaxes == ax1:
            if ax1.button_pressed in ax1._rotate_btn:
                update_rotation(ax2,ax1)
                update_rotation(ax3,ax1)
            elif ax1.button_pressed in ax1._zoom_btn:
                update_zoom(ax2,ax1)
                update_zoom(ax3,ax1)
        elif event.inaxes == ax2:
            if ax2.button_pressed in ax2._rotate_btn:
                update_rotation(ax1,ax2)
                update_rotation(ax3,ax2)
            elif ax2.button_pressed in ax2._zoom_btn:
                update_zoom(ax1,ax2)
                update_zoom(ax3,ax2)
        elif event.inaxes == ax3:
            if ax3.button_pressed in ax3._rotate_btn:
                update_rotation(ax1,ax3)
                update_rotation(ax2,ax3)
            elif ax3.button_pressed in ax3._zoom_btn:
                update_zoom(ax1,ax3)
                update_zoom(ax2,ax3)
        else:
            return
        fig.canvas.draw_idle()
    canvas = fig.canvas.mpl_connect('motion_notify_event', bind_on_move)
    return canvas


# Run main() function when called directly
if __name__ == '__main__':
    main()
