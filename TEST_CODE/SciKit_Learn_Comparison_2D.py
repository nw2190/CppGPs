import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import csv
import time

from mpl_toolkits.mplot3d import Axes3D

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
            i1, i2, t, m, v = row
            inVals.append([i1,i2])
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
            x1, x2, y = row
            obsX.append([x1,x2])
            obsY.append(y)
    obsX = np.array(obsX).astype(np.float32)
    obsY = np.array(obsY).astype(np.float32)



    ### SCIKIT LEARN IMPLEMENTATION
    X = np.reshape(obsX, [-1, 2])
    Y = np.reshape(obsY, [-1])
    Xtest = np.reshape(inVals, [-1, 2])

    """
    ### VERIFY DATA HAS BEEN PARSED CORRECTLY
    for i in range(0,10):
        print(" X = ({:.4f} , {:.4f})   y = {:.4f}".format(X[i,0],X[i,1],Y[i]))

    print("\n")
    for i in range(0,10):
        print(" inVals = ({:.4f} , {:.4f})   trueVals = {:.4f}".format(inVals[i,0],inVals[i,1],trueVals[i]))
    """
        
    # Model parameters
    n_restarts = 0
    normalize_y = False
    use_white_noise = True
    RBF_bounds = [0.01, 100.0]
    Noise_bounds = [0.00001, 10.0]
    jitter = 1e-7

    if use_white_noise:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(RBF_bounds[0],RBF_bounds[1])) + WhiteKernel(noise_level=1, noise_level_bounds=(Noise_bounds[0], Noise_bounds[1]))
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
    #model_samples = model.sample_y(Xtest, samples.shape[0])

    NLML = -model.log_marginal_likelihood()
    print("NLML:   {:.4f}\n".format(NLML))


    

    """
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
    """









    """ TRISURF ATTEMPT """
    plot_X_flat = []
    plot_Y_flat = []
    R = inVals.shape[0]
    for n in range(0,R):
        plot_X_flat.append(inVals[n,0])
        plot_Y_flat.append(inVals[n,1])
                
    tri_fig = plt.figure()
    tri_ax1 = tri_fig.add_subplot(121, projection='3d')
    linewidth = 0.1
    cmap = 'plasma'
    tri_ax1.plot_trisurf(plot_X_flat,plot_Y_flat, predMean, cmap=cmap, linewidth=linewidth, antialiased=True)
    #tri_ax1.plot_trisurf(plot_X_flat,plot_Y_flat, trueVals, cmap=cmap, linewidth=linewidth, antialiased=True, alpha=0.1)

    pred_title = "CppGPs"
    tri_ax1.set_title(pred_title)

    
    tri_ax2 = tri_fig.add_subplot(122, projection='3d')
    tri_ax2.plot_trisurf(plot_X_flat,plot_Y_flat, mean, cmap=cmap, linewidth=linewidth, antialiased=True)

    soln_title = "SciKit Learn"
    tri_ax2.set_title(soln_title)

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

    # Set initial view angles
    #tri_ax1.view_init(view_elev, view_angle)
    #tri_ax2.view_init(view_elev, view_angle)
    plt.show()
    


# Run main() function when called directly
if __name__ == '__main__':
    main()
