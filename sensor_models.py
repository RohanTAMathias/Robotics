import numpy as np
# import scipy
from scipy.optimize import curve_fit
from scipy.optimize import brute
from numpy import loadtxt
from matplotlib.pyplot import subplots, show

# Sensor Object Class
#------------------------------------------------------------------------------------------------------
class Sensor_Model:
    '''
    Sensor Model constructor: Class for a singular sensor.
     inputs -> sensor model function (f), sensor dataset (sensor_d), distance dataset (dist_d), 
               No. standard deviations for outlier removal (std_devs), number of outlier removal iterations (outlier_iters)
    outputs -> None
    '''   
    def __init__(self, f, sensor_d, dist_d, std_devs=3, outlier_iters=2):
        
        self.f = f
        self.dist_d = dist_d
        self.__sensor_d__ = sensor_d
        self.__dx__ = 0.001
        self.__Ndata__ = len(dist_d)
        
        self.new_d = self.dist_d
        self.new_s = self.__sensor_d__
        
        #Calculates the best model for the sensor function (f)
        self.params, self.cov = curve_fit(f, self.new_d, self.new_s)
        self.__z_fit__ = f(self.new_d, *self.params)
        self.error = self.__z_fit__ - self.new_s
            #for more than 1 iteration
        for j in range(outlier_iters):
            #creates mask for outliers to be removed
            mask = abs(self.error) < np.std(self.error) * std_devs
            #removing error terms
            self.new_d = self.new_d[mask]
            self.new_s = self.new_s[mask]
            #recreates the parameters with the processed data
            self.params[:], self.cov = curve_fit(f, self.new_d, self.new_s)
            self.__z_fit__ = f(self.new_d, *self.params)
            self.error = self.__z_fit__ - self.new_s   
        
        #final model and the error between it and the total dataset
        self.__z_fit__ = f(self.dist_d, *self.params)
        self.error = self.__z_fit__ - self.__sensor_d__
        
        #creates the variance look up table
        self.var_array = []
        self.x_var_array = []
        index = 100
        for i in range(index, self.__Ndata__-index):
            self.var_array.append(np.var(self.error[i-index:i+index]))
            self.x_var_array.append(self.dist_d[i])
        
    
    '''
    Sensor Model noise variance: Getter for the variance function.
     inputs -> sensor output value (z)
    outputs -> Current variance from the sensors variance lookup table (array)
    '''   
    def noise_var(self, x):
        if x <= self.x_var_array[0]: var = self.var_array[0]
        elif x >= self.x_var_array[-1]: var = self.var_array[-1]
        else:
            for i in range(len(self.var_array)-1):    
                if x > self.x_var_array[i] and x <= self.x_var_array[i+1]:
                    var = self.var_array[i]
                    break
        return var
    
    
    def f_handler(self, x):
        return self.f(x, *self.params)
    
    '''
    Sensor Model linearize: Linearizes the sensor model at the best guess of the current x position.
     inputs -> estimate distance value (x)
    outputs -> gradient of the linearized function (a), y-intercept of the linearized function (b)
    '''          
    def linearize(self, x):
        a = (self.f(x+self.__dx__, *self.params) - self.f(x, *self.params)) / self.__dx__
        b = self.f(x, *self.params) - a*x
        return a, b
    
    '''
    Sensor Model plot data: Plotter for the internal sensor functions and data.
     inputs -> axes for the current plot (axes), axes index for the current plot (axes_index), title for the subplot (title)
    outputs -> No return values, Outputs a subplot
    '''   
    def plot_data(self, axes, axes_index, title):
        ax1 = axes[0][*axes_index]
        ax2 = axes[1][*axes_index]
        ax3 = axes[2][*axes_index]
        
        alpha = 0.2
        ax1.plot(self.dist_d, self.__sensor_d__, '.', alpha=alpha)
        ax1.plot(self.new_d, self.new_s, '.', alpha=alpha)
        ax1.plot(self.dist_d, self.__z_fit__)
        ax1.set_title(title, fontdict = {'fontsize' : 10})
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_ylim((0, None))
        ax1.set_xlim((0, None))
        
        ax2.plot(self.dist_d, self.error)
        ax2.set_title(title + " Error", fontdict = {'fontsize' : 10})
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('deviation (m)')
        
        ax3.set_title(title + " Error")
        ax3.hist(self.error, bins=80, alpha=0.5, color='red')  # You can adjust the number of bins and color
        ax3.set_xlabel('Deviation (m)')
        ax3.set_ylabel('Frequency')
        ax3.set_xlim((-1, 1))


# Main sensor initialiser code        
#------------------------------------------------------------------------------------------------------
# init_sensors: Initialises all six sensors based on set functions
def init_sensors():
    filename = 'partA/calibration.csv'
    data = loadtxt(filename, delimiter=',', skiprows=1)

    def f1(x, a, b): #sonar function (linear)
        return a*x + b
    def f2(x, a, b, c): #IR1 ,2 and 3 function (inverse x)
        return a + b*x + c/x
    def f3(x, a, b, c, d): #IR4 function (sin(e^x))
        return a*np.sin(np.exp(-b*x+d))+c

    index, time, dist, v_com, ir1, ir2, ir3, ir4, sonar1, sonar2 = data.T
    
    Sonar1 = Sensor_Model(f1, sonar1, dist, outlier_iters=3)
    Sonar2 = Sensor_Model(f1, sonar2, dist, outlier_iters=2)
    IR1 = Sensor_Model(f2, ir1, dist, outlier_iters=2)
    IR2 = Sensor_Model(f2, ir2, dist, outlier_iters=2)
    IR3 = Sensor_Model(f2, ir3, dist, outlier_iters=2)
    IR4 = Sensor_Model(f3, ir4, dist, outlier_iters=2)
    
    # fig, ax = subplots(1)
    # sensor = IR4
    # ax.plot(sensor.x_var_array, sensor.var_array)
    # show()
    # print(np.var(sensor.error))
    
    return [Sonar1, Sonar2, IR1, IR2, IR3, IR4]

# plot_sensors: Plots the sensor data by calling each sensors plot_data() method.
def plot_sensors(sensors):
    fig1, axes1 = subplots(2, 3)
    fig2, axes2 = subplots(2, 3)
    fig3, axes3 = subplots(2, 3)
    axes = [axes1, axes2, axes3]
    
    sensors[0].plot_data(axes, [0,0], "IR1")
    sensors[1].plot_data(axes, [0,1], "IR2")
    sensors[2].plot_data(axes, [0,2], "IR3")
    sensors[3].plot_data(axes, [1,0], "IR4")
    sensors[4].plot_data(axes, [1,1], "Sonar1")
    sensors[5].plot_data(axes, [1,2], "Sonar2")
    
    fig1.legend(["Raw Data", "Processed Data", "Sensor Model"])
    fig1.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.5)
    fig2.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.5)
    fig3.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.5)
    
    show()