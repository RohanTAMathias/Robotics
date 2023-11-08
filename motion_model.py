import numpy as np
from numpy import gradient
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show

class Motion_Model:
    '''
    Motion Model constructor: Class for easier access to the motion model
     inputs -> Commanded velocity dataset (v_com), Actual x positon dataset (x_meas), Time dataset (t)
    outputs -> None
    '''   
    def __init__(self, v_com, x_meas, t):
        # data arrays
        self.__v_com__ = v_com
        self.__t__ = t
        self.__dt__ = t[1:] - t[0:-1]
        self.__x_meas__ = x_meas
        self.__Ndata__ = len(t)
        self.__w__ = np.ndarray(self.__Ndata__, np.float64)
        self.__m_model__ = np.ndarray(self.__Ndata__, np.float64)
        
        self.var_W = 0
    
    '''
    Motion Model function handler g: Calculates the output from the motion model function.
     inputs -> previous distance (x), previous commanded velocity (u), timestep (dt)
    outputs -> motion model prediction (x_est)
    '''
    def g(self, x, u_prev, dt):
        return x + u_prev*dt
        
    '''
    Motion Model calculate noise variance: Calculates the noise variance based on the calibration dataset.
     inputs -> None
    outputs -> Model noise variance (var_W)
    '''    
    def calc_model_noise_var(self):
        self.__m_model__[0] = 0 
        
        for i in range(1, self.__Ndata__):
            self.__m_model__[i] = self.g(self.__x_meas__[i-1], self.__v_com__[i-1], self.__dt__[i-1])
        
        self.__w__[0] = 0
        self.__w__[1:] = self.__x_meas__[1:] - self.__m_model__[0:-1]
        self.var_W = np.var(self.__w__)
        
        return self.var_W
        
    '''
    Motion Model plot model: Plots the motion model data.
     inputs -> None
    outputs -> No Returns, outputs three plots: distance, speed and error
    '''            
    def plot_model(self):
        v_est = gradient(self.__x_meas__, self.__t__)
        v_model = gradient(self.__m_model__, self.__t__)
        
        fig, axes = subplots(1)
        axes.plot(self.__t__, self.__v_com__, '-k', label='commanded speed')
        axes.plot(self.__t__, v_est, color='tab:red', label='measured speed')
        axes.plot(self.__t__, v_model, label="Motion Model")
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Speed (m/s)')
        axes.legend()
        plt.ylim([-0.6,0.6])
        #plt.xlim([0,300])
        fig.canvas.manager.set_window_title('Speed Model')
        
        
        # Distance model
        fig, axes = subplots(1)
        #axes.plot(time, v_com*time, label='commanded distance')
        axes.plot(self.__t__, self.__x_meas__, label='measured distance')
        axes.plot(self.__t__, self.__m_model__, label='model distance')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Distance (m)')
        axes.legend()
        plt.ylim([-0.1,3.8])
        #plt.xlim([0,300])
        fig.canvas.manager.set_window_title('Distance Model')
        
        # Process noise variance
        fig, axes = subplots(1)
        axes.plot(self.__t__, self.__w__)
        axes.set_xlabel('Time')
        axes.set_ylabel('Process Noise (m)')
        #plt.ylim([-1.6,1.6])
        #plt.xlim([0,300])
        fig.canvas.manager.set_window_title('Error Model')
        
        show()