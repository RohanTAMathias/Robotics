"""
ENMT482 Robotics Assignment 1, Part A

Author: Harrison Johnson (hjo48)
Date: 13/09/2023
"""

import numpy as np
from matplotlib.pyplot import subplots, show
from motion_model import *
from sensor_models import *


# Get Data
#-----------------------------------------------------------------------------------
# File to load
filename = 'partA/training1.csv'
# Load data
data = loadtxt(filename, delimiter=',', skiprows=1)
# Split into columns
index, t, x_meas, v_com, ir1, ir2, ir3, ir4, sonar1, sonar2 = data.T
Nsteps = len(index)


# Initialise Sensors
#-----------------------------------------------------------------------------------
Sonar1, Sonar2, IR1, IR2, IR3, IR4 = init_sensors()
#plot_sensors([IR1, IR2, IR3, IR4, Sonar1, Sonar2])


# Get Motion Model
#-----------------------------------------------------------------------------------
motion_model = Motion_Model(v_com, x_meas, t)
Var_W = motion_model.calc_model_noise_var()
#motion_model.plot_model()

## FOR Test.csv ONLY
# filename = 'partA/test.csv'
# # Load data
# data = loadtxt(filename, delimiter=',', skiprows=1)
# # Split into columns
# index, t, v_com, ir1, ir2, ir3, ir4, sonar1, sonar2 = data.T
# Nsteps = len(index)


# Do the EKF (Extended Kalmann FIlter)
#-----------------------------------------------------------------------------------
sensors = [IR3, Sonar1, IR4] #MUST MATCH
z_input = [ir3, sonar1, ir4] #MUST MATCH 
num_sensors = len(sensors)         

P_0 = 0.0001 # variance at the start position
x_0 = 0.01 # estimate starting position
dt = t[1:] - t[0:-1]

x_Post = np.zeros((Nsteps,), dtype=np.float64)
x_Prior = np.zeros((Nsteps,), dtype=np.float64)
x_Est = np.zeros((Nsteps,), dtype=np.float64)
x_Post[0] = x_0

P_Post = np.zeros((Nsteps,), dtype=np.float64)
P_Prior = np.zeros((Nsteps,), dtype=np.float64)
P_Est = np.zeros((Nsteps,), dtype=np.float64)
P_Post[0] = P_0

for i in range(1, Nsteps):
    #Predicts the position from the motion model
    x_Prior[i] = motion_model.g(x_Post[i-1], v_com[i-1], dt[i-1])
    P_Prior[i] = P_Post[i-1] + Var_W
    
    #Estimates the position from the sensors
    Est_den = 0
    for j in range(num_sensors):
        a, b = sensors[j].linearize(x_Post[i-1])
        xZ_Est = (z_input[j][i] - b) / a
        PZ_Est = sensors[j].noise_var(x_Prior[i]) / a**2
         
        x_Est[i] += xZ_Est / PZ_Est
        Est_den += 1.0 / PZ_Est
    
    x_Est[i] = x_Est[i] / Est_den
    P_Est[i] = 1.0 / Est_den
    
    #Combines the prediction and estimate and updates the posterior
    x_Post[i] = (x_Est[i]/P_Est[i] + x_Prior[i]/P_Prior[i]) / ((1.0/P_Est[i])+(1.0/P_Prior[i]))
    P_Post[i] = 1.0 / ((1.0/P_Prior[i]) + (1.0/P_Est[i]))
    
    # x_Post[i] = x_Prior[i] # For ignoring the sensors
    # P_Post[i] = P_Prior[i]


for i in range(num_sensors):
    print("Sensor {}: params {}".format(i+1, sensors[i].params))

# Plotting
#-----------------------------------------------------------------------------------
fig, axes = subplots(2,1)
axes[0].plot(t, x_Post, label='X Posterior')
# axes[0].plot(t, x_Prior, label='X Prior')
# axes[0].plot(t, x_Est, label='X Estimate')
# axes[0].plot(t, x_meas, label='X Real')
# axes[0].plot(t, sonar2, label='z sonar2')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Distance (m)')
axes[0].legend()
axes[0].set_ylim([0,None])
#axes[0].set_xlim([0,300])
ax2 = axes[0].twinx()
ax2.plot(t, np.sqrt(P_Post), '--', label='Std Dev')
ax2.set_ylabel('Standard Deviation (m)')
ax2.set_ylim([0, 0.2])
ax2.legend(loc='upper left')

# axes[1].plot(t, P_Post, label='Posterior Variance')
axes[1].plot(t, P_Prior, label='Prior Variance')
axes[1].plot(t, P_Est, label='Sensor Variance')
axes[1].plot(t, P_Post, label='Posterior Variance')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Variance (m**2)')
axes[1].legend()
axes[1].set_ylim([-0.01,0.4])
#plt.xlim([0,300])
fig.canvas.manager.set_window_title('EKF')

fig, axes = subplots(1)
axes.plot(t, gradient(x_Prior, t), '-', label='Model Speed', color='tab:blue')
axes.plot(t, gradient(x_meas, t), '-', label='Estimate Speed', color='tab:red')
axes.plot(t, v_com, '-k', label='Commanded Speed')
# axes.plot(t, gradient(x_Est, t), label='Estimated Speed')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Speed (m/s)')
axes.legend()
axes.set_ylim([-1,1.5])
# axes.set_xlim([0, 85])


show()