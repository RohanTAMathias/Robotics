"""Particle filter sensor and motion model implementations.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy.random import randn, normal
from utils import gauss, wraptopi, angle_difference


def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt):
    """Apply motion model and return updated array of particle_poses.

    Parameters
    ----------

    particle_poses: an M x 3 array of particle_poses where M is the
    number of particles.  Each pose is (x, y, theta) where x and y are
    in metres and theta is in radians.

    speed_command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    dt is the time step (s).

    Returns
    -------
    An M x 3 array of updated particle_poses.

    """

    M = particle_poses.shape[0]
    
    # TODO.  For each particle calculate its predicted pose plus some
    # additive error to represent the process noise.  With this demo
    # code, the particles move in the -y direction with some Gaussian
    # additive noise in the x direction.  Hint, to start with do not
    # add much noise.
    dx = (odom_pose[0] - odom_pose_prev[0])
    dy = (odom_pose[1] - odom_pose_prev[1])
    
    local_theta1 = np.arctan2(dy, dx) - odom_pose_prev[2]
    d = np.sqrt(dy**2 + dx**2) 
    local_theta2 = odom_pose[2] - odom_pose_prev[2] - local_theta1

    #assuming error in local pose change is small so can approx as global

    #adding in error
    alpha1 = 0.00015
    alpha2 = 0.00014
    alpha3 = 0.001
    alpha4 = 0.002

    rand_d = alpha3*(abs(local_theta1) + abs(local_theta2)) + alpha4*d
    rand_theta1 = alpha1*abs(local_theta1) + alpha2*d
    rand_theta2 = alpha1*abs(local_theta2) + alpha2*d


    for m in range(M):
        # d = normal(d, rand_d)
        # local_theta1 = normal(local_theta1, rand_theta1)
        # local_theta2 = normal(local_theta2, rand_theta2)

        particle_poses[m, 0] += d * np.cos(particle_poses[m, 2] + local_theta1) + randn(1) * 0.007
        particle_poses[m, 1] += d * np.sin(particle_poses[m, 2] + local_theta2) + randn(1) * 0.007
        particle_poses[m, 2] += local_theta1 + local_theta2 + randn(1) * (1/300)
    
    return particle_poses


def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system.

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system.

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """

    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)
    
    # TODO.  For each particle calculate its weight based on its pose,
    # the relative beacon pose, and the beacon location.
    r = np.sqrt(beacon_pose[0]**2 + beacon_pose[1]**2)
    theta = np.arctan2(beacon_pose[1], beacon_pose[0])

    for m in range(M):
        #relative distance and location of particle with respect to the beacon
        r_particle = np.sqrt((beacon_loc[0] - particle_poses[m, 0])**2 + (beacon_loc[1] - particle_poses[m, 1])**2)
        theta_particle = angle_difference(particle_poses[m,2], np.arctan2(beacon_loc[1] - particle_poses[m,1], beacon_loc[0] - particle_poses[m,0]))
        #angle diff for same thing (use angle diff)

        #calculate error between particle and robot
        #r_error = 

        #calculate std_deviation
        std_dev = 0.02 + abs(wraptopi(theta + np.pi/2))*0.005 + r*0.005
        #print(std_dev)

        #calculate likelihood of range using error and std (use gauss fn)
        range_likelihood = gauss(r- r_particle, 0, std_dev)
        #calculate bearing angle likelihood with same fn
        theta_likelihood = gauss(angle_difference(theta, theta_particle), 0, std_dev)
        #weighing is pdf range * pdf theta

        #particle_weights[m] = gauss(r - r_particle) * gauss(angle_difference(theta, theta_particle))
        particle_weights[m] = range_likelihood * theta_likelihood 
        
    return particle_weights
