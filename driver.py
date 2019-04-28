#!/usr/bin/env python
# coding: utf-8
from __future__ import division, print_function
import time
import numpy as np
import kalman

# Example showing how to
if __name__ == "__main__":

    timestamps = np.array([0.1, 0.2])
    velocity   = np.array([3.0, 3.0])
    acc_ned    = np.array(
        [
            [10., 10.],
            [10., 10.]
        ]
    )

    filter_parameters = {
        "measurement_noise"  : 0.1,
        "process_noise"      : 0.5,
        "initial_covariance" : 0.1,
        "damping"            : 0.3,
    },

    kalman_result = kalman.run_kalman_filter(
        acc_ned[0,:],    # north acceleration
        acc_ned[1,:],    # east accelleration
        timestamps,      # timestamps
        velocity,        # GPS velocities
        filter_parameters
    )

    vel_north = kalman_result[0]
    vel_east  = kalman_result[1]

    print("Walltime: {}".format(time.time()-t))
