#include <iostream>
#include <armadillo>
#include <complex>

#include <math.h>
#include <python2.7/Python.h>
#include <numpy/arrayobject.h>

// struct for kalman filter parameters
struct KalmanParams {
    float measurement_noise;
    float process_noise;
    float initial_covariance;
    float damping;
};

static PyObject* run_kalman_filter(double *acc_north, double *acc_east, double *timestamp,
                                   double *velocity, int size, KalmanParams p) {
    /* Implements a simple kalman filter estimating velocity
       from IMU acceleration and GPS velocities. This code is
       meant to highlight how to interface numpy.
    */
    
    arma::vec acc;           /* data vector */
    arma::vec pred_mean;     /* predicted mean */
    arma::vec meas_mat_lin;  /* predicted mean */
    arma::vec inno;          /* innovation */

    arma::mat cov_est = arma::eye<arma::mat>(2,2);
    arma::mat eye = arma::eye<arma::mat>(2,2);

    arma::vec x_est(2, arma::fill::zeros);
    arma::mat pred_cov(2, 2, arma::fill::zeros);
    arma::mat kal_gain(2, 2, arma::fill::zeros);
    arma::mat mat_temp(2, 2, arma::fill::zeros);
    arma::mat cov_pred_meas(2, 2, arma::fill::zeros);

    // setup python lists to return results
    PyObject *vel_est_x = PyList_New(0);
    PyObject *vel_est_y = PyList_New(0);

    PyList_Append(vel_est_x, PyFloat_FromDouble(0.));
    PyList_Append(vel_est_y, PyFloat_FromDouble(0.));

    PyObject* result = PyList_New(0);

    // initialize variables
    cov_est *= p.initial_covariance;

    double dt = 0.;
    double k1 = 0.;
    double k0 = 0.;

    // start filter loop
    for(int i=1; i<size; ++i) {

        // determine delta t
        dt = timestamp[i] - timestamp[i-1];

        // setup the acc measurement vector
        acc << acc_north[i-1] << acc_east[i-1];

        // prediction step
        pred_mean = exp((-1.) * dt * p.damping) * x_est + dt * acc;
        pred_cov = pow(exp((-1.) * dt * p.damping), 2.) * cov_est + p.process_noise * pow(dt, 2.) * eye;

        if (arma::norm(pred_mean) != 0.) {

            meas_mat_lin = pred_mean / arma::norm(pred_mean);

            // covariance of predicted measurement
            cov_pred_meas = meas_mat_lin.t() * pred_cov * meas_mat_lin + p.measurement_noise;

            // compute innovation
            inno = velocity[i] - arma::norm(pred_mean);

            // compute kalman gain
            kal_gain = (meas_mat_lin.t() * pred_cov) / arma::as_scalar(cov_pred_meas);

            // estimated mean
            x_est = pred_mean + kal_gain.t() * arma::as_scalar(inno);

            // estimated covariance
            k0 = arma::as_scalar(kal_gain.col(0));
            k1 = arma::as_scalar(kal_gain.col(1));

            mat_temp << k0*k0 << k0*k1 << arma::endr
                     << k1*k0 << k1*k1 << arma::endr;

            cov_est = pred_cov - mat_temp * arma::as_scalar(cov_pred_meas);

        } else {

            x_est = pred_mean;
            cov_est = pred_cov;

        }

        PyList_Append(vel_x, PyFloat_FromDouble(x_est[0]));
        PyList_Append(vel_y, PyFloat_FromDouble(x_est[1]));
        cov_est = (cov_est + cov_est.t()) / 2.;

    }
    
    // create a list of list and return it
    PyList_Append(result, vel_x);
    PyList_Append(result, vel_y);

    return result;
};


static PyObject* run_kalman_filter_wrapper(PyObject *self, PyObject *args) {
    // Parse arguments from Python out of the args parameter
    PyObject *acc_north_arg = NULL;
    PyObject *acc_east_arg  = NULL;
    PyObject *timestamp_arg = NULL;
    PyObject *velocity_arg  = NULL;
    PyObject *param_arg     = NULL;

    if (!PyArg_ParseTuple(args, "OOOOO", &acc_north_narg, &acc_east_arg, &timestamp_arg, &velocity_arg, &param_arg))  {
        return NULL;
    }

    double *acc_north, *acc_east, *timestamp, *vel;

    int size = PyArray_SIZE(PyArray_FROM_OTF(timestamp_arg, NPY_DOUBLE, NPY_IN_ARRAY));

    acc_north = (double *)PyArray_DATA(
        PyArray_FROM_OTF(acc_north_arg, NPY_DOUBLE, NPY_IN_ARRAY));
    acc_east  = (double *)PyArray_DATA(
        PyArray_FROM_OTF(acc_east_arg, NPY_DOUBLE, NPY_IN_ARRAY));
    timestamp = (double *)PyArray_DATA(
        PyArray_FROM_OTF(timestamp_arg,  NPY_DOUBLE, NPY_IN_ARRAY));
    velocity  = (double *)PyArray_DATA(
        PyArray_FROM_OTF(velocity_arg,  NPY_DOUBLE, NPY_IN_ARRAY));

    // Parameters are passed as a Python dict, turn them into a struct
    struct KalmanParams p;
    p.measurement_noise  = PyFloat_AsDouble(
        PyDict_GetItem(pars, PyString_FromString("measurement_noise")));
    p.process_noise      = PyFloat_AsDouble(
        PyDict_GetItem(pars, PyString_FromString("process_noise")));
    p.initial_covariance = PyFloat_AsDouble(
        PyDict_GetItem(pars, PyString_FromString("initial_covariance")));
    p.damping            = PyFloat_AsDouble(
        PyDict_GetItem(pars, PyString_FromString("damping")));

    // Print Kalman filter parameters
    std::cout << "--------------------------------\n\n";
    std::cout << "SIMPLE KALMAN FILTER\n";
    std::cout << "--------------------------------\n\n";
    std::cout << "Parameters: \n";
    std::cout << "- - - - - - \n";
    std::cout << "measurement_noise  : " << p.measurement_noise << "\n";
    std::cout << "process_noise      : " << p.process_noise << "\n";
    std::cout << "initial_covariance : " << p.initial_covariance << "\n";
    std::cout << "damping            : " << p.damping << "\n";

    // Now we can call the function with the arguments from the Python call.
    PyObject* result = run_kalman_filter(acc_north, acc_east, timestamp, velocity, size, p);

    return result;
};


// Mapping between python and c function names. 
static PyMethodDef kalman_module_methods[] = {
    {"run_kalman_filter", run_kalman_filter_wrapper, METH_VARARGS},
    {NULL, NULL}
};  

// Module initialisation routine.
PyMODINIT_FUNC initkalman(void) {
    // Init module.
    (void) Py_InitModule("kalman", kalman_module_methods);
    import_array();
};
