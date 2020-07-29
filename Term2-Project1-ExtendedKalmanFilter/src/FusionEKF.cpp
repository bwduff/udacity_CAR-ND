#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

	/**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
	*/

	//FROM FORUM MENTOR:  You can initialize your F, P, H here and noise_ax and noise_ay. In the prediction part, it says that we should use noise_ax = 9 and noise_ay = 9 for your Q matrix. 
	/**
	* Init Initializes Kalman filter
	* @param x_in Initial state
	* @param P_in Initial state covariance
	* @param F_in Transition matrix
	* @param H_in Measurement matrix
	* @param R_in Measurement covariance matrix
	* @param Q_in Process covariance matrix
	*/

	H_laser_ << 1, 0, 0, 0, //Setting measurement matrix H for laser readings only
				0, 1, 0, 0;
	
	Hj_ <<  0,0,0,0, //Initiliazing for later jacobian calculations.
			0,0,0,0,
			0,0,0,0;

	//Initial state trans. mtx.
	ekf_.F_ = MatrixXd(4, 4);
	ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

	// Initial state covar. mtx.
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ <<  1, 0, 0, 0, //QUESTION: Why 1000? Ms conversion?
				0, 1, 0, 0,
				0, 0, 1000, 0,
				0, 0, 0, 1000;

	// Accereleration noise components, to be used in process covariance Q.
	//Made this a private member variable (which it should be), but design needs to be better.
	noise_ax = 9;
	noise_ay = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
	  * DIRECTIVES:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 0, 0; //Changed velocities from default of 1 to 0.

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Directive: Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = measurement_pack.raw_measurements_[0]; //Range 
  	  float phi = measurement_pack.raw_measurements_[1]; //Angle
  	  float rho_dot = measurement_pack.raw_measurements_[2]; //Range rate
	  float x = rho * cos(phi); 
	  float y = rho * sin(phi);
	  float vx = rho_dot * cos(phi);
	  float vy = rho_dot * sin(phi);
	  ekf_.x_ << x, y, vx , vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      //Initialize state with lidar data (location only)
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1],
      0, 0;
    }

	previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * DIRECTIVES:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  float dt_2 = delta_t * delta_t;
  float dt_3 = dt_2 * delta_t;
  float dt_4 = dt_3 * delta_t;
    
  //Update state trans. mtx. F for new time
  ekf_.F_(0,2) = delta_t;
  ekf_.F_(1,3) = delta_t;
    
  //Update process noise covar. mtx, Q.
  ekf_.Q_ = MatrixXd(4,4);
  ekf_.Q_ <<  dt_4/4 * noise_ax, 0, dt_3/2 * noise_ax, 0,
              0, dt_4/4 * noise_ay, 0, dt_3/2 * noise_ay,
              dt_3/2 * noise_ax, 0, dt_2 * noise_ax, 0,
              0, dt_3/2 * noise_ay, 0, dt_2 * noise_ay;
                
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * DIRECTIVES:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */
  // Radar updates  
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_); //Storing jacobian matrix within H to later be used for UpdateEKF. Not ideal way of passing in. 
	ekf_.R_ = R_radar_;
	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } 
  
  // Laser updates
  else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
