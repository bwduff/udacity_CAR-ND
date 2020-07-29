#include "kalman_filter.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	//Taken from Lesson 5-13.
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	//Taken from Lesson 5-7.
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//New State estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  * DIRECTIVES:
  * Update the state by using Extended Kalman Filter equations
  */
    VectorXd h(3);
    double px = x_[0];
    double py = x_[1];
    double p2 = px*px + py*py;
    if(fabs(p2) < 0.0001) {
        h << 0.0, 0.0, 0.0;  //If it is close enough to 0, set it such.
    }
    else { //Otherwise, OK to use atan. Atan is what causes issues with small values.
        double vx = x_[2];
        double vy = x_[3];
        double p = sqrt(p2);
        double h1 = p;
        double h2 = atan2(py, px);
        double h3 = (px*vx + py*vy)/p;
        h << h1, h2, h3;
    }

	VectorXd y = z - h;
	
	//SCALING ANGLE
	//Needed to prevent predictions from getting way off track during radar usage.
	#define PI (3.14159265F)
    if (y[1] < -PI)
	  y[1] += 2*PI;
    if (y[1] > PI)
	  y[1] -= 2*PI;
	 
	//This part is identical to Update() method but reproducing to avoid header file and key class changes.
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//New state estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_; 
}
