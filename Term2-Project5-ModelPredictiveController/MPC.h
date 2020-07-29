#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

#define TIMESLICE_DURATION 20
#define DT 0.1
#define LF 2.67
#define DEG25RAD 0.436332 // 25 deg in rad, used as delta bound


// Desired cte, epsi and speed
// Both the reference cross track and orientation errors are 0.
// The reference velocity is set to 40 mph
#define DES_CTE 0
#define DES_EPSI 0
#define DES_V 50

// Set weights parameters for the cost function
//5,5 best so far.
#define WEIGHT_CTE 10 //was 2 and worked OK, but way too off.
#define WEIGHT_EPSI 5 //was 20 and worked OK. we are turning too quick
#define WEIGHT_V 1
#define WEIGHT_DELTA 100000
#define WEIGHT_A 20

// Set lower and upper limits for variables.
#define MAX_THROTTLE 1.0 // Maximum 'a' value
#define BOUND 1.0e3 // Bound value for other variables

using namespace std;
class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
  vector<double> forecast_x;
  vector<double> forecast_y;
};

#endif /* MPC_H */
