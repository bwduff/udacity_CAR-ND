/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *
 *  Modified on: Aug 22, 2017
 *           By: Brent Wylie
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle_filter.h"

using namespace std;

#define NUMBER_OF_PARTICLES 100
#define EPS 0.0000001 //Epsilon, used to avoid div by 0
#define INF 1/EPS //Psuedo infinity, used in comparisons for min distance.

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Number of priticles
  num_particles = NUMBER_OF_PARTICLES;
  
  // Creating normal distributions for initial coordinates
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  // Initialize the weights and particles
  particles.resize(num_particles);
  weights.resize(num_particles);
  for (int i = 0; i < num_particles; i++){
    double current_x = dist_x(gen);
    double current_y = dist_y(gen);
    double current_theta = dist_theta(gen);
    double current_weight = 1.;
    
	Particle current_part { i, current_x, current_y,  current_theta, current_weight};
    particles[i] = current_part;
    weights[i] = current_weight;
  }
  // Mark filter as initialized
  is_initialized = true;
}

 void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Implemented: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine rgen;
	for(Particle& current_part : particles){
		//Calculate mean for x, y and theta coords
		double x_mean = current_part.x + (
			abs(yaw_rate) < EPS ?  //Use correct formula depending if yaw rate is essentially 0 to avoid dividing by 0. If yaw rate less than eps, use first formula, otherwise use next.
			velocity * delta_t * cos(current_part.theta) :
			velocity / yaw_rate * (sin(current_part.theta + yaw_rate * delta_t) - sin(current_part.theta))
			);
		double y_mean = current_part.y + (
			abs(yaw_rate) < EPS ?  //Use correct formula depending if yaw rate is essentially 0 to avoid dividing by 0. If yaw rate less than eps, use first formula, otherwise use next.
			velocity * delta_t * sin(current_part.theta) :
			velocity / yaw_rate * (cos(current_part.theta) - cos(current_part.theta + yaw_rate * delta_t))
			);
		double theta_mean = current_part.theta + yaw_rate * delta_t;

		//Create normal distributions for new coords
		std::normal_distribution<double> dist_x(x_mean,std_pos[0]);
		std::normal_distribution<double> dist_y(y_mean,std_pos[1]);
		std::normal_distribution<double> dist_theta(theta_mean,std_pos[2]);
		//Update particle with new predictions
		current_part.x = dist_x(rgen);
		current_part.y = dist_y(rgen);
		current_part.theta = dist_theta(rgen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Implemented: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//Iterate through landmark observations
	for(LandmarkObs& current_obs : observations){
		double min_dist = INFINITY; //Initialize as very large
		int min_id = 0; //Initialize nearest id as 0
		for(const LandmarkObs& current_pred: predicted){
			//Calculate distance (error) between current prediction and current observation
			double current_dist = dist(current_pred.x, current_pred.y, current_obs.x,current_obs.y);
			if(current_dist < min_dist){ min_dist = current_dist; min_id = current_pred.id;}
		}
		//Assign closest prediction to nearest landmark id
		current_obs.id = min_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Implemented: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//For each partical
	for(int p_idx=0; p_idx<particles.size(); p_idx++){
		//current particle
		Particle& current_part = particles[p_idx];
		//Convert obsv into map coordinates
		std::vector<LandmarkObs> transformed_obs;
		for(const LandmarkObs& current_obs : observations){
			LandmarkObs t_landmark;
			t_landmark.id = current_obs.id;
			t_landmark.x = current_obs.x * cos(current_part.theta) - current_obs.y * sin(current_part.theta) + current_part.x;
			t_landmark.y = current_obs.x * sin(current_part.theta) + current_obs.y * cos(current_part.theta) + current_part.y;
			transformed_obs.push_back(t_landmark);
		}
		//Convert landmarks within sensor range to predicted landmarks
		std::vector<LandmarkObs> predicted;
		for(const Map::single_landmark_s& landmark : map_landmarks.landmark_list){
			if(dist(landmark.x_f, landmark.y_f, current_part.x, current_part.y)<= sensor_range){
				LandmarkObs landpred {landmark.id_i, landmark.x_f, landmark.y_f};
				predicted.push_back(landpred);
			}
		}
		//Associate every obs and predicted location
		dataAssociation(predicted, transformed_obs);
		//Go through each observation and update weights
		double running_weight_prod = 1.0;
		for(const LandmarkObs& current_obs : transformed_obs){
			Map::single_landmark_s current_pred = map_landmarks.landmark_list[current_obs.id - 1 ]; //Assuming this index will never become negative
			double dx = current_obs.x - current_pred.x_f;
			double dy = current_obs.y - current_pred.y_f;
			
			double newWeight = 1 / (M_PI * 2 * std_landmark[0] * std_landmark[1]) *
				std::exp(-1 * (pow(dx, 2) / pow(std_landmark[0], 2) + pow(dy, 2) / pow(std_landmark[1], 2)));
			running_weight_prod *= newWeight;
		}

		//Update new weight & add to list
		current_part.weight = running_weight_prod;
		weights[p_idx] = running_weight_prod;
	}
}

void ParticleFilter::resample() {
	// Implemented: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	//Construct random generator and distribution
	std::default_random_engine rgen;
	std::discrete_distribution<int> disc_dist { weights.begin(),weights.end()};
	
	//Draw new particles at random
	std::vector<Particle> new_parts;
	for(int i=0; i < num_particles; i++){
		//Gen particle index at random
		int current_part_i = disc_dist(rgen);
		//Select particle from list
		Particle current_part = particles[current_part_i];
		//Push back onto new list
		new_parts.push_back(current_part);
	}
	//Assign new list
	particles = new_parts;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates
	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();
	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;
 	return particle;
}

string ParticleFilter::getAssociations(Particle best){
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best){
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best){
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
