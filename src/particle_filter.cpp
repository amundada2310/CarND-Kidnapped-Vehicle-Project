/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

    //1. check if the particle filter is already initialized
    //if yes we do not have to initialize again
    if (is_initialized)
    {
        return;
    }

    // 2.set the number of particles for our filter
    num_particles = 100;  // TODO: Set the number of particles

    // 3. initialize our random engine generator for adding gaussian random noise to each of the particle
    std::default_random_engine gen;

    // 4. Std deviation x,y and theta values
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    // 5. normal ditribution with mean 0 and std deviation defined above
    std::normal_distribution<double> dist_x(0, std_x);
    std::normal_distribution<double> dist_y(0, std_y);
    std::normal_distribution<double> dist_theta(0, std_theta);

    // 6. Initialize, position, weight = 1, and add noise to each particle num_particles
    for (auto i = 0; i < num_particles; i++)
    {
        Particle p;
        p.id = i;
        p.x = x;
        p.y = y;
        p.theta = theta;
        p.weight = 1.0;

        // adding noise to each particle
        p.x = p.x + dist_x(gen);
        p.y = p.y + dist_y(gen);
        p.theta = p.theta + dist_theta(gen);

        //add to list of particles
        particles.push_back(p);
    }

    // 7. set is_initialized to true ; i.e. initialize the filter now
    is_initialized = true;
    std::cout << "Initialization done" << std::endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) 
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

   // 1. initialize our random engine generator for adding gaussian random noise to each of the particle
    std::default_random_engine gen;

    // 2. Std deviation x,y and theta values
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    // 3. normal ditribution with mean 0 and std deviation defined above
    std::normal_distribution<double> dist_x(0, std_x);
    std::normal_distribution<double> dist_y(0, std_y);
    std::normal_distribution<double> dist_theta(0, std_theta);

    //4. Addthe control values (measurements) to each of the particle and predict the state for next time step
    for (auto i = 0; i < num_particles; i++)
    {
        // 4.1. if the change in value of yawrate is negligible then use the below formula predicting next state
        if (fabs(yaw_rate) < 0.00001)
        {
            particles[i].x = (particles[i].x) + (velocity * delta_t * cos(particles[i].theta));
            particles[i].y = (particles[i].y) + (velocity * delta_t * sin(particles[i].theta));
        }
        //4.2 if that is not the case then use below formula for predicting next state
        else 
        {
            particles[i].x = (particles[i].x) + ((velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)));
            particles[i].y = (particles[i].y) + ((velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)));
            particles[i].theta = (particles[i].theta) + (yaw_rate * delta_t);
        }

        //4.3 adding noise to each new predicted state of particle 
        //New predicted state of each particle are given as below
        particles[i].x = particles[i].x + dist_x(gen);
        particles[i].y = particles[i].y + dist_y(gen);
        particles[i].theta = particles[i].theta + dist_theta(gen);

    }

    std::cout << "New states of each particle prediction done" << std::endl;

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) 
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

    //1. Loop over all the observations and for each observation determine the associated predicted landmark pair
    for (auto i = 0; i < observations.size(); i++)
    {

        // 1.1 init minimum distance to maximum possible value (before starting the distance calculations
        double min_dist = numeric_limits<double>::max();

        //1.2 initialize the id to store the id of the minimun distance
        int p_id = -1;

        //1.3 loop over all predicted landmarks
        for (auto j = 0; j < predicted.size(); j++)
        {
            //1.3.1 calculate the distance using the helper function (dist)
            double cur_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

            //1.3.2 check if that distance is less then the min_dist initialized earlier
            // if yes input that value as the min_dist and store the id 
            if (cur_distance < min_dist)
            {
                min_dist = cur_distance;
                p_id = predicted[j].id;
            }
        }
        // 1.4 finally after going thropugh all the predicted landmarks get the id of the data associate (min distance)
        observations[i].id = p_id;

    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) 
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

    //step 1. We need to create a prediction landmark list first for each particle whcih are within the sensor range
    //step 2 Once done we need to transform the car observed landmarks from car coordinates to map coordinates using homogemous tranformation for each particle
    //step 3. do data association
    //step 4 assign weights to each particle

    for (auto i = 0; i < num_particles; i++)
    {
        // 1.1 get the x,y and theta for each particle
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        // 1.2 create a landmark vector to store predicted landmarks list for each particle
        vector<LandmarkObs> predict;

        //1.3 loop through list of all the landmarks in the map
        for(auto j =0; j < map_landmarks.landmark_list.size(); j++)
        {
            // 1.3.1 get the x,y coordinates for each landmark in map and its id
            float landmark_map_x = map_landmarks.landmark_list[j].x_f;
            float landmark_map_y = map_landmarks.landmark_list[j].y_f;
            int landmark_map_id = map_landmarks.landmark_list[j].id_i;

            // 1.3.2 determine which landmarks from the map are within the sensor range of the particle

            //this can be done by 2 methods 1 using the dist function in help.h or by using the rectangular method
            // I am defining both but using the dist function method

            //rectangular method
            /*if (fabs(landmark_map_x - p_x) <= sensor_range && fabs(landmark_map_y - p_y) <= sensor_range)
            {
                // add to the predict vector
                predict.push_back(LandmarkObs{ landmark_map_id, landmark_map_x, landmark_map_y });
            }*/

            //determining distance and check which ones are within the sensor_range for the particle
            double cur_distance = dist(p_x, p_y, landmark_map_x, landmark_map_y);
            if (cur_distance < sensor_range)
            {
                //add to the predicted values to the created predict vector
                predict.push_back(LandmarkObs{ landmark_map_id, landmark_map_x, landmark_map_y });
            }


        }

        //1.3.3 Doing teh homogenous tarnformation for each of the car observation from car coordinates into map coordination

        // 1.3.4 create a landmark vector to store transformed landmarks list for each particle
        vector<LandmarkObs> transform;

        //1.3.4.1 loop through all obseravations
        for (auto t = 0; t < observations.size(); t++)
        {
            //get the x transform
            double transform_x = p_x + (cos(p_theta) * observations[t].x) - (sin(p_theta) * observations[t].y);
            //get the y transform
            double transform_y = p_y + (sin(p_theta) * observations[t].x) + (cos(p_theta) * observations[t].y);
            //get the id
            int transform_id = observations[t].id;
            //add to the tranform values to the created transforms vector
            transform.push_back(LandmarkObs{ transform_id, transform_x, transform_y });

        }

        // 1.3.5 perform the data association or the predictions and transformed observations on current particle
        dataAssociation(predict, transform);

        // 1.3.6 Now we need determine weights for each particle
        //step 1 :for that we need to get each pair -  transformed observed landmark coordinates and its associated predicted (nearby) landmark coordinates
        //step 2 :multivariant gaussian formula to use for determining the weight for each pair
        //step 3 : finally multiple weights of each pair to get the final weight of particle (update step)

        //1.3.6.1 reinitiallizing the weights to 1
        particles[i].weight = 1;

        //1.3.6.2 loop over all the tranformed observations for each particle
        for (auto m = 0; m < transform.size(); m++)
        {

            // initialize placeholders for the transformed obervations and its associated prediction
            double o_x, o_y, pr_x, pr_y;

            //get the coordinates for the transformed landmark
            o_x = transform[m].x;
            o_y = transform[m].y;

            //get the associated landmark id (obtained in data Association function)
            int associated_predict = transform[m].id;

            //loop through all the predict landmarks
            for (auto k = 0; k < predict.size(); k++)
            {
                // get the matched id associated predict landmark
                if (predict[k].id == associated_predict)
                {
                    // get the x, y coordinates of the predict associated with the current observation
                    pr_x = predict[k].x;
                    pr_y = predict[k].y;
                }
            }

            //1.3.6.3 calculate the weight for the combination using multivariant gaussian ditribution method
            // calculate normalization term
            double gauss_norm;
            double sig_x = std_landmark[0];
            double sig_y = std_landmark[1];
            gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

            // calculate exponent
            double exponent;
            exponent = (pow(o_x - pr_x, 2) / (2 * pow(sig_x, 2))) + (pow(o_y - pr_y, 2) / (2 * pow(sig_y, 2)));

            // calculate weight using normalization terms and exponent
            double weight_o;
            weight_o = gauss_norm * exp(-exponent);

            // product of this obersvation weight with total observations weight
            particles[i].weight = particles[i].weight * weight_o;

        }


    }

}

void ParticleFilter::resample() 
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    //1. Get weights of all the particles:
    //initialize weights vector to store weights

    vector<double> weights;
    for (auto i = 0; i < num_particles; i++)
    {
        weights.push_back(particles[i].weight);
    }

    // 2. get max weight from the list
    //initialize weight with som minimum weight value
    double max_weight = numeric_limits<double>::min();

    //loop through all the weights
    for (auto j = 0; j < weights.size(); j++)
    {
        if (weights[j] > max_weight)
        {
            max_weight = weights[j];
        }
    }


    //4. generate uniform distribution for index 
   //initialize our random engine generator for adding gaussian random noise to each of the particle
    std::default_random_engine gen;

    uniform_int_distribution<int> dist_indx(0, num_particles - 1);
    //randomly picked a index from the distribution
    int index = dist_indx(gen);

    //5. generate uniform distribution for weights depending on the impotance weight (large or small)
    uniform_real_distribution<double> dist_weight(0.0, max_weight);

    //6. initialize better
    double beta = 0.0;

    //7. Create new vector for storing list of new particles
    vector<Particle> new_particles;

    //7. Loop over num_particles times - going into resampling wheel
    for (auto t = 0; t < num_particles; t++)
    {
        beta = beta + (2*dist_weight(gen));//////////////do we need to multiply by 2 here or not??????????
        while (beta > weights[index])
        {
            beta = beta - weights[index];
            index = (index + 1) % num_particles;
        }

        new_particles.push_back(particles[index]);
    }

    // update with the new list
    particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}