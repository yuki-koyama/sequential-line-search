#include <iostream>
#include <sequential-line-search/sequential-line-search.h>
#include "core.h"

using namespace sequential_line_search;
using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace
{
    constexpr double   a          = 0.500;
    constexpr double   r          = 0.500;
    constexpr double   noise      = 0.001;
    constexpr double   btl_scale  = 0.010;
    constexpr double   variance   = 0.100;
    constexpr bool     use_MAP    = true;
}

int main(int argc, char *argv[])
{
    Core core;
    
    core.use_MAP_hyperparameters = use_MAP;
    
    PreferenceRegressor::Params::getInstance().a         = a;
    PreferenceRegressor::Params::getInstance().r         = r;
    PreferenceRegressor::Params::getInstance().variance  = variance;
    PreferenceRegressor::Params::getInstance().b         = noise;
    PreferenceRegressor::Params::getInstance().btl_scale = btl_scale;

    core.computeRegression();
    core.updateSliderEnds();
    
    constexpr unsigned n_trials = 3;
    constexpr unsigned n_iterations = 10;

    for (unsigned trial = 0; trial < n_trials; ++ trial)
    {
        std::cout << "========================" << std::endl;
        std::cout << "Trial " << trial + 1 << std::endl;
        std::cout << "========================" << std::endl;
        
        // Iterate optimization steps
        for (unsigned i = 0; i < n_iterations; ++ i)
        {
            std::cout << "---- Iteration " << i + 1 << " ----" << std::endl;
            
            // search the best position
            double max_slider_position = 0.0;
            double max_y               = - 1e+10;
            for (double slider_position = 0.0; slider_position <= 1.0; slider_position += 0.0001)
            {
                const double y = core.evaluateObjectiveFunction(core.computeParametersFromSlider(slider_position));
                if (y > max_y)
                {
                    max_y = y;
                    max_slider_position = slider_position;
                }
            }
            
            std::cout << "x: " << core.computeParametersFromSlider(max_slider_position).transpose() << std::endl;
            std::cout << "y: " << max_y << std::endl;

            core.proceedOptimization(max_slider_position);
        }
        
        std::cout << std::endl << "Found maximizer: " << core.x_max.transpose() << std::endl;
        std::cout << "Found maximum: " << core.evaluateObjectiveFunction(core.x_max) << std::endl << std::endl;
        
        // Reset the optimization
        core.clear();
        core.computeRegression();
        core.updateSliderEnds();
    }
    
    return 0;
}
