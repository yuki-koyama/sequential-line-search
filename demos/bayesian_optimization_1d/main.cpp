#include "core.hpp"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char* argv[])
{
    Core core;

    constexpr unsigned n_trials     = 5;
    constexpr unsigned n_iterations = 15;

    for (unsigned trial = 0; trial < n_trials; ++trial)
    {
        std::cout << "========================" << std::endl;
        std::cout << "Trial " << trial + 1 << std::endl;
        std::cout << "========================" << std::endl;

        // Iterate optimization steps
        for (unsigned i = 0; i < n_iterations; ++i)
        {
            std::cout << "---- Iteration " << i + 1 << " ----" << std::endl;
            core.proceedOptimization();
        }

        std::cout << std::endl << "Found maximizer: " << core.x_max.transpose() << std::endl;
        std::cout << "Found maximum: " << core.y_max << std::endl << std::endl;

        // Reset the optimization
        core.X     = MatrixXd::Zero(0, 0);
        core.y     = VectorXd::Zero(0);
        core.x_max = VectorXd::Zero(0);
        core.y_max = NAN;
        core.computeRegression();
    }

    return 0;
}
