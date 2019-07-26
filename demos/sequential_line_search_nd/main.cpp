#include <iostream>
#include <sequential-line-search/sequential-line-search.h>
#include <sequential-line-search/utils.h>

namespace
{
    constexpr double a         = 0.500;
    constexpr double r         = 0.500;
    constexpr double b         = 0.001;
    constexpr double variance  = 0.100;
    constexpr double btl_scale = 0.010;

    constexpr bool use_slider_enlargement = true;
    constexpr bool use_MAP                = true;

    constexpr unsigned test_dimension = 8;

    // Define a test function
    double evaluateObjectiveFunction(const Eigen::VectorXd& x)
    {
        assert(x.rows() == test_dimension);

        auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma) {
            return std::exp(-(x - mu).squaredNorm() / (sigma * sigma));
        };

        return lambda(x, Eigen::VectorXd::Constant(test_dimension, 0.5), 1.0);
    }

    // Define an analytic solution of the test function
    const Eigen::VectorXd analytic_solution = Eigen::VectorXd::Constant(test_dimension, 0.5);
} // namespace

int main(int argc, char* argv[])
{
    constexpr unsigned n_trials     = 3;
    constexpr unsigned n_iterations = 10;

    // Storage for performance reports
    Eigen::MatrixXd objective_values(n_iterations, n_trials);
    Eigen::MatrixXd residual_norms(n_iterations, n_trials);

    for (unsigned trial = 0; trial < n_trials; ++trial)
    {
        sequential_line_search::SequentialLineSearchOptimizer optimizer(
            test_dimension, use_slider_enlargement, use_MAP);

        optimizer.setHyperparameters(a, r, b, variance, btl_scale);

        std::cout << "========================" << std::endl;
        std::cout << "Trial " << trial + 1 << std::endl;
        std::cout << "========================" << std::endl;

        // Iterate optimization steps
        for (unsigned i = 0; i < n_iterations; ++i)
        {
            std::cout << "---- Iteration " << i + 1 << " ----" << std::endl;

            // search the best position
            double max_slider_position = 0.0;
            double max_y               = -1e+10;
            for (double slider_position = 0.0; slider_position <= 1.0; slider_position += 0.0001)
            {
                const double y = evaluateObjectiveFunction(optimizer.getParameters(slider_position));
                if (y > max_y)
                {
                    max_y               = y;
                    max_slider_position = slider_position;
                }
            }

            std::cout << "x: " << optimizer.getParameters(max_slider_position).transpose() << std::endl;
            std::cout << "y: " << max_y << std::endl;

            objective_values(i, trial) = max_y;
            residual_norms(i, trial)   = (optimizer.getParameters(max_slider_position) - analytic_solution).norm();

            optimizer.submit(max_slider_position);
        }

        std::cout << std::endl << "Found maximizer: " << optimizer.getMaximizer().transpose() << std::endl;
        std::cout << "Found maximum: " << evaluateObjectiveFunction(optimizer.getMaximizer()) << std::endl << std::endl;
    }

    // Export a report as a CSV file
    sequential_line_search::utils::exportMatrixToCsv("objective_values.csv", objective_values);
    sequential_line_search::utils::exportMatrixToCsv("residual_norms.csv", residual_norms);

    return 0;
}
