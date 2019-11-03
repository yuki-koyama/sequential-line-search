#include <iostream>
#include <sequential-line-search/sequential-line-search.hpp>
#include <sequential-line-search/utils.hpp>
#ifdef PARALLEL
#include <parallel-util.hpp>
#endif
#include <timer.hpp>

namespace
{
    constexpr double a         = 0.500;
    constexpr double r         = 0.500;
    constexpr double b         = 0.001;
    constexpr double variance  = 0.100;
    constexpr double btl_scale = 0.010;

    constexpr unsigned n_trials     = 3;
    constexpr unsigned n_iterations = 10;

    constexpr bool use_slider_enlargement = true;
    constexpr bool use_MAP_hyperparams    = true;

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
    // Storage for performance reports
    Eigen::MatrixXd objective_values(n_iterations, n_trials);
    Eigen::MatrixXd residual_norms(n_iterations, n_trials);
    Eigen::MatrixXd elapsed_times(n_iterations, n_trials);

    const auto perform_test = [&](const int trial_index) {
        sequential_line_search::SequentialLineSearchOptimizer optimizer(
            test_dimension, use_slider_enlargement, use_MAP_hyperparams);

        optimizer.SetHyperparameters(a, r, b, variance, btl_scale);

        std::cout << "========================" << std::endl;
        std::cout << "Trial " << trial_index + 1 << std::endl;
        std::cout << "========================" << std::endl;

        // Iterate optimization steps
        for (unsigned i = 0; i < n_iterations; ++i)
        {
            std::cout << "---- Iteration " << i + 1 << " ----" << std::endl;

            constexpr double search_epsilon = 1e-05;

            // Search the best position in the current slider space
            double max_slider_position = 0.0;
            double max_y               = -1e+10;
            for (double slider_position = 0.0; slider_position <= 1.0; slider_position += search_epsilon)
            {
                const double y = evaluateObjectiveFunction(optimizer.GetParameters(slider_position));
                if (y > max_y)
                {
                    max_y               = y;
                    max_slider_position = slider_position;
                }
            }

            const Eigen::VectorXd max_x = optimizer.GetParameters(max_slider_position);

            std::cout << "x: " << max_x.transpose().format(Eigen::IOFormat(3)) << std::endl;
            std::cout << "y: " << max_y << std::endl;

            objective_values(i, trial_index) = max_y;
            residual_norms(i, trial_index)   = (max_x - analytic_solution).norm();

            timer::Timer t;

            // Perform Bayesian optimization
            optimizer.SubmitLineSearchResult(max_slider_position);

            elapsed_times(i, trial_index) = t.get_elapsed_time_in_milliseconds();
        }

        const Eigen::VectorXd x_star = optimizer.GetMaximizer();
        const double          y_star = evaluateObjectiveFunction(x_star);

        std::cout << std::endl;
        std::cout << "Found maximizer: " << x_star.transpose().format(Eigen::IOFormat(3)) << std::endl;
        std::cout << "Found maximum: " << y_star << std::endl << std::endl;
    };

#ifdef PARALLEL
    parallelutil::queue_based_parallel_for(n_trials, perform_test);
#else
    for (unsigned trial_index = 0; trial_index < n_trials; ++trial_index)
    {
        perform_test(trial_index);
    }
#endif

    // Export a report as a CSV file
    sequential_line_search::utils::ExportMatrixToCsv("objective_values.csv", objective_values);
    sequential_line_search::utils::ExportMatrixToCsv("residual_norms.csv", residual_norms);
    sequential_line_search::utils::ExportMatrixToCsv("elapsed_times.csv", elapsed_times);

    return 0;
}
