#include <cmath>
#include <cstring>
#include <iostream>
#include <mathtoolbox/acquisition-functions.hpp>
#include <nlopt-util.hpp>
#include <parallel-util.hpp>
#include <sequential-line-search/acquisition-function.hpp>
#include <sequential-line-search/gaussian-process-regressor.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace
{
    using namespace sequential_line_search;

    // A wrapper struct for an nlopt-style objective function
    struct RegressorWrapper
    {
        const Regressor* regressor;
    };

    // A wrapper struct for an nlopt-style objective function
    struct RegressorPairWrapper
    {
        const Regressor* orig_regressor;
        const Regressor* updated_regressor;
    };

    double objective(const std::vector<double>& x, std::vector<double>& grad, void* data)
    {
        const Regressor* regressor = static_cast<RegressorWrapper*>(data)->regressor;

        if (!grad.empty())
        {
            const VectorXd derivative = acquisition_function::CalculateAcquisitionValueDerivative(
                *regressor, Eigen::Map<const VectorXd>(&x[0], x.size()));
            std::memcpy(grad.data(), derivative.data(), sizeof(double) * derivative.size());
        }

        return acquisition_function::CalculateAcqusitionValue(*regressor, Eigen::Map<const VectorXd>(&x[0], x.size()));
    }

    // Schonlau et al, Global Versus Local Search in Constrained Optimization of Computer Models, 1997.
    double objective_for_multiple_points(const std::vector<double>& x, std::vector<double>& grad, void* data)
    {
        const Regressor* orig_regressor    = static_cast<RegressorPairWrapper*>(data)->orig_regressor;
        const Regressor* updated_regressor = static_cast<RegressorPairWrapper*>(data)->updated_regressor;

        const VectorXd eigen_x = Eigen::Map<const VectorXd>(&x[0], x.size());

        const VectorXd x_best = orig_regressor->PredictMaximumPointFromData();

        const auto mu    = [&](const VectorXd& x) { return orig_regressor->PredictMu(x); };
        const auto sigma = [&](const VectorXd& x) { return updated_regressor->PredictSigma(x); };

        if (!grad.empty())
        {
            const auto mu_derivative    = [&](const VectorXd& x) { return orig_regressor->PredictMuDerivative(x); };
            const auto sigma_derivative = [&](const VectorXd& x) {
                return updated_regressor->PredictSigmaDerivative(x);
            };

            const VectorXd derivative = mathtoolbox::GetExpectedImprovementDerivative(
                eigen_x, mu, sigma, x_best, mu_derivative, sigma_derivative);

            std::memcpy(grad.data(), derivative.data(), sizeof(double) * derivative.size());
        }

        return mathtoolbox::GetExpectedImprovement(eigen_x, mu, sigma, x_best);
    }

    VectorXd FindGlobalSolution(nlopt::vfunc       objective,
                                void*              data,
                                const unsigned int num_dim,
                                const unsigned int num_global_trials)
    {
        const VectorXd upper = VectorXd::Constant(num_dim, 1.0);
        const VectorXd lower = VectorXd::Constant(num_dim, 0.0);

        constexpr unsigned max_num_local_search_iters = 50;

#ifdef SEQUENTIAL_LINE_SEARCH_USE_PARALLELIZED_MULTI_START_SEARCH
        MatrixXd x_stars(num_dim, num_global_trials);
        VectorXd y_stars(num_global_trials);

        const auto perform_local_optimization_from_random_initialization = [&](const int i) {
            const VectorXd x_ini  = 0.5 * (VectorXd::Random(num_dim) + VectorXd::Ones(num_dim));
            const VectorXd x_star = nloptutil::solve(
                x_ini, upper, lower, objective, nlopt::LD_LBFGS, data, true, max_num_local_search_iters);
            const double y_star = [&]() {
                std::vector<double> x_star_std(num_dim);
                std::vector<double> grad_std;
                std::memcpy(x_star_std.data(), x_star.data(), sizeof(double) * num_dim);

                return objective(x_star_std, grad_std, data);
            }();

            x_stars.col(i) = x_star;
            y_stars(i)     = y_star;
        };

        parallelutil::queue_based_parallel_for(num_global_trials,
                                               perform_local_optimization_from_random_initialization);

        const int best_index = [&]() {
            int index;
            y_stars.maxCoeff(&index);
            return index;
        }();

        return x_stars.col(best_index);
#else
        const VectorXd x_ini = 0.5 * (VectorXd::Random(num_dim) + VectorXd::Ones(num_dim));

        // Find a global solution by the DIRECT method
        const VectorXd x_global =
            nloptutil::solve(x_ini, upper, lower, objective, nlopt::GN_DIRECT, data, true, num_global_trials);

        // Refine the solution by a quasi-Newton method
        const VectorXd x_local = nloptutil::solve(
            x_global, upper, lower, objective, nlopt::LD_LBFGS, data, true, max_num_local_search_iters);

        return x_local;
#endif
    }
} // namespace

double sequential_line_search::acquisition_function::CalculateAcqusitionValue(const Regressor&   regressor,
                                                                              const VectorXd&    x,
                                                                              const FunctionType function_type)
{
    assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

    if (regressor.gety().rows() == 0)
    {
        return 0.0;
    }

    const VectorXd x_best = regressor.PredictMaximumPointFromData();

    const auto mu    = [&](const VectorXd& x) { return regressor.PredictMu(x); };
    const auto sigma = [&](const VectorXd& x) { return regressor.PredictSigma(x); };

    return mathtoolbox::GetExpectedImprovement(x, mu, sigma, x_best);
}

VectorXd
sequential_line_search::acquisition_function::CalculateAcquisitionValueDerivative(const Regressor&   regressor,
                                                                                  const VectorXd&    x,
                                                                                  const FunctionType function_type)
{
    assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

    if (regressor.gety().rows() == 0)
    {
        return VectorXd::Zero(x.size());
    }

    const VectorXd x_best = regressor.PredictMaximumPointFromData();

    const auto mu               = [&](const VectorXd& x) { return regressor.PredictMu(x); };
    const auto sigma            = [&](const VectorXd& x) { return regressor.PredictSigma(x); };
    const auto mu_derivative    = [&](const VectorXd& x) { return regressor.PredictMuDerivative(x); };
    const auto sigma_derivative = [&](const VectorXd& x) { return regressor.PredictSigmaDerivative(x); };

    return mathtoolbox::GetExpectedImprovementDerivative(x, mu, sigma, x_best, mu_derivative, sigma_derivative);
}

VectorXd sequential_line_search::acquisition_function::FindNextPoint(const Regressor&   regressor,
                                                                     const unsigned     num_trials,
                                                                     const FunctionType function_type)
{
    assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

    const unsigned num_dim = regressor.GetNumDims();

    RegressorWrapper data = {&regressor};

    return FindGlobalSolution(objective, &data, num_dim, num_trials);
}

vector<VectorXd> sequential_line_search::acquisition_function::FindNextPoints(const Regressor&   regressor,
                                                                              const unsigned     num_points,
                                                                              const unsigned     num_trials,
                                                                              const FunctionType function_type)
{
    assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

    const unsigned num_dim = regressor.GetNumDims();

    vector<VectorXd> points;

    GaussianProcessRegressor temporary_regressor(
        regressor.getX(), regressor.gety(), regressor.geta(), regressor.getb(), regressor.getr());

    for (unsigned i = 0; i < num_points; ++i)
    {
        // Create a data object for the nlopt-style objective function
        RegressorPairWrapper data{&regressor, &temporary_regressor};

        // Find a global solution
        const VectorXd x_star = FindGlobalSolution(objective_for_multiple_points, &data, num_dim, num_trials);

        // Register the found solution
        points.push_back(x_star);

        // If this is not the final iteration, prepare data for the next iteration
        if (points.size() != num_points)
        {
            const unsigned N = temporary_regressor.getX().cols();

            MatrixXd new_X(num_dim, N + 1);
            new_X.block(0, 0, num_dim, N) = temporary_regressor.getX();
            new_X.col(N)                  = x_star;

            VectorXd new_y(temporary_regressor.gety().rows() + 1);
            new_y << temporary_regressor.gety(), temporary_regressor.PredictMu(x_star);

            temporary_regressor =
                GaussianProcessRegressor(new_X, new_y, regressor.geta(), regressor.getb(), regressor.getr());
        }
    }

    return points;
}
