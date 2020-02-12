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

    /// \brief A wrapper struct for an nlopt-style objective function
    struct RegressorWrapper
    {
        const Regressor*          regressor;
        const AcquisitionFuncType func_type;
        const double              gaussian_process_upper_confidence_bound_hyperparam;
    };

    /// \brief A wrapper struct for an nlopt-style objective function
    struct RegressorPairWrapper
    {
        const Regressor*          orig_regressor;
        const Regressor*          updated_regressor;
        const AcquisitionFuncType func_type;
        const double              gaussian_process_upper_confidence_bound_hyperparam;
    };

    /// \brief NLopt-style objective function definition for finding the next (single) point.
    double objective(const std::vector<double>& x, std::vector<double>& grad, void* data)
    {
        const Regressor*           regressor = static_cast<RegressorWrapper*>(data)->regressor;
        const AcquisitionFuncType& func_type = static_cast<RegressorWrapper*>(data)->func_type;
        const double&              hyperparam =
            static_cast<RegressorWrapper*>(data)->gaussian_process_upper_confidence_bound_hyperparam;

        const auto eigen_x = Eigen::Map<const VectorXd>(&x[0], x.size());

        if (!grad.empty())
        {
            const VectorXd derivative =
                acquisition_func::CalcAcquisitionValueDerivative(*regressor, eigen_x, func_type, hyperparam);
            std::memcpy(grad.data(), derivative.data(), sizeof(double) * derivative.size());
        }

        return acquisition_func::CalcAcqusitionValue(*regressor, eigen_x, func_type, hyperparam);
    }

    /// \brief NLopt-style objective function definition for finding the next multiple points.
    ///
    /// \details Ref: Schonlau et al, Global Versus Local Search in Constrained Optimization of Computer Models, 1997.
    double objective_for_multiple_points(const std::vector<double>& x, std::vector<double>& grad, void* data)
    {
        const Regressor*           orig_regressor    = static_cast<RegressorPairWrapper*>(data)->orig_regressor;
        const Regressor*           updated_regressor = static_cast<RegressorPairWrapper*>(data)->updated_regressor;
        const AcquisitionFuncType& func_type         = static_cast<RegressorWrapper*>(data)->func_type;
        const double&              hyperparam =
            static_cast<RegressorWrapper*>(data)->gaussian_process_upper_confidence_bound_hyperparam;

        const auto mu               = [&](const VectorXd& x) { return orig_regressor->PredictMu(x); };
        const auto sigma            = [&](const VectorXd& x) { return updated_regressor->PredictSigma(x); };
        const auto mu_derivative    = [&](const VectorXd& x) { return orig_regressor->PredictMuDerivative(x); };
        const auto sigma_derivative = [&](const VectorXd& x) { return updated_regressor->PredictSigmaDerivative(x); };

        const auto eigen_x = Eigen::Map<const VectorXd>(&x[0], x.size());

        switch (func_type)
        {
            case AcquisitionFuncType::ExpectedImprovement:
            {
                const VectorXd x_best = orig_regressor->PredictMaximumPointFromData();

                if (!grad.empty())
                {
                    const VectorXd derivative = mathtoolbox::GetExpectedImprovementDerivative(
                        eigen_x, mu, sigma, x_best, mu_derivative, sigma_derivative);

                    std::memcpy(grad.data(), derivative.data(), sizeof(double) * derivative.size());
                }

                return mathtoolbox::GetExpectedImprovement(eigen_x, mu, sigma, x_best);
            }
            case AcquisitionFuncType::GaussianProcessUpperConfidenceBound:
            {
                if (!grad.empty())
                {
                    const VectorXd derivative = mathtoolbox::GetGaussianProcessUpperConfidenceBoundDerivative(
                        eigen_x, mu, sigma, hyperparam, mu_derivative, sigma_derivative);

                    std::memcpy(grad.data(), derivative.data(), sizeof(double) * derivative.size());
                }

                return mathtoolbox::GetGaussianProcessUpperConfidenceBound(eigen_x, mu, sigma, hyperparam);
            }
        }
    }

    VectorXd FindGlobalSolution(nlopt::vfunc   objective,
                                void*          data,
                                const unsigned num_dim,
                                const unsigned num_global_search_iters,
                                const unsigned num_local_search_iters)
    {
        const VectorXd upper = VectorXd::Constant(num_dim, 1.0);
        const VectorXd lower = VectorXd::Constant(num_dim, 0.0);

#ifdef SEQUENTIAL_LINE_SEARCH_USE_PARALLELIZED_MULTI_START_SEARCH
        MatrixXd x_stars(num_dim, num_global_search_iters);
        VectorXd y_stars(num_global_search_iters);

        const auto perform_local_optimization_from_random_initialization = [&](const int i) {
            const VectorXd x_ini = 0.5 * (VectorXd::Random(num_dim) + VectorXd::Ones(num_dim));
            const VectorXd x_star =
                nloptutil::solve(x_ini, upper, lower, objective, nlopt::LD_LBFGS, data, true, num_local_search_iters);
            const double y_star = [&]() {
                vector<double> x_star_std(num_dim);
                vector<double> grad_std;
                std::memcpy(x_star_std.data(), x_star.data(), sizeof(double) * num_dim);

                return objective(x_star_std, grad_std, data);
            }();

            x_stars.col(i) = x_star;
            y_stars(i)     = y_star;
        };

        parallelutil::queue_based_parallel_for(num_global_search_iters,
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
            nloptutil::solve(x_ini, upper, lower, objective, nlopt::GN_DIRECT, data, true, num_global_search_iters);

        // Refine the solution by a quasi-Newton method
        const VectorXd x_local =
            nloptutil::solve(x_global, upper, lower, objective, nlopt::LD_LBFGS, data, true, num_local_search_iters);

        return x_local;
#endif
    }
} // namespace

double sequential_line_search::acquisition_func::CalcAcqusitionValue(
    const Regressor&          regressor,
    const VectorXd&           x,
    const AcquisitionFuncType func_type,
    const double              gaussian_process_upper_confidence_bound_hyperparam)
{
    if (regressor.GetSmallY().rows() == 0)
    {
        return 0.0;
    }

    const auto mu    = [&](const VectorXd& x) { return regressor.PredictMu(x); };
    const auto sigma = [&](const VectorXd& x) { return regressor.PredictSigma(x); };

    switch (func_type)
    {
        case AcquisitionFuncType::ExpectedImprovement:
        {
            const VectorXd x_best = regressor.PredictMaximumPointFromData();

            return mathtoolbox::GetExpectedImprovement(x, mu, sigma, x_best);
        }
        case AcquisitionFuncType::GaussianProcessUpperConfidenceBound:
        {
            return mathtoolbox::GetGaussianProcessUpperConfidenceBound(
                x, mu, sigma, gaussian_process_upper_confidence_bound_hyperparam);
        }
    }
}

VectorXd sequential_line_search::acquisition_func::CalcAcquisitionValueDerivative(
    const Regressor&          regressor,
    const VectorXd&           x,
    const AcquisitionFuncType func_type,
    const double              gaussian_process_upper_confidence_bound_hyperparam)
{
    if (regressor.GetSmallY().rows() == 0)
    {
        return VectorXd::Zero(x.size());
    }

    const auto mu               = [&](const VectorXd& x) { return regressor.PredictMu(x); };
    const auto sigma            = [&](const VectorXd& x) { return regressor.PredictSigma(x); };
    const auto mu_derivative    = [&](const VectorXd& x) { return regressor.PredictMuDerivative(x); };
    const auto sigma_derivative = [&](const VectorXd& x) { return regressor.PredictSigmaDerivative(x); };

    switch (func_type)
    {
        case AcquisitionFuncType::ExpectedImprovement:
        {
            const VectorXd x_best = regressor.PredictMaximumPointFromData();

            return mathtoolbox::GetExpectedImprovementDerivative(x, mu, sigma, x_best, mu_derivative, sigma_derivative);
        }
        case AcquisitionFuncType::GaussianProcessUpperConfidenceBound:
        {
            return mathtoolbox::GetGaussianProcessUpperConfidenceBoundDerivative(
                x, mu, sigma, gaussian_process_upper_confidence_bound_hyperparam, mu_derivative, sigma_derivative);
        }
    }
}

VectorXd
sequential_line_search::acquisition_func::FindNextPoint(const Regressor&          regressor,
                                                        const unsigned            num_global_search_iters,
                                                        const unsigned            num_local_search_iters,
                                                        const AcquisitionFuncType func_type,
                                                        const double gaussian_process_upper_confidence_bound_hyperparam)
{
    const unsigned num_dim = regressor.GetNumDims();

    RegressorWrapper data{&regressor, func_type, gaussian_process_upper_confidence_bound_hyperparam};

    return FindGlobalSolution(objective, &data, num_dim, num_global_search_iters, num_local_search_iters);
}

vector<VectorXd> sequential_line_search::acquisition_func::FindNextPoints(
    const Regressor&          regressor,
    const unsigned            num_points,
    const unsigned            num_global_search_iters,
    const unsigned            num_local_search_iters,
    const AcquisitionFuncType func_type,
    const double              gaussian_process_upper_confidence_bound_hyperparam)
{
    const unsigned num_dim = regressor.GetNumDims();

    vector<VectorXd> points;

    const VectorXd kernel_hyperparams = regressor.GetKernelHyperparams();

    GaussianProcessRegressor temp_regressor(
        regressor.GetLargeX(), regressor.GetSmallY(), kernel_hyperparams, regressor.GetNoiseHyperparam());

    for (unsigned i = 0; i < num_points; ++i)
    {
        // Create a data object for the nlopt-style objective function
        RegressorPairWrapper data{
            &regressor, &temp_regressor, func_type, gaussian_process_upper_confidence_bound_hyperparam};

        // Find a global solution
        const VectorXd x_star = FindGlobalSolution(
            objective_for_multiple_points, &data, num_dim, num_global_search_iters, num_local_search_iters);

        // Register the found solution
        points.push_back(x_star);

        // If this is not the final iteration, prepare data for the next iteration
        if (points.size() != num_points)
        {
            const unsigned N = temp_regressor.GetLargeX().cols();

            MatrixXd new_X(num_dim, N + 1);
            new_X.block(0, 0, num_dim, N) = temp_regressor.GetLargeX();
            new_X.col(N)                  = x_star;

            VectorXd new_y(temp_regressor.GetSmallY().rows() + 1);
            new_y << temp_regressor.GetSmallY(), temp_regressor.PredictMu(x_star);

            temp_regressor = GaussianProcessRegressor(new_X, new_y, kernel_hyperparams, regressor.GetNoiseHyperparam());
        }
    }

    return points;
}
