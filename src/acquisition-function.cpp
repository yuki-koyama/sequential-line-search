#include <cmath>
#include <cstring>
#include <iostream>
#include <mathtoolbox/acquisition-functions.hpp>
#include <nlopt-util.hpp>
#include <parallel-util.hpp>
#include <sequential-line-search/acquisition-function.h>
#include <sequential-line-search/gaussian-process-regressor.h>
#include <utility>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::pair;
using std::vector;

namespace
{
    using namespace sequential_line_search;

    double objective(const std::vector<double>& x, std::vector<double>& grad, void* data)
    {
        const Regressor* regressor = static_cast<const Regressor*>(data);

        if (!grad.empty())
        {
            const Eigen::VectorXd derivative = acquisition_function::CalculateAcquisitionValueDerivative(
                *regressor, Eigen::Map<const VectorXd>(&x[0], x.size()));
            std::memcpy(grad.data(), derivative.data(), sizeof(double) * derivative.size());
        }

        return acquisition_function::CalculateAcqusitionValue(*regressor, Eigen::Map<const VectorXd>(&x[0], x.size()));
    }

    // Schonlau et al, Global Versus Local Search in Constrained Optimization of Computer Models, 1997.
    double objective_for_multiple_points(const std::vector<double>& x, std::vector<double>& grad, void* data)
    {
        const Regressor* orig_regressor =
            static_cast<pair<const Regressor*, const GaussianProcessRegressor*>*>(data)->first;
        const GaussianProcessRegressor* updated_regressor =
            static_cast<pair<const Regressor*, const GaussianProcessRegressor*>*>(data)->second;

        const VectorXd eigen_x = Eigen::Map<const VectorXd>(&x[0], x.size());

        const Eigen::VectorXd x_best = [&]() {
            const int num_data_points = orig_regressor->getX().cols();

            Eigen::VectorXd f(num_data_points);
            for (int i = 0; i < num_data_points; ++i)
            {
                f(i) = orig_regressor->PredictMu(orig_regressor->getX().col(i));
            }

            int best_index;
            f.maxCoeff(&best_index);

            return orig_regressor->getX().col(best_index);
        }();

        const auto mu    = [&](const Eigen::VectorXd& x) { return orig_regressor->PredictMu(x); };
        const auto sigma = [&](const Eigen::VectorXd& x) { return updated_regressor->PredictSigma(x); };

        if (!grad.empty())
        {
            const auto mu_derivative    = [&](const VectorXd& x) { return orig_regressor->PredictMuDerivative(x); };
            const auto sigma_derivative = [&](const VectorXd& x) {
                return updated_regressor->PredictSigmaDerivative(x);
            };

            const Eigen::VectorXd derivative = mathtoolbox::GetExpectedImprovementDerivative(
                eigen_x, mu, sigma, x_best, mu_derivative, sigma_derivative);

            std::memcpy(grad.data(), derivative.data(), sizeof(double) * derivative.size());
        }

        return mathtoolbox::GetExpectedImprovement(eigen_x, mu, sigma, x_best);
    }
} // namespace

namespace sequential_line_search
{
    namespace acquisition_function
    {
        double
        CalculateAcqusitionValue(const Regressor& regressor, const Eigen::VectorXd& x, const FunctionType function_type)
        {
            assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

            if (regressor.gety().rows() == 0)
            {
                return 0.0;
            }

            const Eigen::VectorXd x_best = [&]() {
                const int num_data_points = regressor.getX().cols();

                Eigen::VectorXd f(num_data_points);
                for (int i = 0; i < num_data_points; ++i)
                {
                    f(i) = regressor.PredictMu(regressor.getX().col(i));
                }

                int best_index;
                f.maxCoeff(&best_index);

                return regressor.getX().col(best_index);
            }();

            const auto mu    = [&](const Eigen::VectorXd& x) { return regressor.PredictMu(x); };
            const auto sigma = [&](const Eigen::VectorXd& x) { return regressor.PredictSigma(x); };

            return mathtoolbox::GetExpectedImprovement(x, mu, sigma, x_best);
        }

        VectorXd CalculateAcquisitionValueDerivative(const Regressor&   regressor,
                                                     const VectorXd&    x,
                                                     const FunctionType function_type)
        {
            assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

            if (regressor.gety().rows() == 0)
            {
                return VectorXd::Zero(x.size());
            }

            const VectorXd x_best = [&]() {
                const int num_data_points = regressor.getX().cols();

                VectorXd f(num_data_points);
                for (int i = 0; i < num_data_points; ++i)
                {
                    f(i) = regressor.PredictMu(regressor.getX().col(i));
                }

                int best_index;
                f.maxCoeff(&best_index);

                return regressor.getX().col(best_index);
            }();

            const auto mu               = [&](const VectorXd& x) { return regressor.PredictMu(x); };
            const auto sigma            = [&](const VectorXd& x) { return regressor.PredictSigma(x); };
            const auto mu_derivative    = [&](const VectorXd& x) { return regressor.PredictMuDerivative(x); };
            const auto sigma_derivative = [&](const VectorXd& x) { return regressor.PredictSigmaDerivative(x); };

            return mathtoolbox::GetExpectedImprovementDerivative(x, mu, sigma, x_best, mu_derivative, sigma_derivative);
        }

        Eigen::VectorXd FindNextPoint(Regressor& regressor, const FunctionType function_type)
        {
            assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

            const unsigned D = regressor.getX().rows();

            const VectorXd upper = VectorXd::Constant(D, 1.0);
            const VectorXd lower = VectorXd::Constant(D, 0.0);

            constexpr int num_trials = 20;

            Eigen::MatrixXd x_stars(D, num_trials);
            Eigen::VectorXd y_stars(num_trials);
            const auto      perform_local_optimization_from_random_initialization = [&](const int i) {
                const VectorXd x_ini = 0.5 * (VectorXd::Random(D) + VectorXd::Ones(D));
                const VectorXd x_star =
                    nloptutil::solve(x_ini, upper, lower, objective, nlopt::LD_LBFGS, &regressor, true, 50);

                const double y_star = CalculateAcqusitionValue(regressor, x_star);

                x_stars.col(i) = x_star;
                y_stars(i)     = y_star;
            };

            parallelutil::queue_based_parallel_for(num_trials, perform_local_optimization_from_random_initialization);

            const int best_index = [&]() {
                int index;
                y_stars.maxCoeff(&index);
                return index;
            }();

            return x_stars.col(best_index);
        }

        vector<VectorXd> FindNextPoints(const Regressor& regressor, const unsigned n, const FunctionType function_type)
        {
            assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

            const unsigned D = regressor.getX().rows();

            const VectorXd upper = VectorXd::Constant(D, 1.0);
            const VectorXd lower = VectorXd::Constant(D, 0.0);

            vector<VectorXd> points;

            GaussianProcessRegressor temporary_regressor(
                regressor.getX(), regressor.gety(), regressor.geta(), regressor.getb(), regressor.getr());

            const Eigen::VectorXd x_best = [&]() {
                const int num_data_points = regressor.getX().cols();

                Eigen::VectorXd f(num_data_points);
                for (int i = 0; i < num_data_points; ++i)
                {
                    f(i) = regressor.PredictMu(regressor.getX().col(i));
                }

                int best_index;
                f.maxCoeff(&best_index);

                return regressor.getX().col(best_index);
            }();

            for (unsigned i = 0; i < n; ++i)
            {
                pair<const Regressor*, const GaussianProcessRegressor*> data(&regressor, &temporary_regressor);

                const VectorXd x_star = [&]() {
                    constexpr int num_trials = 20;

                    Eigen::MatrixXd x_stars(D, num_trials);
                    Eigen::VectorXd y_stars(num_trials);
                    const auto      perform_local_optimization_from_random_initialization = [&](const int i) {
                        const VectorXd x_ini  = 0.5 * (VectorXd::Random(D) + VectorXd::Ones(D));
                        const VectorXd x_star = nloptutil::solve(
                            x_ini, upper, lower, objective_for_multiple_points, nlopt::LD_LBFGS, &data, true, 100);

                        const double y_star = [&]() {
                            const auto mu    = [&](const Eigen::VectorXd& x) { return regressor.PredictMu(x); };
                            const auto sigma = [&](const Eigen::VectorXd& x) {
                                return temporary_regressor.PredictSigma(x);
                            };

                            return mathtoolbox::GetExpectedImprovement(x_star, mu, sigma, x_best);
                        }();

                        x_stars.col(i) = x_star;
                        y_stars(i)     = y_star;
                    };

                    parallelutil::queue_based_parallel_for(num_trials,
                                                           perform_local_optimization_from_random_initialization);

                    const int best_index = [&]() {
                        int index;
                        y_stars.maxCoeff(&index);
                        return index;
                    }();

                    return x_stars.col(best_index);
                }();

                points.push_back(x_star);

                if (i + 1 == n)
                {
                    break;
                }

                const unsigned N = temporary_regressor.getX().cols();

                MatrixXd new_X(D, N + 1);
                new_X.block(0, 0, D, N) = temporary_regressor.getX();
                new_X.col(N)            = x_star;

                VectorXd new_y(temporary_regressor.gety().rows() + 1);
                new_y << temporary_regressor.gety(), temporary_regressor.PredictMu(x_star);

                temporary_regressor =
                    GaussianProcessRegressor(new_X, new_y, regressor.geta(), regressor.getb(), regressor.getr());
            }

            return points;
        }
    } // namespace acquisition_function
} // namespace sequential_line_search
