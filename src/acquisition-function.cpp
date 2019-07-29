#include <cmath>
#include <iostream>
#include <nlopt-util.hpp>
#include <sequential-line-search/acquisition-function.h>
#include <sequential-line-search/gaussian-process-regressor.h>
#include <utility>
#include <mathtoolbox/acquisition-functions.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::pair;
using std::vector;

namespace
{
    using namespace sequential_line_search;

    double objective(const std::vector<double>& x, std::vector<double>& /*grad*/, void* data)
    {
        const Regressor* regressor = static_cast<const Regressor*>(data);
        return acquisition_function::CalculateAcqusitionValue(*regressor, Eigen::Map<const VectorXd>(&x[0], x.size()));
    }

    // Schonlau et al, Global Versus Local Search in Constrained Optimization of Computer Models, 1997.
    double objective_for_multiple_points(const std::vector<double>& x, std::vector<double>& /*grad*/, void* data)
    {
        const Regressor* origRegressor =
            static_cast<pair<const Regressor*, const GaussianProcessRegressor*>*>(data)->first;
        const GaussianProcessRegressor* updatedRegressor =
            static_cast<pair<const Regressor*, const GaussianProcessRegressor*>*>(data)->second;

        const VectorXd eigen_x = Eigen::Map<const VectorXd>(&x[0], x.size());

        const Eigen::VectorXd x_best = [&]()
        {
            int best_index;
            origRegressor->gety().maxCoeff(&best_index);
            return origRegressor->getX().col(best_index);
        }();

        const auto mu    = [&](const Eigen::VectorXd& x) { return origRegressor->estimate_y(x); };
        const auto sigma = [&](const Eigen::VectorXd& x) { return updatedRegressor->estimate_s(x); };

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

            const Eigen::VectorXd x_best = [&]()
            {
                int best_index;
                regressor.gety().maxCoeff(&best_index);
                return regressor.getX().col(best_index);
            }();

            const auto mu    = [&](const Eigen::VectorXd& x) { return regressor.estimate_y(x); };
            const auto sigma = [&](const Eigen::VectorXd& x) { return regressor.estimate_s(x); };

            return mathtoolbox::GetExpectedImprovement(x, mu, sigma, x_best);
        }

        Eigen::VectorXd FindNextPoint(Regressor& regressor, const FunctionType function_type)
        {
            assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

            const unsigned D = regressor.getX().rows();

            const VectorXd upper = VectorXd::Constant(D, 1.0);
            const VectorXd lower = VectorXd::Constant(D, 0.0);
            const VectorXd x_ini = VectorXd::Constant(D, 0.5);

            const VectorXd x_star_global =
                nloptutil::solve(x_ini, upper, lower, objective, nlopt::GN_DIRECT, &regressor, 800);
            const VectorXd x_star_local =
                nloptutil::solve(x_star_global, upper, lower, objective, nlopt::LN_COBYLA, &regressor, 200);

            return x_star_local;
        }

        vector<VectorXd> FindNextPoints(const Regressor& regressor, const unsigned n, const FunctionType function_type)
        {
            assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");

            const unsigned D = regressor.getX().rows();

            const VectorXd upper = VectorXd::Constant(D, 1.0);
            const VectorXd lower = VectorXd::Constant(D, 0.0);
            const VectorXd x_ini = VectorXd::Constant(D, 0.5);

            vector<VectorXd> points;

            GaussianProcessRegressor reg(
                regressor.getX(), regressor.gety(), regressor.geta(), regressor.getb(), regressor.getr());

            for (unsigned i = 0; i < n; ++i)
            {
                pair<const Regressor*, const GaussianProcessRegressor*> data(&regressor, &reg);
                const VectorXd                                          x_star_global =
                    nloptutil::solve(x_ini, upper, lower, objective_for_multiple_points, nlopt::GN_DIRECT, &data, 800);
                const VectorXd x_star_local = nloptutil::solve(
                    x_star_global, upper, lower, objective_for_multiple_points, nlopt::LN_COBYLA, &data, 200);

                points.push_back(x_star_local);

                if (i + 1 == n)
                {
                    break;
                }

                const unsigned N = reg.getX().cols();

                MatrixXd newX(D, N + 1);
                newX.block(0, 0, D, N) = reg.getX();
                newX.col(N)            = x_star_local;

                VectorXd newY(reg.gety().rows() + 1);
                newY << reg.gety(), reg.estimate_y(x_star_local);

                reg = GaussianProcessRegressor(newX, newY, regressor.geta(), regressor.getb(), regressor.getr());
            }

            return points;
        }
    } // namespace acquisition_function
} // namespace sequential_line_search
