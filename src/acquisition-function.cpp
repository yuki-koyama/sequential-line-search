#include <sequential-line-search/acquisition-function.h>
#include <sequential-line-search/gaussianprocessregressor.h>
#include <cmath>
#include <iostream>
#include <utility>
#include <nlopt-util.hpp>

using std::vector;
using std::pair;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
    double objective(const std::vector<double> &x, std::vector<double>& /*grad*/, void* data)
    {
        const Regressor* regressor = static_cast<const Regressor*>(data);
        return sequential_line_search::acquisition_function::CalculateAcqusitionValue(*regressor, Eigen::Map<const VectorXd>(&x[0], x.size()));
    }
    
    // Schonlau et al, Global Versus Local Search in Constrained Optimization of Computer Models, 1997.
    double objective_for_multiple_points(const std::vector<double> &x, std::vector<double>& /*grad*/, void* data)
    {
        const Regressor*                origRegressor    = static_cast<pair<const Regressor*, const GaussianProcessRegressor*>*>(data)->first;
        const GaussianProcessRegressor* updatedRegressor = static_cast<pair<const Regressor*, const GaussianProcessRegressor*>*>(data)->second;
        
        const VectorXd _x = Eigen::Map<const VectorXd>(&x[0], x.size());
        
        const double y_best = origRegressor->gety().maxCoeff();
        const double s_x    = updatedRegressor->estimate_s(_x);
        const double u      = (origRegressor->estimate_y(_x) - y_best) / s_x;
        const double Phi    = 0.5 * std::erf(u / std::sqrt(2.0)) + 0.5;
        const double phi    = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(- u * u / 2.0);
        const double EI     = s_x * (u * Phi + phi);
        
        return (s_x < 1e-08 || std::isnan(EI)) ? 0.0 : EI;
    }
}

namespace sequential_line_search
{
    namespace acquisition_function
    {
        double CalculateAcqusitionValue(const Regressor&       regressor,
                                        const Eigen::VectorXd& x,
                                        const FunctionType     function_type)
        {
            assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");
            
            if (regressor.gety().rows() == 0) { return 0.0; }
            
            const double y_best = regressor.gety().maxCoeff();
            const double s_x    = regressor.estimate_s(x);
            const double u      = (regressor.estimate_y(x) - y_best) / s_x;
            const double Phi    = 0.5 * std::erf(u / std::sqrt(2.0)) + 0.5;
            const double phi    = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(- u * u / 2.0);
            const double EI     = (regressor.estimate_y(x) - y_best) * Phi + s_x * phi;
            
            return (s_x < 1e-10 || std::isnan(EI)) ? 0.0 : EI;
        }
        
        Eigen::VectorXd FindNextPoint(Regressor& regressor,
                                      const FunctionType function_type)
        {
            assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");
            
            const unsigned D = regressor.getX().rows();
            
            const VectorXd upper = VectorXd::Constant(D, 1.0);
            const VectorXd lower = VectorXd::Constant(D, 0.0);
            const VectorXd x_ini = VectorXd::Constant(D, 0.5);
            
            const VectorXd x_star_global = nloptutil::solve(x_ini, upper, lower, objective, nlopt::GN_DIRECT, &regressor, 800);
            const VectorXd x_star_local  = nloptutil::solve(x_star_global, upper, lower, objective, nlopt::LN_COBYLA, &regressor, 200);
            
            return x_star_local;
        }
        
        vector<VectorXd> FindNextPoints(const Regressor&   regressor,
                                        const unsigned     n,
                                        const FunctionType function_type)
        {
            assert(function_type == FunctionType::ExpectedImprovement && "FunctionType not supported yet.");
            
            const unsigned D = regressor.getX().rows();
            
            const VectorXd upper = VectorXd::Constant(D, 1.0);
            const VectorXd lower = VectorXd::Constant(D, 0.0);
            const VectorXd x_ini = VectorXd::Constant(D, 0.5);
            
            vector<VectorXd> points;
            
            GaussianProcessRegressor reg(regressor.getX(), regressor.gety(), regressor.geta(), regressor.getb(), regressor.getr());
            
            for (unsigned i = 0; i < n; ++ i)
            {
                pair<const Regressor*, const GaussianProcessRegressor*> data(&regressor, &reg);
                const VectorXd x_star_global = nloptutil::solve(x_ini, upper, lower, objective_for_multiple_points, nlopt::GN_DIRECT, &data, 800);
                const VectorXd x_star_local  = nloptutil::solve(x_star_global, upper, lower, objective_for_multiple_points, nlopt::LN_COBYLA, &data, 200);
                
                points.push_back(x_star_local);
                
                if (i + 1 == n) { break; }
                
                const unsigned N = reg.getX().cols();
                
                MatrixXd newX(D, N + 1);
                newX.block(0, 0, D, N) = reg.getX();
                newX.col(N) = x_star_local;
                
                VectorXd newY(reg.gety().rows() + 1);
                newY << reg.gety(), reg.estimate_y(x_star_local);
                
                reg = GaussianProcessRegressor(newX, newY, regressor.geta(), regressor.getb(), regressor.getr());
            }
            
            return points;
        }
    }
}
