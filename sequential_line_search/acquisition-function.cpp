#include "acquisition-function.h"
#include <cmath>
#include <iostream>
#include <utility>
#include <nlopt-util.hpp>
#include "regressor.h"
#include "gaussianprocessregressor.h"

using std::vector;
using std::pair;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace ExpectedImprovement
{
    double objective(const std::vector<double> &x, std::vector<double>& /*grad*/, void* data)
    {
        const Regressor* regressor = static_cast<const Regressor*>(data);
        return calculateExpectedImprovedment(*regressor, Eigen::Map<const VectorXd>(&x[0], x.size()));
    }
    
    double calculateExpectedImprovedment(const Regressor &regressor, const VectorXd& x)
    {
        if (regressor.gety().rows() == 0) { return 0.0; }
        
        const double y_best = regressor.gety().maxCoeff();
        const double s_x    = regressor.estimate_s(x);
        const double u      = (regressor.estimate_y(x) - y_best) / s_x;
        const double Phi    = 0.5 * std::erf(u / std::sqrt(2.0)) + 0.5;
        const double phi    = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(- u * u / 2.0);
        const double EI     = (regressor.estimate_y(x) - y_best) * Phi + s_x * phi;
        
        return (s_x < 1e-10 || std::isnan(EI)) ? 0.0 : EI;
    }
    
    VectorXd findNextPoint(Regressor& regressor)
    {
        const unsigned D = regressor.getX().rows();
        
        const VectorXd upper = VectorXd::Constant(D, 1.0);
        const VectorXd lower = VectorXd::Constant(D, 0.0);
        const VectorXd x_ini = VectorXd::Constant(D, 0.5);
        
        const VectorXd x_star_global = nloptutil::solve(x_ini, upper, lower, objective, nlopt::GN_DIRECT, &regressor, 800);
        const VectorXd x_star_local  = nloptutil::solve(x_star_global, upper, lower, objective, nlopt::LN_COBYLA, &regressor, 200);
        
        return x_star_local;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Schonlau et al, Global Versus Local Search in Constrained Optimization of Computer Models, 1997.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    double obj(const std::vector<double> &x, std::vector<double>& /*grad*/, void* data)
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
    
    vector<VectorXd> findNextPoints(const Regressor &regressor, unsigned n)
    {
        const unsigned D = regressor.getX().rows();
        
        const VectorXd upper = VectorXd::Constant(D, 1.0);
        const VectorXd lower = VectorXd::Constant(D, 0.0);
        const VectorXd x_ini = VectorXd::Constant(D, 0.5);
        
        vector<VectorXd> points;
        
        GaussianProcessRegressor reg(regressor.getX(), regressor.gety(), regressor.geta(), regressor.getb(), regressor.getr());
        
        for (unsigned i = 0; i < n; ++ i)
        {
            pair<const Regressor*, const GaussianProcessRegressor*> data(&regressor, &reg);
            const VectorXd x_star_global = nloptutil::solve(x_ini, upper, lower, obj, nlopt::GN_DIRECT, &data, 800);
            const VectorXd x_star_local  = nloptutil::solve(x_star_global, upper, lower, obj, nlopt::LN_COBYLA, &data, 200);
            
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
