#include "preferenceregressor.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/LU>
#include "utility.h"
#include "nloptutility.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

//#define VERBOSE
//#define NOISELESS

#ifdef VERBOSE
#include <timer.hpp>
#endif

namespace
{

#ifdef NOISELESS
const double b_fixed = 1e-06;
#endif

inline double calc_grad_a(const VectorXd& y, const MatrixXd& C_inv, const MatrixXd& X, double a, double b, const VectorXd& r)
{
    const double   a_prior  = PreferenceRegressor::Params::getInstance().a;
    const double   variance = PreferenceRegressor::Params::getInstance().variance;

    const MatrixXd C_grad_a             = Regressor::calc_C_grad_a(X, a, b, r);
    const double   log_p_f_theta_grad_a = 0.5 * y.transpose() * C_inv * C_grad_a * C_inv * y - 0.5 * (C_inv * C_grad_a).trace();
    const double   log_prior            = (std::log(a_prior) - variance - std::log(a)) / (variance * a);
    return log_p_f_theta_grad_a + log_prior;
}

#ifndef NOISELESS
inline double calc_grad_b(const VectorXd& y, const MatrixXd& C_inv, const MatrixXd& X, double a, double b, const VectorXd& r)
{
    const double   b_prior  = PreferenceRegressor::Params::getInstance().b;
    const double   variance = PreferenceRegressor::Params::getInstance().variance;

    const MatrixXd C_grad_b             = Regressor::calc_C_grad_b(X, a, b, r);
    const double   log_p_f_theta_grad_b = 0.5 * y.transpose() * C_inv * C_grad_b * C_inv * y - 0.5 * (C_inv * C_grad_b).trace();
    const double   log_prior            = (std::log(b_prior) - variance - std::log(b)) / (variance * b);
    return log_p_f_theta_grad_b + log_prior;
}
#endif

inline VectorXd calc_grad_r(const VectorXd& y, const MatrixXd& C_inv, const MatrixXd& X, double a, double b, const VectorXd& r)
{
    const double   r_prior  = PreferenceRegressor::Params::getInstance().r;
    const double   variance = PreferenceRegressor::Params::getInstance().variance;

    VectorXd grad = VectorXd::Zero(r.rows());
    for (unsigned i = 0; i < r.rows(); ++ i)
    {
        const MatrixXd C_grad_r               = Regressor::calc_C_grad_r_i(X, a, b, r, i);
        const double   log_p_f_theta_grad_r_i = 0.5 * y.transpose() * C_inv * C_grad_r * C_inv * y - 0.5 * (C_inv * C_grad_r).trace();
        grad(i) += log_p_f_theta_grad_r_i;
    }
    for (unsigned i = 0; i < r.rows(); ++ i)
    {
        const double log_prior = (std::log(r_prior) - variance - std::log(r(i))) / (variance * r(i));
        grad(i) += log_prior;
    }

    return grad;
}

// log p(d_k | f)
inline double calc_log_likelihood(const Preference& p, const double w, const VectorXd& y)
{
    const double btl_scale = w * PreferenceRegressor::Params::getInstance().btl_scale;

    VectorXd tmp(p.size()); for (unsigned i = 0; i < p.size(); ++ i) tmp(i) = y(p[i]);
    return std::log(Utility::BTL(tmp, btl_scale));
}

// Log likelihood that will be maximized
double objective(const std::vector<double> &x, std::vector<double>& grad, void* data)
{
    const PreferenceRegressor* regressor = static_cast<PreferenceRegressor*>(data);

    const MatrixXd&                X = regressor->X;
    const std::vector<Preference>& D = regressor->D;
    const VectorXd&                w = regressor->w;
    const unsigned                 M = X.cols();
    const VectorXd                 y = Eigen::Map<const VectorXd>(&x[0], M);

    const double   a = (regressor->use_MAP_hyperparameters) ? x[M + 0] : PreferenceRegressor::Params::getInstance().a;
#ifdef NOISELESS
    const double   b = b_fixed;
#else
    const double   b = (regressor->use_MAP_hyperparameters) ? x[M + 1] : PreferenceRegressor::Params::getInstance().b;
#endif
    const VectorXd r = (regressor->use_MAP_hyperparameters) ? VectorXd(Eigen::Map<const VectorXd>(&x[M + 2], X.rows())) : VectorXd::Constant(X.rows(), PreferenceRegressor::Params::getInstance().r);

    double obj = 0.0;

    // Log likelihood of data
    for (unsigned i = 0; i < D.size(); ++ i)
    {
        obj += calc_log_likelihood(D[i], w(i), y);
    }

    // Log likelihood of y distribution
    const MatrixXd C     = Regressor::calc_C(X, a, b, r);
    const MatrixXd C_inv = C.inverse();
    const double   C_det = C.determinant();
    const double   term1 = - 0.5 * y.transpose() * C_inv * y;
    const double   term2 = - 0.5 * std::log(C_det);
    const double   term3 = - 0.5 * M * std::log(2.0 * M_PI);
    obj += term1 + term2 + term3;

    if (regressor->use_MAP_hyperparameters)
    {
        // Priors for GP parameters
        const double   a_prior  = PreferenceRegressor::Params::getInstance().a;
#ifndef NOISELESS
        const double   b_prior  = PreferenceRegressor::Params::getInstance().b;
#endif
        const double   r_prior  = PreferenceRegressor::Params::getInstance().r;
        const double   variance = PreferenceRegressor::Params::getInstance().variance;

        obj += std::log(Utility::log_normal(a, std::log(a_prior), variance));
#ifndef NOISELESS
        obj += std::log(Utility::log_normal(b, std::log(b_prior), variance));
#endif
        for (unsigned i = 0; i < r.rows(); ++ i)
        {
            obj += std::log(Utility::log_normal(r(i), std::log(r_prior), variance));
        }
    }

    // When the algorithm is gradient-based, compute the gradient vector
    if (grad.size() == x.size())
    {
        VectorXd grad_y = VectorXd::Zero(y.rows());

        // Accumulate per-data derivatives
        const double btl_scale = PreferenceRegressor::Params::getInstance().btl_scale;
        for (unsigned i = 0; i < D.size(); ++ i)
        {
            const Preference& p = D[i];
            const double      s = btl_scale * w(i);
            VectorXd tmp1(p.size()); for (unsigned i = 0; i < p.size(); ++ i) tmp1(i) = y(p[i]);
            const VectorXd tmp2 = Utility::derivative_BTL(tmp1, s) / Utility::BTL(tmp1, s);
            for (unsigned i = 0; i < p.size(); ++ i) grad_y(p[i]) += tmp2(i);
        }

        // Add GP term
        grad_y += - C_inv * y;

        Eigen::Map<VectorXd>(&grad[0], grad_y.rows()) = grad_y;

        if (regressor->use_MAP_hyperparameters)
        {
            grad[M + 0] = calc_grad_a(y, C_inv, X, a, b, r);
#ifdef NOISELESS
            grad[M + 1] = 0.0;
#else
            grad[M + 1] = calc_grad_b(y, C_inv, X, a, b, r);
#endif
            VectorXd grad_r = calc_grad_r(y, C_inv, X, a, b, r);
            for (unsigned i = 0; i < grad_r.rows(); ++ i)
            {
                grad[M + 2 + i] = grad_r(i);
            }
        }
        else
        {
            grad[M + 0] = 0.0;
            grad[M + 1] = 0.0;
            grad[M + 2] = 0.0;
        }
    }

    return obj;
}

}

PreferenceRegressor::PreferenceRegressor(const MatrixXd &X, const std::vector<Preference>& D, bool use_MAP_hyperparameters) :
    use_MAP_hyperparameters(use_MAP_hyperparameters),
    X(X),
    D(D)
{
    if (X.cols() == 0 || D.size() == 0) return;

    w = Eigen::VectorXd::Ones(D.size());

    compute_MAP();

    C     = calc_C(X, a, b, r);
    C_inv = C.inverse();
}

PreferenceRegressor::PreferenceRegressor(const MatrixXd &X, const std::vector<Preference>& D, const Eigen::VectorXd &w, bool use_MAP_hyperparameters) :
    use_MAP_hyperparameters(use_MAP_hyperparameters),
    X(X),
    D(D),
    w(w)
{
    if (X.cols() == 0 || D.size() == 0) return;

    compute_MAP();

    C     = calc_C(X, a, b, r);
    C_inv = C.inverse();
}

PreferenceRegressor::PreferenceRegressor(const MatrixXd &X, const std::vector<Preference>& D, const Eigen::VectorXd &w, bool use_MAP_hyperparameters, const PreferenceRegressor* previous) :
    use_MAP_hyperparameters(use_MAP_hyperparameters),
    X(X),
    D(D),
    w(w)
{
    if (X.cols() == 0 || D.size() == 0) return;

    compute_MAP(previous);

    C     = calc_C(X, a, b, r);
    C_inv = C.inverse();
}

double PreferenceRegressor::estimate_y(const VectorXd &x) const
{
    const VectorXd k = Regressor::calc_k(x, X, a, b, r);
    return k.transpose() * C_inv * y;
}

double PreferenceRegressor::estimate_s(const VectorXd &x) const
{
    const VectorXd k = Regressor::calc_k(x, X, a, b, r);
    return std::sqrt(a + b - k.transpose() * C_inv * k);
}

void PreferenceRegressor::compute_MAP(const PreferenceRegressor *previous)
{
    const unsigned M = X.cols();
    const unsigned d = X.rows();

    VectorXd upper = VectorXd::Constant(M + 2 + d, + 1e+01);
    VectorXd lower = VectorXd::Constant(M + 2 + d, - 1e+01); lower.block(M, 0, 2 + d, 1) = VectorXd::Constant(2 + d, 1e-05);
    VectorXd x_ini = VectorXd::Constant(M + 2 + d, 0.0);
    x_ini(M + 0) = Params::getInstance().a;
    x_ini(M + 1) = Params::getInstance().b;
    x_ini.block(M + 2, 0, d, 1) = VectorXd::Constant(d, Params::getInstance().r);

    // Use the MAP estimated values in previous regression as initial values
    if (previous != nullptr)
    {
        for (unsigned i = 0; i < M; ++ i)
        {
            x_ini(i) = previous->estimate_y(X.col(i));
        }
        x_ini(M + 0) = previous->a;
        x_ini(M + 1) = previous->b;
        x_ini.block(M + 2, 0, d, 1) = previous->r;
    }

#ifdef VERBOSE
    timer::Timer t("PreferenceRegressor::compute_MAP");
#endif

    const VectorXd x_opt = nloptUtility::compute(x_ini, upper, lower, objective, this, nlopt::LD_TNEWTON, 500);

    y = x_opt.block(0, 0, M, 1);

    if (use_MAP_hyperparameters)
    {
        a = x_opt(M + 0);
#ifdef NOISELESS
        b = b_fixed;
#else
        b = x_opt(M + 1);
#endif
        r = x_opt.block(M + 2, 0, d, 1);

#ifdef VERBOSE
        std::cout << "Learned hyperparameters ... a: " << a << ", \tb: " << b << ", \tr: " << r.transpose() << std::endl;
#endif
    }
    else
    {
        a = PreferenceRegressor::Params::getInstance().a;
        b = PreferenceRegressor::Params::getInstance().b;
        r = VectorXd::Constant(d, PreferenceRegressor::Params::getInstance().r);
    }
}

///////////////////////////////////////////////////////////////////

#if 0
namespace
{

double objective_function(const std::vector<double> &x, std::vector<double>& grad, void* data)
{
    const PreferenceRegressor* regressor = static_cast<PreferenceRegressor*>(data);

    const unsigned M = x.size();

    if (!grad.empty())
    {
        const VectorXd  x_vec = Eigen::Map<const VectorXd>(&x[0], M);
        const MatrixXd& X     = regressor->X;
        const MatrixXd& C_inv = regressor->C_inv;
        const VectorXd& y     = regressor->y;
        const unsigned  N     = y.rows();
        const double    a     = regressor->a;
        const double    l     = regressor->l;

        MatrixXd k_derivative(M, N);

        for (unsigned i = 0; i < N; ++ i)
        {
            const double   tmp            = (- 0.5 / (l * l)) * (x_vec - X.col(i)).squaredNorm();
            const VectorXd k_i_derivative = a * std::exp(tmp) * (- 1.0 / (l * l) * (x_vec - X.col(i)));

            k_derivative.col(i) = k_i_derivative;
        }

        const MatrixXd grad_vec = k_derivative * (C_inv * y);
        Eigen::Map<VectorXd>(&grad[0], M) = grad_vec;
    }

    return regressor->estimate_y(Eigen::Map<const VectorXd>(&x[0], M));
}

}
#endif

VectorXd PreferenceRegressor::find_arg_max()
{
    const unsigned M = X.rows();

    assert (M != 0);

    int i; y.maxCoeff(&i);
#if 0
    const VectorXd x_initial = X.col(i);
    const VectorXd upper     = VectorXd::Constant(M, 1.0);
    const VectorXd lower     = VectorXd::Constant(M, 0.0);

    return nloptUtility::compute(x_initial, upper, lower, objective_function, static_cast<void*>(this), nlopt::LD_TNEWTON, 100);
#else
    return X.col(i);
#endif
}

void PreferenceRegressor::dampData(const std::string &dirPath) const
{
    // Export X using CSV
    Utility::exportMatrixToCsv(dirPath + "/X.csv", X);

    // Export D using CSV
    std::ofstream ofs_D(dirPath + "/D.csv");
    for (unsigned i = 0 ; i < D.size(); ++ i)
    {
        for (unsigned j = 0; j < D[i].size(); ++ j)
        {
            ofs_D << D[i][j];
            if (j + 1 != D[i].size()) ofs_D << ",";
        }
        ofs_D << std::endl;
    }
}

