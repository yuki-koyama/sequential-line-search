#include <Eigen/LU>
#include <cmath>
#include <iostream>
#include <mathtoolbox/constants.hpp>
#include <mathtoolbox/probability-distributions.hpp>
#include <nlopt-util.hpp>
#include <sequential-line-search/gaussian-process-regressor.hpp>
#include <sequential-line-search/utils.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
    using namespace sequential_line_search;

    const bool   use_log_normal_prior  = true;
    const double a_prior_mu            = std::log(0.500);
    const double a_prior_sigma_squared = 0.50;
    const double b_prior_mu            = std::log(1e-04);
    const double b_prior_sigma_squared = 0.50;
    const double r_prior_mu            = std::log(0.500);
    const double r_prior_sigma_squared = 0.50;

    inline VectorXd Concat(const double scalar, const VectorXd& vector)
    {
        VectorXd result(vector.size() + 1);

        result(0)                        = scalar;
        result.segment(1, vector.size()) = vector;

        return result;
    }

    double calc_grad_a_prior(const double a)
    {
        return mathtoolbox::GetLogOfLogNormalDistDerivative(a, a_prior_mu, a_prior_sigma_squared);
    }

    double calc_grad_b_prior(const double b)
    {
        return mathtoolbox::GetLogOfLogNormalDistDerivative(b, b_prior_mu, b_prior_sigma_squared);
    }

    double calc_grad_r_i_prior(const Eigen::VectorXd& r, const int index)
    {
        return mathtoolbox::GetLogOfLogNormalDistDerivative(r(index), r_prior_mu, r_prior_sigma_squared);
    }

    double calc_a_prior(const double a)
    {
        return mathtoolbox::GetLogOfLogNormalDist(a, a_prior_mu, a_prior_sigma_squared);
    }

    double calc_b_prior(const double b)
    {
        return mathtoolbox::GetLogOfLogNormalDist(b, b_prior_mu, b_prior_sigma_squared);
    }

    double calc_r_i_prior(const Eigen::VectorXd& r, const int index)
    {
        return mathtoolbox::GetLogOfLogNormalDist(r(index), r_prior_mu, r_prior_sigma_squared);
    }

    double calc_grad_b(const MatrixXd& X,
                       const MatrixXd& C_inv,
                       const VectorXd& y,
                       const double    a,
                       const double    b,
                       const VectorXd& r)
    {
        const MatrixXd C_grad_b = CalcLargeKYNoiseLevelDerivative(X, Concat(a, r), b);
        const double   term1    = +0.5 * y.transpose() * C_inv * C_grad_b * C_inv * y;
        const double   term2    = -0.5 * (C_inv * C_grad_b).trace();
        return term1 + term2 + (use_log_normal_prior ? calc_grad_b_prior(b) : 0.0);
    }

    VectorXd
    calc_grad_theta(const MatrixXd& X, const MatrixXd& C_inv, const VectorXd& y, const VectorXd& kernel_hyperparams)
    {
        const std::vector<MatrixXd> tensor = CalcLargeKYThetaDerivative(X, kernel_hyperparams);

        VectorXd grad(kernel_hyperparams.size());
        for (unsigned i = 0; i < kernel_hyperparams.size(); ++i)
        {
            const MatrixXd& K_y_grad_r_i = tensor[i];

            const double term1 = +0.5 * y.transpose() * C_inv * K_y_grad_r_i * C_inv * y;
            const double term2 = -0.5 * (C_inv * K_y_grad_r_i).trace();

            const double prior =
                use_log_normal_prior
                    ? (i == 0
                           ? calc_grad_a_prior(kernel_hyperparams(i))
                           : calc_grad_r_i_prior(kernel_hyperparams.segment(1, kernel_hyperparams.size() - 1), i - 1))
                    : 0.0;

            grad(i) = term1 + term2 + prior;
        }

        return grad;
    }

    VectorXd calc_grad(const MatrixXd& X,
                       const MatrixXd& C_inv,
                       const VectorXd& y,
                       const double    a,
                       const double    b,
                       const VectorXd& r)
    {
        const unsigned D = X.rows();

        VectorXd grad(D + 2);

        const VectorXd grad_theta = calc_grad_theta(X, C_inv, y, Concat(a, r));

        grad(0)            = grad_theta(0);
        grad(1)            = calc_grad_b(X, C_inv, y, a, b, r);
        grad.segment(2, D) = grad_theta.segment(1, D);

        return grad;
    }

    struct Data
    {
        const MatrixXd X;
        const VectorXd y;
    };

    // For counting the number of function evaluations
    unsigned count;

    // Log likelihood that will be maximized
    double objective(const std::vector<double>& x, std::vector<double>& grad, void* data)
    {
        // For counting the number of function evaluations
        ++count;

        const MatrixXd& X = static_cast<const Data*>(data)->X;
        const VectorXd& y = static_cast<const Data*>(data)->y;

        const unsigned N = X.cols();

        const double   a = x[0];
        const double   b = x[1];
        const VectorXd r = Eigen::Map<const VectorXd>(&x[2], x.size() - 2);

        const MatrixXd C     = CalcLargeKY(X, Concat(a, r), b);
        const MatrixXd C_inv = C.inverse();

        // When the algorithm is gradient-based, compute the gradient vector
        if (grad.size() == x.size())
        {
            const VectorXd g = calc_grad(X, C_inv, y, a, b, r);
            for (unsigned i = 0; i < g.rows(); ++i)
            {
                grad[i] = g(i);
            }
        }

        // Constant
        constexpr double prod_of_two_and_pi = 2.0 * mathtoolbox::constants::pi;

        const double term1 = -0.5 * y.transpose() * C_inv * y;
        const double term2 = -0.5 * std::log(C.determinant());
        const double term3 = -0.5 * N * std::log(prod_of_two_and_pi);

        // Computing the regularization terms from a prior assumptions
        const double a_prior = calc_a_prior(a);
        const double b_prior = calc_b_prior(b);
        const double r_prior = [&r]() {
            double sum = 0.0;
            for (unsigned i = 0; i < r.rows(); ++i)
            {
                sum += calc_r_i_prior(r, i);
            }
            return sum;
        }();
        const double regularization = use_log_normal_prior ? (a_prior + b_prior + r_prior) : 0.0;

        return term1 + term2 + term3 + regularization;
    }
} // namespace

namespace sequential_line_search
{
    GaussianProcessRegressor::GaussianProcessRegressor(const MatrixXd& X, const VectorXd& y)
    {
        this->X = X;
        this->y = y;

        if (X.rows() == 0)
        {
            return;
        }

        PerformMapEstimation();

        C     = CalcLargeKY(X, Concat(a, r), b);
        C_inv = C.inverse();
    }

    GaussianProcessRegressor::GaussianProcessRegressor(const Eigen::MatrixXd& X,
                                                       const Eigen::VectorXd& y,
                                                       const Eigen::VectorXd& kernel_hyperparams,
                                                       double                 b)
    {
        this->X = X;
        this->y = y;
        this->a = kernel_hyperparams[0];
        this->b = b;
        this->r = kernel_hyperparams.segment(1, kernel_hyperparams.size() - 1);

        if (X.rows() == 0)
        {
            return;
        }

        C     = CalcLargeKY(X, Concat(a, r), b);
        C_inv = C.inverse();
    }

    double GaussianProcessRegressor::PredictMu(const VectorXd& x) const
    {
        // TODO: Incorporate a mean function
        const VectorXd k = CalcSmallK(x, X, Concat(a, r));
        return k.transpose() * C_inv * y;
    }

    double GaussianProcessRegressor::PredictSigma(const VectorXd& x) const
    {
        const VectorXd k = CalcSmallK(x, X, Concat(a, r));
        return std::sqrt(a - k.transpose() * C_inv * k);
    }

    Eigen::VectorXd GaussianProcessRegressor::PredictMuDerivative(const Eigen::VectorXd& x) const
    {
        // TODO: Incorporate a mean function
        const MatrixXd k_x_derivative = CalcSmallKSmallXDerivative(x, X, Concat(a, r));
        return k_x_derivative * C_inv * y;
    }

    Eigen::VectorXd GaussianProcessRegressor::PredictSigmaDerivative(const Eigen::VectorXd& x) const
    {
        const MatrixXd k_x_derivative = CalcSmallKSmallXDerivative(x, X, Concat(a, r));
        const VectorXd k              = CalcSmallK(x, X, Concat(a, r));
        const double   sigma          = PredictSigma(x);
        return -(1.0 / sigma) * k_x_derivative * C_inv * k;
    }

    void GaussianProcessRegressor::PerformMapEstimation()
    {
        const unsigned D = X.rows();

        Data data{X, y};

        const VectorXd x_ini = [&]() {
            VectorXd x(D + 2);

            x(0)            = std::exp(a_prior_mu);
            x(1)            = std::exp(b_prior_mu);
            x.segment(2, D) = VectorXd::Constant(D, std::exp(r_prior_mu));

            return x;
        }();

        const VectorXd upper = VectorXd::Constant(D + 2, 5e+01);
        const VectorXd lower = VectorXd::Constant(D + 2, 1e-08);

        const VectorXd x_glo = nloptutil::solve(x_ini, upper, lower, objective, nlopt::GN_DIRECT, &data, true, 300);
        const VectorXd x_loc = nloptutil::solve(x_glo, upper, lower, objective, nlopt::LD_TNEWTON, &data, true, 1000);

        a = x_loc(0);
        b = x_loc(1);
        r = x_loc.block(2, 0, D, 1);
    }
} // namespace sequential_line_search
