#include <cmath>
#include <fstream>
#include <iostream>
#include <mathtoolbox/constants.hpp>
#include <mathtoolbox/log-determinant.hpp>
#include <mathtoolbox/probability-distributions.hpp>
#include <nlopt-util.hpp>
#include <sequential-line-search/preference-regressor.hpp>
#include <sequential-line-search/utils.hpp>

using Eigen::LLT;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// By define this macro, this class will report many verbose messages.
// #define VERBOSE

// By define this macro, this class will use a formulation based on "no noise" kernel matrix. That is, K = K(X, X) will
// be used instead of K = K(X, X) + sigma^{2} I. This macro can be enabled via CMake option.
// #define SEQUENTIAL_LINE_SEARCH_USE_NOISELESS_FORMULATION

// In this source file, we use the following notations:
//
// - a: the kernel signal variance
// - r: the kernel length scale
// - b: the noise level
//
// for making the source code compact.

#ifdef VERBOSE
#include <timer.hpp>
#endif

namespace
{
    using namespace sequential_line_search;

    inline VectorXd Concat(const double scalar, const VectorXd& vector)
    {
        VectorXd result(vector.size() + 1);

        result(0)                        = scalar;
        result.segment(1, vector.size()) = vector;

        return result;
    }

#ifdef SEQUENTIAL_LINE_SEARCH_USE_NOISELESS_FORMULATION
    const double b_fixed = 0.0;
#endif

#ifndef SEQUENTIAL_LINE_SEARCH_USE_NOISELESS_FORMULATION
    inline double CalcObjectiveNoiseLevelDerivative(const VectorXd&      y,
                                                    const LLT<MatrixXd>& K_llt,
                                                    const VectorXd&      K_inv_y,
                                                    const MatrixXd&      X,
                                                    const double         a,
                                                    const double         b,
                                                    const VectorXd&      r,
                                                    const double         b_prior_mean,
                                                    const double         b_prior_variance)
    {
        const MatrixXd K_y_grad_b = CalcLargeKYNoiseLevelDerivative(X, Concat(a, r), b);

        const double term_1 = 0.5 * K_inv_y.transpose() * K_y_grad_b * K_inv_y;
        const double term_2 = -0.5 * K_llt.solve(K_y_grad_b).trace();

        const double log_p_f_theta_grad_b = term_1 + term_2;

        const double log_prior =
            mathtoolbox::GetLogOfLogNormalDistDerivative(b, std::log(b_prior_mean), b_prior_variance);

        return log_p_f_theta_grad_b + log_prior;
    }
#endif

    inline VectorXd CalcObjectiveThetaDerivative(const VectorXd&             y,
                                                 const LLT<MatrixXd>&        K_llt,
                                                 const VectorXd&             K_inv_y,
                                                 const MatrixXd&             X,
                                                 const VectorXd&             kernel_hyperparams,
                                                 const double                a_prior_mean,
                                                 const double                a_prior_variance,
                                                 const double                r_prior_mean,
                                                 const double                r_prior_variance,
                                                 const KernelThetaDerivative kernel_theta_derivative)
    {
        VectorXd grad = VectorXd::Zero(kernel_hyperparams.size());

        const std::vector<MatrixXd> K_y_grad_r =
            CalcLargeKYThetaDerivative(X, kernel_hyperparams, kernel_theta_derivative);
        for (unsigned i = 0; i < kernel_hyperparams.size(); ++i)
        {
            const MatrixXd& K_y_grad_theta_i = K_y_grad_r[i];

            const double term_1 = 0.5 * K_inv_y.transpose() * K_y_grad_theta_i * K_inv_y;
            const double term_2 = -0.5 * K_llt.solve(K_y_grad_theta_i).trace();

            const double log_p_f_theta_grad_theta_i = term_1 + term_2;

            grad(i) += log_p_f_theta_grad_theta_i;
        }
        for (unsigned i = 0; i < kernel_hyperparams.size(); ++i)
        {
            const double prior_mean     = (i == 0) ? a_prior_mean : r_prior_mean;
            const double prior_variance = (i == 0) ? a_prior_variance : r_prior_variance;

            const double log_prior = mathtoolbox::GetLogOfLogNormalDistDerivative(
                kernel_hyperparams(i), std::log(prior_mean), prior_variance);

            grad(i) += log_prior;
        }

        return grad;
    }

    // log p(d_k | f)
    inline double calc_log_likelihood(const Preference& p, const VectorXd& y, const double btl_scale)
    {
        VectorXd tmp(p.size());
        for (unsigned i = 0; i < p.size(); ++i)
        {
            tmp(i) = y(p[i]);
        }
        return std::log(utils::CalcBtl(tmp, btl_scale));
    }

    // Log likelihood that will be maximized
    double objective(const std::vector<double>& x, std::vector<double>& grad, void* data)
    {
        const PreferenceRegressor* regressor = static_cast<PreferenceRegressor*>(data);

        const MatrixXd&                X = regressor->m_X;
        const std::vector<Preference>& D = regressor->m_D;
        const unsigned                 M = X.cols();
        const VectorXd                 y = Eigen::Map<const VectorXd>(&x[0], M);

        const double a = (regressor->m_use_map_hyperparams) ? x[M + 0] : regressor->m_default_kernel_signal_var;
#ifdef SEQUENTIAL_LINE_SEARCH_USE_NOISELESS_FORMULATION
        const double b = b_fixed;
#else
        const double b = (regressor->m_use_map_hyperparams) ? x[M + 1] : regressor->m_default_noise_level;
#endif
        const VectorXd r = (regressor->m_use_map_hyperparams)
                               ? VectorXd(Eigen::Map<const VectorXd>(&x[M + 2], X.rows()))
                               : VectorXd::Constant(X.rows(), regressor->m_default_kernel_length_scale);

        double obj = 0.0;

        // Log likelihood of data
        for (unsigned i = 0; i < D.size(); ++i)
        {
            obj += calc_log_likelihood(D[i], y, regressor->m_btl_scale);
        }

        // Constant
        constexpr double prod_of_two_and_pi = 2.0 * mathtoolbox::constants::pi;

        // Kernel matrix
        const MatrixXd K =
            regressor->m_use_map_hyperparams ? CalcLargeKY(X, Concat(a, r), b, regressor->GetKernel()) : regressor->m_K;
        const LLT<MatrixXd> K_llt = regressor->m_use_map_hyperparams ? LLT<MatrixXd>(K) : regressor->m_K_llt;

        // Log likelihood of y distribution
        const VectorXd K_inv_y   = K_llt.solve(y);
        const double   log_det_K = mathtoolbox::CalcLogDetOfSymmetricPositiveDefiniteMatrix(K_llt);
        const double   term1     = -0.5 * y.transpose() * K_inv_y;
        const double   term2     = -0.5 * log_det_K;
        const double   term3     = -0.5 * M * std::log(prod_of_two_and_pi);
        obj += term1 + term2 + term3;

        assert(!std::isnan(obj));

        // Priors for Gaussian process hyperparameters
        if (regressor->m_use_map_hyperparams)
        {
            const double a_prior = regressor->m_default_kernel_signal_var;
#ifndef SEQUENTIAL_LINE_SEARCH_USE_NOISELESS_FORMULATION
            const double b_prior = regressor->m_default_noise_level;
#endif
            const double r_prior  = regressor->m_default_kernel_length_scale;
            const double variance = regressor->m_kernel_hyperparams_prior_var;

            obj += mathtoolbox::GetLogOfLogNormalDist(a, std::log(a_prior), variance);
#ifndef SEQUENTIAL_LINE_SEARCH_USE_NOISELESS_FORMULATION
            obj += mathtoolbox::GetLogOfLogNormalDist(b, std::log(b_prior), variance);
#endif
            for (unsigned i = 0; i < r.rows(); ++i)
            {
                obj += mathtoolbox::GetLogOfLogNormalDist(r(i), std::log(r_prior), variance);
            }
        }

        // When the algorithm is gradient-based, compute the gradient vector
        const bool is_gradient_based = grad.size() == x.size();
        if (is_gradient_based)
        {
            VectorXd grad_y = VectorXd::Zero(y.rows());

            // Accumulate per-data derivatives
            const double btl_scale = regressor->m_btl_scale;
            for (unsigned i = 0; i < D.size(); ++i)
            {
                const Preference& p = D[i];
                const double      s = btl_scale;
                VectorXd          tmp1(p.size());
                for (unsigned i = 0; i < p.size(); ++i)
                {
                    tmp1(i) = y(p[i]);
                }
                const VectorXd tmp2 = utils::CalcBtlDerivative(tmp1, s) / utils::CalcBtl(tmp1, s);
                for (unsigned i = 0; i < p.size(); ++i)
                {
                    grad_y(p[i]) += tmp2(i);
                }
            }

            // Add the GP term
            grad_y += -K_inv_y;

            Eigen::Map<VectorXd>(&grad[0], grad_y.rows()) = grad_y;

            if (regressor->m_use_map_hyperparams)
            {
                const VectorXd grad_theta = CalcObjectiveThetaDerivative(y,
                                                                         K_llt,
                                                                         K_inv_y,
                                                                         X,
                                                                         Concat(a, r),
                                                                         regressor->m_default_kernel_signal_var,
                                                                         regressor->m_kernel_hyperparams_prior_var,
                                                                         regressor->m_default_kernel_length_scale,
                                                                         regressor->m_kernel_hyperparams_prior_var,
                                                                         regressor->GetKernelThetaDerivative());

                grad[M + 0] = grad_theta(0);
#ifdef SEQUENTIAL_LINE_SEARCH_USE_NOISELESS_FORMULATION
                grad[M + 1] = 0.0;
#else
                grad[M + 1] = CalcObjectiveNoiseLevelDerivative(y,
                                                                K_llt,
                                                                K_inv_y,
                                                                X,
                                                                a,
                                                                b,
                                                                r,
                                                                regressor->m_default_noise_level,
                                                                regressor->m_kernel_hyperparams_prior_var);
#endif
                const VectorXd grad_r = grad_theta.segment(1, r.size());
                for (unsigned i = 0; i < grad_r.size(); ++i)
                {
                    grad[M + 2 + i] = grad_r(i);
                }
            }
        }

        return obj;
    }
} // namespace

sequential_line_search::PreferenceRegressor::PreferenceRegressor(const MatrixXd&                X,
                                                                 const std::vector<Preference>& D,
                                                                 bool                           use_map_hyperparams,
                                                                 const double     default_kernel_signal_var,
                                                                 const double     default_kernel_length_scale,
                                                                 const double     default_noise_level,
                                                                 const double     kernel_hyperparams_prior_var,
                                                                 const double     btl_scale,
                                                                 const unsigned   num_map_estimation_iters,
                                                                 const KernelType kernel_type)
    : Regressor(kernel_type),
      m_use_map_hyperparams(use_map_hyperparams),
      m_X(X),
      m_D(D),
      m_default_kernel_signal_var(default_kernel_signal_var),
      m_default_kernel_length_scale(default_kernel_length_scale),
      m_default_noise_level(default_noise_level),
      m_kernel_hyperparams_prior_var(kernel_hyperparams_prior_var),
      m_btl_scale(btl_scale)
{
    if (X.cols() == 0 || D.size() == 0)
    {
        return;
    }

    PerformMapEstimation(num_map_estimation_iters);

    m_K     = CalcLargeKY(X, m_kernel_hyperparams, m_noise_hyperparam, m_kernel);
    m_K_llt = LLT<MatrixXd>(m_K);
}

double sequential_line_search::PreferenceRegressor::PredictMu(const VectorXd& x) const
{
    const VectorXd k = CalcSmallK(x, m_X, m_kernel_hyperparams, m_kernel);
    return k.transpose() * m_K_llt.solve(m_y);
}

double sequential_line_search::PreferenceRegressor::PredictSigma(const VectorXd& x) const
{
    const VectorXd k = CalcSmallK(x, m_X, m_kernel_hyperparams, m_kernel);

    // This code assumes that the kernel is either ARD squared exponential or ARD Matern and the first hyperparameter
    // represents the intensity of the kernel.
    assert(m_kernel_hyperparams.size() == x.size() + 1);
    const double intensity = m_kernel_hyperparams[0];

    return std::sqrt(intensity - k.transpose() * m_K_llt.solve(k));
}

VectorXd sequential_line_search::PreferenceRegressor::PredictMuDerivative(const VectorXd& x) const
{
    // TODO: Incorporate a mean function
    const MatrixXd k_x_derivative =
        CalcSmallKSmallXDerivative(x, m_X, m_kernel_hyperparams, m_kernel_first_arg_derivative);
    return k_x_derivative * m_K_llt.solve(m_y);
}

VectorXd sequential_line_search::PreferenceRegressor::PredictSigmaDerivative(const VectorXd& x) const
{
    const MatrixXd k_x_derivative =
        CalcSmallKSmallXDerivative(x, m_X, m_kernel_hyperparams, m_kernel_first_arg_derivative);
    const VectorXd k     = CalcSmallK(x, m_X, m_kernel_hyperparams, m_kernel);
    const double   sigma = PredictSigma(x);
    return -(1.0 / sigma) * k_x_derivative * m_K_llt.solve(k);
}

void sequential_line_search::PreferenceRegressor::PerformMapEstimation(const unsigned num_iters)
{
    const unsigned M = m_X.cols();
    const unsigned d = m_X.rows();

    // When hyperparameters are estimated jointly, the number of the optimization variables increases by "2 + d"
    const unsigned opt_dim = m_use_map_hyperparams ? M + 2 + d : M;

    VectorXd upper = VectorXd::Constant(opt_dim, +1e+01);
    VectorXd lower = VectorXd::Constant(opt_dim, -1e+01);
    VectorXd x_ini = VectorXd::Constant(opt_dim, 0.0);

    if (m_use_map_hyperparams)
    {
        // Set bounding conditions for hyperparameters
        lower.segment(M, 2 + d) = VectorXd::Constant(2 + d, 1e-08);

        // Set initial solutions for hyperparameters
        x_ini(M + 0) = m_default_kernel_signal_var;
#ifdef SEQUENTIAL_LINE_SEARCH_USE_NOISELESS_FORMULATION
        x_ini(M + 1) = 0.5 * (upper(M + 1) + lower(M + 1));
#else
        x_ini(M + 1)       = m_default_noise_level;
#endif
        x_ini.segment(M + 2, d) = VectorXd::Constant(d, m_default_kernel_length_scale);

        // Ensure that the initial solution is in the bounding box
        x_ini = x_ini.cwiseMax(lower).cwiseMin(upper);
    }

    // Calculate kernel matrices if hyperparameters are not estimated by the MAP estimation
    if (!m_use_map_hyperparams)
    {
        m_kernel_hyperparams =
            Concat(m_default_kernel_signal_var, VectorXd::Constant(d, m_default_kernel_length_scale));
        m_noise_hyperparam = m_default_noise_level;

        m_K     = CalcLargeKY(m_X, m_kernel_hyperparams, m_noise_hyperparam, m_kernel);
        m_K_llt = LLT<MatrixXd>(m_K);
    }

#ifdef VERBOSE
    timer::Timer t("PreferenceRegressor::PerformMapEstimation");
#endif

    const VectorXd x_opt = nloptutil::solve(x_ini, upper, lower, objective, nlopt::LD_TNEWTON, this, true, num_iters);

    if (m_use_map_hyperparams)
    {
        m_y = x_opt.segment(0, M);

        m_kernel_hyperparams = Concat(x_opt(M + 0), x_opt.segment(M + 2, d));
#ifdef SEQUENTIAL_LINE_SEARCH_USE_NOISELESS_FORMULATION
        m_noise_hyperparam = b_fixed;
#else
        m_noise_hyperparam = x_opt(M + 1);
#endif

#ifdef VERBOSE
        std::cout << "Learned hyperparameters ... ";
        std::cout << "a : " << a << ", \tb : " << b << ", \tr : " << r.transpose() << std::endl;
#endif
    }
    else
    {
        m_y = x_opt;
    }

#ifdef VERBOSE
    std::cout << "Estimated values: " << y.transpose().format(Eigen::IOFormat(3)) << std::endl;
#endif
}

VectorXd sequential_line_search::PreferenceRegressor::FindArgMax() const
{
    int i;
    m_y.maxCoeff(&i);
    return m_X.col(i);
}

void sequential_line_search::PreferenceRegressor::DampData(const std::string& dir_path, const std::string& prefix) const
{
    // Export X using CSV
    utils::ExportMatrixToCsv(dir_path + "/" + prefix + "X.csv", m_X);

    // Export D using CSV
    std::ofstream ofs_D(dir_path + "/" + prefix + "D.csv");
    for (unsigned i = 0; i < m_D.size(); ++i)
    {
        for (unsigned j = 0; j < m_D[i].size(); ++j)
        {
            ofs_D << m_D[i][j];

            if (j + 1 != m_D[i].size())
            {
                ofs_D << ",";
            }
        }
        ofs_D << std::endl;
    }
}
