#include <mathtoolbox/kernel-functions.hpp>
#include <sequential-line-search/regressor.h>
#include <sequential-line-search/utils.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline VectorXd concat(const double a, const VectorXd& b)
{
    VectorXd result(b.size() + 1);

    result(0)                   = a;
    result.segment(1, b.size()) = b;

    return result;
}

namespace sequential_line_search
{
#ifdef SEQUENTIAL_LINE_SEARCH_USE_ARD_SQUARED_EXP_KERNEL
    constexpr auto kernel                      = mathtoolbox::GetArdSquaredExpKernel;
    constexpr auto kernel_theta_i_derivative   = mathtoolbox::GetArdSquaredExpKernelThetaIDerivative;
    constexpr auto kernel_first_arg_derivative = mathtoolbox::GetArdSquaredExpKernelFirstArgDerivative;
#else
    constexpr auto kernel                      = mathtoolbox::GetArdMatern52Kernel;
    constexpr auto kernel_theta_i_derivative   = mathtoolbox::GetArdMatern52KernelThetaIDerivative;
    constexpr auto kernel_first_arg_derivative = mathtoolbox::GetArdMatern52KernelFirstArgDerivative;
#endif

    VectorXd Regressor::PredictMaximumPointFromData() const
    {
        const int num_data_points = getX().cols();

        VectorXd f(num_data_points);
        for (int i = 0; i < num_data_points; ++i)
        {
            f(i) = PredictMu(getX().col(i));
        }

        int best_index;
        f.maxCoeff(&best_index);

        return getX().col(best_index);
    }

    MatrixXd Regressor::calc_C_grad_a(const MatrixXd& X, const double a, const double /*b*/, const VectorXd& r)
    {
        const unsigned N = X.cols();

        MatrixXd C_grad_a(N, N);
        for (unsigned i = 0; i < N; ++i)
        {
            for (unsigned j = i; j < N; ++j)
            {
                const double value = kernel_theta_i_derivative(X.col(i), X.col(j), concat(a, r), 0);

                C_grad_a(i, j) = value;
                C_grad_a(j, i) = value;
            }
        }

        return C_grad_a;
    }

    MatrixXd Regressor::calc_C_grad_b(const MatrixXd& X, const double /*a*/, const double /*b*/, const VectorXd& /*r*/)
    {
        const unsigned N = X.cols();
        return MatrixXd::Identity(N, N);
    }

    MatrixXd Regressor::calc_C_grad_r_i(
        const MatrixXd& X, const double a, const double /*b*/, const VectorXd& r, const unsigned index)
    {
        const unsigned N = X.cols();

        MatrixXd C_grad_r(N, N);
        for (unsigned i = 0; i < N; ++i)
        {
            for (unsigned j = i; j < N; ++j)
            {
                const double value = kernel_theta_i_derivative(X.col(i), X.col(j), concat(a, r), index + 1);

                C_grad_r(i, j) = value;
                C_grad_r(j, i) = value;
            }
        }

        return C_grad_r;
    }
} // namespace sequential_line_search

VectorXd
sequential_line_search::CalcSmallK(const VectorXd& x, const MatrixXd& X, const Eigen::VectorXd& kernel_hyperparameters)
{
    const unsigned N = X.cols();

    VectorXd k(N);
    for (unsigned i = 0; i < N; ++i)
    {
        k(i) = kernel(x, X.col(i), kernel_hyperparameters);
    }

    return k;
}

Eigen::MatrixXd sequential_line_search::CalcLargeKY(const Eigen::MatrixXd& X,
                                                    const Eigen::VectorXd& kernel_hyperparameters,
                                                    const double           noise_level)
{
    const unsigned N   = X.cols();
    const MatrixXd K_f = CalcLargeKF(X, kernel_hyperparameters);

    return K_f + noise_level * MatrixXd::Identity(N, N);
}

Eigen::MatrixXd sequential_line_search::CalcLargeKF(const Eigen::MatrixXd& X,
                                                    const Eigen::VectorXd& kernel_hyperparameters)
{
    const unsigned N = X.cols();

    MatrixXd C(N, N);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = i; j < N; ++j)
        {
            const double value = kernel(X.col(i), X.col(j), kernel_hyperparameters);

            C(i, j) = value;
            C(j, i) = value;
        }
    }
    return C;
}

Eigen::MatrixXd sequential_line_search::CalcSmallKSmallXDerivative(const Eigen::VectorXd& x,
                                                                   const Eigen::MatrixXd& X,
                                                                   const Eigen::VectorXd& kernel_hyperparameters)
{
    const unsigned N   = X.cols();
    const unsigned dim = X.rows();

    assert(dim != 0);

    MatrixXd k_x_derivative(dim, N);
    for (unsigned i = 0; i < N; ++i)
    {
        k_x_derivative.col(i) = kernel_first_arg_derivative(x, X.col(i), kernel_hyperparameters);
    }

    return k_x_derivative;
}
