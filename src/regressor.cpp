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
    constexpr auto kernel                      = mathtoolbox::GetArdMatern52Kernel;
    constexpr auto kernel_theta_i_derivative   = mathtoolbox::GetArdMatern52KernelThetaIDerivative;
    constexpr auto kernel_first_arg_derivative = mathtoolbox::GetArdMatern52KernelFirstArgDerivative;

    MatrixXd Regressor::calc_C(const MatrixXd& X, const double a, const double b, const VectorXd& r)
    {
        const unsigned N = X.cols();

        MatrixXd C(N, N);
        for (unsigned i = 0; i < N; ++i)
        {
            for (unsigned j = i; j < N; ++j)
            {
                const double value = kernel(X.col(i), X.col(j), concat(a, r));

                C(i, j) = value;
                C(j, i) = value;
            }
        }

        return C + b * MatrixXd::Identity(N, N);
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

    VectorXd
    Regressor::calc_k(const VectorXd& x, const MatrixXd& X, const double a, const double /*b*/, const VectorXd& r)
    {
        const unsigned N = X.cols();

        VectorXd k(N);
        for (unsigned i = 0; i < N; ++i)
        {
            k(i) = kernel(x, X.col(i), concat(a, r));
        }

        return k;
    }

    Eigen::MatrixXd Regressor::CalcSmallKSmallXDerivative(
        const Eigen::VectorXd& x, const Eigen::MatrixXd& X, const double a, const double b, const Eigen::VectorXd& r)
    {
        const unsigned N   = X.cols();
        const unsigned dim = X.rows();

        assert(dim == r.size());
        assert(dim != 0);

        MatrixXd k_x_derivative(dim, N);
        for (unsigned i = 0; i < N; ++i)
        {
            k_x_derivative.col(i) = kernel_first_arg_derivative(x, X.col(i), concat(a, r));
        }

        return k_x_derivative;
    }
} // namespace sequential_line_search
