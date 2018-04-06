#include "regressor.h"
#include "utility.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd Regressor::calc_C(const MatrixXd& X, const double a, const double b, const VectorXd& r)
{
    const unsigned N = X.cols();

    MatrixXd C(N, N);
    for (unsigned i = 0; i < N; ++ i)
    {
        for (unsigned j = i; j < N; ++ j)
        {
            const double value = Utility::ARD_squared_exponential_kernel(X.col(i), X.col(j), a, r);
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
    for (unsigned i = 0; i < N; ++ i)
    {
        for (unsigned j = i; j < N; ++ j)
        {
            const double value = Utility::ARD_squared_exponential_kernel_derivative_a(X.col(i), X.col(j), a, r);
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

MatrixXd Regressor::calc_C_grad_r_i(const MatrixXd& X, const double a, const double /*b*/, const VectorXd& r, const unsigned index)
{
    const unsigned N = X.cols();

    MatrixXd C_grad_r(N, N);
    for (unsigned i = 0; i < N; ++ i)
    {
        for (unsigned j = i; j < N; ++ j)
        {
            const VectorXd value = Utility::ARD_squared_exponential_kernel_derivative_r(X.col(i), X.col(j), a, r);
            C_grad_r(i, j) = value(index);
            C_grad_r(j, i) = value(index);
        }
    }

    return C_grad_r;
}

VectorXd Regressor::calc_k(const VectorXd& x, const MatrixXd& X, const double a, const double /*b*/, const VectorXd& r)
{
    const unsigned N = X.cols();

    VectorXd k(N);
    for (unsigned i = 0; i < N; ++ i)
    {
        k(i) = Utility::ARD_squared_exponential_kernel(x, X.col(i), a, r);
    }

    return k;
}
