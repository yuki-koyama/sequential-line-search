#include <mathtoolbox/kernel-functions.hpp>
#include <sequential-line-search/regressor.hpp>
#include <sequential-line-search/utils.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
    inline VectorXd Concat(const double scalar, const VectorXd& vector)
    {
        return (VectorXd(vector.size() + 1) << scalar, vector).finished();
    }
} // namespace

namespace sequential_line_search
{
#ifdef SEQUENTIAL_LINE_SEARCH_USE_ARD_SQUARED_EXP_KERNEL
    constexpr auto kernel                      = mathtoolbox::GetArdSquaredExpKernel;
    constexpr auto kernel_theta_derivative     = mathtoolbox::GetArdSquaredExpKernelThetaDerivative;
    constexpr auto kernel_first_arg_derivative = mathtoolbox::GetArdSquaredExpKernelFirstArgDerivative;
#else
    constexpr auto kernel                      = mathtoolbox::GetArdMatern52Kernel;
    constexpr auto kernel_theta_derivative     = mathtoolbox::GetArdMatern52KernelThetaDerivative;
    constexpr auto kernel_first_arg_derivative = mathtoolbox::GetArdMatern52KernelFirstArgDerivative;
#endif
} // namespace sequential_line_search

VectorXd sequential_line_search::Regressor::PredictMaximumPointFromData() const
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

VectorXd sequential_line_search::Regressor::GetKernelHyperparams() const
{
    return Concat(geta(), getr());
}

VectorXd
sequential_line_search::CalcSmallK(const VectorXd& x, const MatrixXd& X, const VectorXd& kernel_hyperparameters)
{
    const unsigned N = X.cols();

    VectorXd k(N);
    for (unsigned i = 0; i < N; ++i)
    {
        k(i) = kernel(x, X.col(i), kernel_hyperparameters);
    }

    return k;
}

MatrixXd
sequential_line_search::CalcLargeKY(const MatrixXd& X, const VectorXd& kernel_hyperparameters, const double noise_level)
{
    const unsigned N   = X.cols();
    const MatrixXd K_f = CalcLargeKF(X, kernel_hyperparameters);

    return K_f + noise_level * MatrixXd::Identity(N, N);
}

MatrixXd sequential_line_search::CalcLargeKF(const MatrixXd& X, const VectorXd& kernel_hyperparameters)
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

MatrixXd sequential_line_search::CalcSmallKSmallXDerivative(const VectorXd& x,
                                                            const MatrixXd& X,
                                                            const VectorXd& kernel_hyperparameters)
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

std::vector<MatrixXd> sequential_line_search::CalcLargeKYThetaDerivative(const MatrixXd& X,
                                                                         const VectorXd& kernel_hyperparameters)
{
    const unsigned N = X.cols();

    std::vector<MatrixXd> tensor(kernel_hyperparameters.size(), MatrixXd(N, N));

    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = i; j < N; ++j)
        {
            const VectorXd grad = kernel_theta_derivative(X.col(i), X.col(j), kernel_hyperparameters);

            for (unsigned k = 0; k < kernel_hyperparameters.size(); ++k)
            {
                tensor[k](i, j) = grad(k);
                tensor[k](j, i) = grad(k);
            }
        }
    }

    return tensor;
}

MatrixXd sequential_line_search::CalcLargeKYNoiseLevelDerivative(const MatrixXd& X,
                                                                 const VectorXd& kernel_hyperparameters,
                                                                 const double    noise_level)
{
    return MatrixXd::Identity(X.cols(), X.cols());
}
