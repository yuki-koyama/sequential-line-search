#include "core.h"
#include <cmath>

double Core::evaluateObjectiveFunction(const Eigen::VectorXd& x) const
{
    assert(x.rows() == 2);
#if 0
    return 0.4 * std::max(0.0, std::sin(x(0)) + x(0) / 3.0 + std::sin(x(0) * 12.0) + std::sin(x(1)) + x(1) / 3.0 + std::sin(x(1) * 12.0) - 1.0);
#elif 0
    return 1.0 - std::abs(x(0) - 0.5) - std::abs(x(1) - 0.5);
#elif 0
    return std::exp(-(x - VectorXd::Constant(x.rows(), 0.5)).squaredNorm() / (0.5 * 0.5));
#elif 0
    double sum = 0.0;
    for (unsigned i = 0; i < x.rows(); ++i)
    {
        sum += std::sin(x(i)) + x(i) / 3.0 + std::sin(12.0 * x(i));
    }
    return sum;
#elif 1
    auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma) {
        return std::exp(-(x - mu).squaredNorm() / (sigma * sigma));
    };

    return lambda(x, Eigen::Vector2d(0.3, 0.3), 0.3) + 1.5 * lambda(x, Eigen::Vector2d(0.7, 0.7), 0.4);
#else
    auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma) {
        return std::exp(-(x - mu).squaredNorm() / (2.0 * sigma * sigma));
    };

    return lambda(x, Eigen::Vector2d(0.5, 0.5), 0.5);
#endif
}
