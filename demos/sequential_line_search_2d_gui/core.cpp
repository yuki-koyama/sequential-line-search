#include "core.hpp"
#include <cmath>

double Core::evaluateObjectiveFunction(const Eigen::VectorXd& x) const
{
    assert(x.rows() == 2);

    auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma) {
        return std::exp(-(x - mu).squaredNorm() / (sigma * sigma));
    };

    return lambda(x, Eigen::Vector2d(0.3, 0.3), 0.3) + 1.5 * lambda(x, Eigen::Vector2d(0.7, 0.7), 0.4);
}
