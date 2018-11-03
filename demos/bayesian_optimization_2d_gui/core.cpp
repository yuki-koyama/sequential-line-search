#include "core.h"
#include <iostream>
#include <cmath>
#include <random>
#include <sequential-line-search/sequential-line-search.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace{
std::random_device seed;
std::default_random_engine gen(seed());
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
}

Core::Core() : show_slider_value(false)
{
    X = MatrixXd::Zero(0, 0);
    y = VectorXd::Zero(0);

    x_max = VectorXd::Zero(0);
    y_max = NAN;
}

VectorXd Core::findNextPoint() const
{
    return ExpectedImprovement::findNextPoint(*regressor);
}

void Core::proceedOptimization()
{
    const VectorXd x = [&]() {
        if (X.cols() == 0)
        {
            VectorXd x(2);
            x << 0.05 + 0.90 * uniform_dist(gen), 0.05 + 0.90 * uniform_dist(gen);
            return x;
        }
        return findNextPoint();
    }();
    const double   y = evaluateObjectiveFunction(x);

    if (std::isnan(y_max) || y > y_max)
    {
        x_max = x;
        y_max = y;
    }

    addData(x, y);
    computeRegression();

    std::cout << y_max << std::endl;

    return;
}

void Core::addData(const VectorXd &x, double y)
{
    if (X.rows() == 0)
    {
        this->X = x;
        this->y = VectorXd::Constant(1, y);
        return;
    }

    const unsigned D = X.rows();
    const unsigned N = X.cols();

    MatrixXd newX(D, N + 1);
    newX.block(0, 0, D, N) = X;
    newX.col(N) = x;
    this->X = newX;

    VectorXd newY(this->y.rows() + 1);
    newY << this->y, y;
    this->y = newY;
}

double Core::evaluateObjectiveFunction(Eigen::VectorXd x) const
{
    assert(x.rows() == 2);
#if 0
    return 0.4 * std::max(0.0, std::sin(x(0)) + x(0) / 3.0 + std::sin(x(0) * 12.0) + std::sin(x(1)) + x(1) / 3.0 + std::sin(x(1) * 12.0) - 1.0);
#elif 1
    auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma)
    {
        return std::exp(- (x - mu).squaredNorm() / (sigma * sigma));
    };

    return lambda(x, Eigen::Vector2d(0.3, 0.3), 0.3) + 1.5 * lambda(x, Eigen::Vector2d(0.7, 0.7), 0.4);
#else
    double sum = 0.0;
    for (unsigned i = 0; i < x.rows(); ++ i)
    {
        sum += std::sin(x(i)) + x(i) / 3.0 + std::sin(12.0 * x(i));
    }
    return sum;
#endif
}

void Core::computeRegression()
{
    regressor = std::make_shared<GaussianProcessRegressor>(X, y);
}
