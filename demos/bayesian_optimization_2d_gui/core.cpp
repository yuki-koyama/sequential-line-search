#include "core.hpp"
#include <cmath>
#include <iostream>
#include <rand-util.hpp>
#include <sequential-line-search/acquisition-function.hpp>
#include <sequential-line-search/gaussian-process-regressor.hpp>

using namespace sequential_line_search;
using Eigen::MatrixXd;
using Eigen::VectorXd;

Core::Core() : show_slider_value(false)
{
    X = MatrixXd::Zero(0, 0);
    y = VectorXd::Zero(0);

    x_max = VectorXd::Zero(0);
    y_max = NAN;
}

VectorXd Core::findNextPoint() const
{
    return acquisition_func::FindNextPoint(*regressor);
}

void Core::proceedOptimization()
{
    const VectorXd x = [&]() {
        if (X.cols() == 0)
        {
            VectorXd x(2);
            x << 0.05 + 0.90 * randutil::GenNumFromUniformDist(), 0.05 + 0.90 * randutil::GenNumFromUniformDist();
            return x;
        }
        return findNextPoint();
    }();
    const double y = evaluateObjectiveFunction(x);

    std::cout << y << std::endl;

    addData(x, y);
    computeRegression();

    const int num_data_points = X.cols();

    VectorXd f(num_data_points);
    for (int i = 0; i < X.cols(); ++i)
    {
        f(i) = regressor->PredictMu(X.col(i));

        int best_index;
        y_max = f.maxCoeff(&best_index);
        x_max = X.col(best_index);
    }
}

void Core::addData(const VectorXd& x, double y)
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
    newX.col(N)            = x;
    this->X                = newX;

    VectorXd newY(this->y.rows() + 1);
    newY << this->y, y;
    this->y = newY;
}

double Core::evaluateObjectiveFunction(Eigen::VectorXd x) const
{
    assert(x.rows() == 2);

    auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma) {
        return std::exp(-(x - mu).squaredNorm() / (sigma * sigma));
    };

    return lambda(x, Eigen::Vector2d(0.3, 0.3), 0.3) + 1.5 * lambda(x, Eigen::Vector2d(0.7, 0.7), 0.4);
}

void Core::computeRegression()
{
    regressor = std::make_shared<GaussianProcessRegressor>(X, y);
}
