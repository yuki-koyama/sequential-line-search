#include "core.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <sequential-line-search/acquisition-function.hpp>
#include <sequential-line-search/gaussian-process-regressor.hpp>
#include <sequential-line-search/utils.hpp>

using namespace sequential_line_search;

using Eigen::MatrixXd;
using Eigen::VectorXd;

Core::Core()
{
    X     = MatrixXd::Zero(0, 0);
    y     = VectorXd::Zero(0);
    x_max = VectorXd::Zero(0);
    y_max = NAN;

    computeRegression();
}

void Core::proceedOptimization()
{
    const VectorXd x = (X.cols() == 0) ? utils::GenerateRandomVector(1) : acquisition_func::FindNextPoint(*regressor);
    const double   y = evaluateObjectiveFunction(x);

    std::cout << "x: " << x.transpose() << ((X.cols() == 0) ? " (randomly chosen)" : "") << std::endl;
    std::cout << "y: " << y << std::endl;

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

double Core::evaluateObjectiveFunction(const Eigen::VectorXd& x) const
{
    return 1.0 - 1.5 * x(0) * std::sin(x(0) * 13.0);
}

void Core::computeRegression()
{
    regressor = std::make_shared<GaussianProcessRegressor>(X, y);
}
