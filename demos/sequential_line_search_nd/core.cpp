#include "core.h"
#include <iostream>
#include <cmath>
#include <sequential-line-search/sequential-line-search.h>

// #define TWO_DIM

namespace
{
    constexpr unsigned test_dimension = 8;
}

using Eigen::MatrixXd;
using Eigen::VectorXd;

Core::Core() : use_MAP_hyperparameters(false)
{
    X = MatrixXd::Zero(0, 0);
    D.clear();

    x_max = VectorXd::Zero(0);
    y_max = NAN;
}

void Core::proceedOptimization(double slider_position)
{
    // Add new preference data
    const VectorXd x = computeParametersFromSlider(slider_position);
    addData(std::vector<VectorXd>{ x, slider->orig_0, slider->orig_1 });

    // Compute regression
    computeRegression();

    // Check the current best
    x_max = regressor->find_arg_max();
    y_max = regressor->estimate_y(x_max);

    // Update slider ends
    updateSliderEnds();
}

void Core::addData(const VectorXd &x1, const VectorXd &x2)
{
    addData(std::vector<Eigen::VectorXd>{x1, x2});
}

void Core::addData(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, const Eigen::VectorXd &x3)
{
    addData(std::vector<Eigen::VectorXd>{x1, x2, x3});
}

void Core::addData(const std::vector<Eigen::VectorXd> &xs)
{
    if (X.rows() == 0)
    {
        this->X = MatrixXd(xs[0].rows(), xs.size());
        for (unsigned i = 0; i < xs.size(); ++ i) X.col(i) = xs[i];
        std::vector<unsigned> indices(xs.size());
        for (unsigned i = 0; i < xs.size(); ++ i) indices[i] = i;
        D.push_back(Preference(indices));
        return;
    }

    const unsigned d = X.rows();
    const unsigned N = X.cols();

    MatrixXd newX(d, N + xs.size());
    newX.block(0, 0, d, N) = X;
    for (unsigned i = 0; i < xs.size(); ++ i) newX.col(N + i) = xs[i];
    this->X = newX;

    std::vector<unsigned> indices(xs.size());
    for (unsigned i = 0; i < xs.size(); ++ i) indices[i] = N + i;
    D.push_back(Preference(indices));

    SliderUtility::mergeData(X, D, 5e-03);
}

double Core::evaluateObjectiveFunction(const Eigen::VectorXd& x) const
{
#ifdef TWO_DIM
    assert(x.rows() == 2);

    auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma)
    {
        return std::exp(- (x - mu).squaredNorm() / (sigma * sigma));
    };

    return lambda(x, Eigen::Vector2d(0.3, 0.3), 0.3) + 1.5 * lambda(x, Eigen::Vector2d(0.7, 0.7), 0.4);
#else
    assert(x.rows() == test_dimension);
    
    auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma)
    {
        return std::exp(- (x - mu).squaredNorm() / (sigma * sigma));
    };
    
    return lambda(x, Eigen::VectorXd::Constant(test_dimension, 0.5), 1.0);
#endif
}

void Core::computeRegression()
{
    regressor = std::make_shared<PreferenceRegressor>(X, D, use_MAP_hyperparameters);
}

void Core::updateSliderEnds()
{
    // If this is the first time...
    if (slider.get() == nullptr)
    {
#ifdef TWO_DIM
        slider = std::make_shared<Slider>(Utility::generateRandomVector(2), Utility::generateRandomVector(2), false);
#else
        slider = std::make_shared<Slider>(Utility::generateRandomVector(test_dimension), Utility::generateRandomVector(test_dimension), false);
#endif
        return;
    }

    const VectorXd x_1 = regressor->find_arg_max();
    const VectorXd x_2 = AcquisitionFunction::findNextPoint(*regressor);

    slider = std::make_shared<Slider>(x_1, x_2, true);
}

VectorXd Core::computeParametersFromSlider(double value) const
{
    assert(slider.get() != nullptr);
    return slider->getValue(value);
}
