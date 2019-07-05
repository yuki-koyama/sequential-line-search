#include "core.h"
#include <cmath>
#include <iostream>
#include <sequential-line-search/sequential-line-search.h>

// #define TWO_DIM

using namespace sequential_line_search;

namespace
{
    constexpr unsigned test_dimension = 8;
}

using Eigen::MatrixXd;
using Eigen::VectorXd;

Core::Core() : use_MAP_hyperparameters(false)
{
    data.X = MatrixXd::Zero(0, 0);
    data.D.clear();

    x_max = VectorXd::Zero(0);
    y_max = NAN;
}

void Core::proceedOptimization(double slider_position)
{
    // Add new preference data
    const VectorXd x = computeParametersFromSlider(slider_position);
    data.AddNewPoints(x, {slider->orig_0, slider->orig_1});

    // Compute regression
    computeRegression();

    // Check the current best
    x_max = regressor->find_arg_max();
    y_max = regressor->estimate_y(x_max);

    // Update slider ends
    updateSliderEnds();
}

double Core::evaluateObjectiveFunction(const Eigen::VectorXd& x) const
{
#ifdef TWO_DIM
    assert(x.rows() == 2);

    auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma) {
        return std::exp(-(x - mu).squaredNorm() / (sigma * sigma));
    };

    return lambda(x, Eigen::Vector2d(0.3, 0.3), 0.3) + 1.5 * lambda(x, Eigen::Vector2d(0.7, 0.7), 0.4);
#else
    assert(x.rows() == test_dimension);

    auto lambda = [](const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const double sigma) {
        return std::exp(-(x - mu).squaredNorm() / (sigma * sigma));
    };

    return lambda(x, Eigen::VectorXd::Constant(test_dimension, 0.5), 1.0);
#endif
}

void Core::computeRegression()
{
    regressor = std::make_shared<PreferenceRegressor>(data.X, data.D, use_MAP_hyperparameters);
}

void Core::updateSliderEnds()
{
    // If this is the first time...
    if (slider.get() == nullptr)
    {
#ifdef TWO_DIM
        slider = std::make_shared<Slider>(utils::generateRandomVector(2), utils::generateRandomVector(2), false);
#else
        slider = std::make_shared<Slider>(
            utils::generateRandomVector(test_dimension), utils::generateRandomVector(test_dimension), false);
#endif
        return;
    }

    const VectorXd x_1 = regressor->find_arg_max();
    const VectorXd x_2 = acquisition_function::FindNextPoint(*regressor);

    slider = std::make_shared<Slider>(x_1, x_2, true);
}

VectorXd Core::computeParametersFromSlider(double value) const
{
    assert(slider.get() != nullptr);
    return slider->getValue(value);
}
