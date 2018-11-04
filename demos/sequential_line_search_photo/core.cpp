#include "core.h"
#include <iostream>
#include <cmath>
#include <sequential-line-search/sequential-line-search.h>
#include "mainwindow.h"

using namespace sequential_line_search;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::make_shared;

#ifdef SEQUENTIAL_LINE_SEARCH_PHOTO_DIM_SUBSET
#define PHOTO_DIM 2
#else
#define PHOTO_DIM 6
#endif

Core::Core() : dim(PHOTO_DIM)
{
    data.X = MatrixXd::Zero(0, 0);
    data.D.clear();
    
    x_max = VectorXd::Zero(0);
    y_max = NAN;
}

VectorXd Core::findNextPoint() const
{
    return acquisition_function::FindNextPoint(*regressor);
}

void Core::proceedOptimization()
{
    // Add new preference data
    const VectorXd x = computeParametersFromSlider();
    data.AddNewPoints(x, { slider->orig_0, slider->orig_1 });
    
    // Compute regression
    computeRegression();
    
    // Check the current best
    unsigned index;
    y_max = regressor->y.maxCoeff(&index);
    x_max = regressor->X.col(index);
    
    // Update slider ends
    updateSliderEnds();
}

void Core::computeRegression()
{
    regressor = std::make_shared<PreferenceRegressor>(data.X, data.D);
}

void Core::updateSliderEnds()
{
    // If this is the first time...
    if (x_max.rows() == 0)
    {
        slider = make_shared<Slider>(utils::generateRandomVector(dim), utils::generateRandomVector(dim), true);
        return;
    }
    
    const VectorXd x_1 = regressor->find_arg_max();
    const VectorXd x_2 = findNextPoint();
    
    slider = make_shared<Slider>(x_1, x_2, true);
}

VectorXd Core::computeParametersFromSlider(int sliderValue, int minValue, int maxValue) const
{
    const double t = static_cast<double>(sliderValue - minValue) / static_cast<double>(maxValue - minValue);
    return computeParametersFromSlider(t);
}

VectorXd Core::computeParametersFromSlider(double value) const
{
    return slider->end_0 * (1.0 - value) + slider->end_1 *  value;
}

VectorXd Core::computeParametersFromSlider() const
{
    return computeParametersFromSlider(mainWindow->obtainSliderPosition());
}
