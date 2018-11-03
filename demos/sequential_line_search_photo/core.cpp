#include "core.h"
#include <iostream>
#include <cmath>
#include "mainwindow.h"
#include "utility.h"
#include "acquisition-function.h"
#include "preferenceregressor.h"
#include "sliderutility.h"

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
    X = MatrixXd::Zero(0, 0);
    D.clear();
    
    x_max = VectorXd::Zero(0);
    y_max = NAN;
}

VectorXd Core::findNextPoint() const
{
    return ExpectedImprovement::findNextPoint(*regressor);
}

void Core::proceedOptimization()
{
    // Add new preference data
    const VectorXd x = computeParametersFromSlider();
    
    addData(x, slider->orig_0, slider->orig_1);
    
    // Compute regression
    computeRegression();
    
    // Check the current best
    unsigned index;
    y_max = regressor->y.maxCoeff(&index);
    x_max = regressor->X.col(index);
    
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
}

void Core::computeRegression()
{
    regressor = std::make_shared<PreferenceRegressor>(X, D);
}

void Core::updateSliderEnds()
{
    // If this is the first time...
    if (x_max.rows() == 0)
    {
        slider = make_shared<Slider>(Utility::generateRandomVector(dim), Utility::generateRandomVector(dim), true);
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
