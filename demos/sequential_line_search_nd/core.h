#ifndef CORE_H
#define CORE_H

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <sequential-line-search/sequential-line-search.h>

namespace sequential_line_search
{
    class PreferenceRegressor;
}

class Core
{
public:
    Core();
    
    std::shared_ptr<sequential_line_search::PreferenceRegressor> regressor;
    
    bool use_MAP_hyperparameters;
    
    sequential_line_search::Data data;
    
    double evaluateObjectiveFunction(const Eigen::VectorXd &x) const;
    
    void clear()
    {
        data.X = Eigen::MatrixXd::Zero(0, 0);
        data.D.clear();
        x_max = Eigen::VectorXd::Zero(0);
        regressor = nullptr;
        slider    = nullptr;
    }
    
    // For optimization
    void proceedOptimization(double slider_position);
    Eigen::VectorXd x_max;
    double          y_max;
    
    // For regression
    void computeRegression();
    
    // For slider management
    void updateSliderEnds();
    std::shared_ptr<sequential_line_search::Slider> slider;
    Eigen::VectorXd computeParametersFromSlider(double value) const;
};

#endif // CORE_H
