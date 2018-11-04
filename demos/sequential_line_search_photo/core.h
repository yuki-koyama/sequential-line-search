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
class MainWindow;

class Core
{
public:
    Core();
    
    static Core& getInstance() {
        static Core core;
        return core;
    }
    
    std::shared_ptr<sequential_line_search::PreferenceRegressor> regressor;
    
    const unsigned dim;
    
    MainWindow* mainWindow;
    
    sequential_line_search::Data data;
    
    // For optimization
    void proceedOptimization();
    Eigen::VectorXd findNextPoint() const;
    Eigen::VectorXd x_max;
    double          y_max;
    
    // For regression
    void computeRegression();
    
    // For slider management
    void updateSliderEnds();
    std::shared_ptr<sequential_line_search::Slider> slider;
    Eigen::VectorXd computeParametersFromSlider(int sliderValue, int minValue, int maxValue) const;
    Eigen::VectorXd computeParametersFromSlider(double value) const;
    Eigen::VectorXd computeParametersFromSlider() const;
};

#endif // CORE_H
