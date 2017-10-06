#ifndef CORE_H
#define CORE_H

#include <vector>
#include <memory>
#include <Eigen/Core>
#include "preference.h"
#include "slider.h"

class PreferenceRegressor;
class MainWindow;

class Core
{
public:
    Core();

    static Core& getInstance() {
        static Core core;
        return core;
    }

    std::shared_ptr<PreferenceRegressor> regressor;

    MainWindow* mainWindow;

    bool use_MAP_hyperparameters;

    Eigen::MatrixXd         X;
    std::vector<Preference> D;

    double evaluateObjectiveFunction(const Eigen::VectorXd &x) const;

    void clear()
    {
        X = Eigen::MatrixXd::Zero(0, 0);
        D.clear();
        x_max = Eigen::VectorXd::Zero(0);
        regressor = nullptr;
        slider    = nullptr;
    }

    // For optimization
    void proceedOptimization();
    Eigen::VectorXd x_max;
    double          y_max;

    // For regression
    void addData(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2);
    void addData(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd &x3);
    void addData(const std::vector<Eigen::VectorXd>& xs);
    void computeRegression();

    // For slider management
    void updateSliderEnds();
    std::shared_ptr<Slider> slider;
    Eigen::VectorXd computeParametersFromSlider(int sliderValue, int minValue, int maxValue) const;
    Eigen::VectorXd computeParametersFromSlider(double value) const;
};

#endif // CORE_H
