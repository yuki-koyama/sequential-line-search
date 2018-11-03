#ifndef CORE_H
#define CORE_H

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <sequential-line-search/sequential-line-search.h>

class PreferenceRegressor;

class Core
{
public:
    Core();

    std::shared_ptr<PreferenceRegressor> regressor;

    bool use_MAP_hyperparameters;

    Eigen::MatrixXd                                 X;
    std::vector<sequential_line_search::Preference> D;

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
    void proceedOptimization(double slider_position);
    Eigen::VectorXd x_max;
    double          y_max;

    // For regression
    void addData(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2);
    void addData(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd &x3);
    void addData(const std::vector<Eigen::VectorXd>& xs);
    void computeRegression();

    // For slider management
    void updateSliderEnds();
    std::shared_ptr<sequential_line_search::Slider> slider;
    Eigen::VectorXd computeParametersFromSlider(double value) const;
};

#endif // CORE_H
