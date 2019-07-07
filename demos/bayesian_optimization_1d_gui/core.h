#ifndef CORE_H
#define CORE_H

#include <Eigen/Core>
#include <memory>

namespace sequential_line_search
{
    class GaussianProcessRegressor;
}

class Core
{
public:
    Core();

    static Core& getInstance()
    {
        static Core core;
        return core;
    }

    std::shared_ptr<sequential_line_search::GaussianProcessRegressor> regressor;

    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    double evaluateObjectiveFunction(const Eigen::VectorXd& x) const;

    bool   show_slider_value;
    double x_slider;
    double y_slider;

    // For optimization
    void            proceedOptimization();
    Eigen::VectorXd x_max;
    double          y_max;

    // For regression
    void addData(const Eigen::VectorXd& x, double y);
    void computeRegression();
};

#endif // CORE_H
