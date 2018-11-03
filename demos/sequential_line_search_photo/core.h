#ifndef CORE_H
#define CORE_H

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <sequential-line-search/sequential-line-search.h>

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

    const unsigned dim;

    MainWindow* mainWindow;

    Eigen::MatrixXd         X;
    std::vector<Preference> D;

    // For optimization
    void proceedOptimization();
    Eigen::VectorXd findNextPoint() const;
    Eigen::VectorXd x_max;
    double          y_max;

    // For regression
    void addData(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2);
    void addData(const Eigen::VectorXd& x1, const Eigen::VectorXd &x2, const Eigen::VectorXd &x3);
    void addData(const std::vector<Eigen::VectorXd>& xs);
    void computeRegression();
    void cleanData(double epsilon = 1e-03);

    // For slider management
    void updateSliderEnds();
    std::shared_ptr<sequential_line_search::Slider> slider;
    Eigen::VectorXd computeParametersFromSlider(int sliderValue, int minValue, int maxValue) const;
    Eigen::VectorXd computeParametersFromSlider(double value) const;
    Eigen::VectorXd computeParametersFromSlider() const;
};

#endif // CORE_H
