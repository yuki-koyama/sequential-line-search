#ifndef CORE_H
#define CORE_H

#include <Eigen/Core>
#include <memory>

namespace sequential_line_search
{
    class PreferenceRegressor;
    class PreferenceDataManager;
} // namespace sequential_line_search

namespace mathtoolbox
{
    class DataNormalizer;
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

    std::shared_ptr<sequential_line_search::PreferenceRegressor>   m_regressor;
    std::shared_ptr<sequential_line_search::PreferenceDataManager> m_data;
    std::shared_ptr<mathtoolbox::DataNormalizer>                   m_normalizer;

    Eigen::VectorXd m_y;

    double evaluateObjectiveFunction(const Eigen::VectorXd& x) const;

    // For optimization
    void            proceedOptimization();
    Eigen::VectorXd m_x_max;
    double          m_y_max;

    // For regression
    void computeRegression();
};

#endif // CORE_H
