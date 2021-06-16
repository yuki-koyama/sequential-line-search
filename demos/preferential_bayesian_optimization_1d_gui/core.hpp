#ifndef CORE_H
#define CORE_H

#include <Eigen/Core>
#include <memory>

namespace sequential_line_search
{
    class PreferentialBayesianOptimizer;
} // namespace sequential_line_search

namespace mathtoolbox
{
    class DataNormalizer;
}

class Core
{
public:
    Core();

    void reset();

    static Core& getInstance()
    {
        static Core core;
        return core;
    }

    std::shared_ptr<sequential_line_search::PreferentialBayesianOptimizer> m_optimizer;
    std::shared_ptr<mathtoolbox::DataNormalizer>                           m_normalizer;

    Eigen::VectorXd m_y;

    double evaluateObjectiveFunction(const Eigen::VectorXd& x) const;

    // For optimization
    void            proceedOptimization();
    Eigen::VectorXd m_x_max;
    double          m_y_max;
};

#endif // CORE_H
