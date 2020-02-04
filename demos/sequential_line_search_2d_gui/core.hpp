#ifndef CORE_H
#define CORE_H

#include <Eigen/Core>
#include <memory>
#include <vector>

class MainWindow;
namespace sequential_line_search
{
    class SequentialLineSearchOptimizer;
}

class Core
{
public:
    Core() {}

    static Core& getInstance()
    {
        static Core core;
        return core;
    }

    std::shared_ptr<sequential_line_search::SequentialLineSearchOptimizer> optimizer;

    MainWindow* mainWindow;

    // x in [0, 1]^2
    double evaluateObjectiveFunction(const Eigen::VectorXd& x) const;
};

#endif // CORE_H
