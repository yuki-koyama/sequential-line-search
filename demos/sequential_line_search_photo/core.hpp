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
    Core();

    static Core& getInstance()
    {
        static Core core;
        return core;
    }

    const int dim;

    std::shared_ptr<sequential_line_search::SequentialLineSearchOptimizer> optimizer;

    MainWindow* mainWindow;
};

#endif // CORE_H
