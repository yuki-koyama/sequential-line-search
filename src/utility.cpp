#include <sequential-line-search/utility.h>
#include <random>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <Eigen/Core>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace
{
    std::random_device seed;
    std::default_random_engine gen(seed());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<> normal_dist(0.0, 1.0);
}

namespace Utility
{
    Eigen::VectorXd generateRandomVector(unsigned n)
    {
        Eigen::VectorXd x(n);
        for (unsigned i = 0; i < n; ++ i)
        {
            x(i) = uniform_dist(gen);
        }
        return x;
    }
    
    double generateUniformReal()
    {
        return uniform_dist(gen);
    }
    
    double generateStandardNormal()
    {
        return normal_dist(gen);
    }
    
    void exportMatrixToCsv(const std::string& filePath, const Eigen::MatrixXd& X)
    {
        std::ofstream ofs(filePath);
        for (unsigned i = 0; i < X.rows(); ++ i)
        {
            for (unsigned j = 0; j < X.cols(); ++ j)
            {
                ofs << X(i, j);
                if (j + 1 != X.cols()) ofs << ",";
            }
            ofs << std::endl;
        }
    }
    
    std::string getCurrentTimeInString()
    {
        const std::time_t t = std::time(nullptr);
        std::stringstream s; s << std::put_time(std::localtime(&t), "%Y%m%d%H%M%S");
        return s.str();
    }
}
