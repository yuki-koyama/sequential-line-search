#include <Eigen/Core>
#include <fstream>
#include <sequential-line-search/utils.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

Eigen::VectorXd sequential_line_search::utils::GenerateRandomVector(unsigned n)
{
    return 0.5 * (Eigen::VectorXd::Random(n) + Eigen::VectorXd::Ones(n));
}

void sequential_line_search::utils::ExportMatrixToCsv(const std::string& file_path, const Eigen::MatrixXd& X)
{
    std::ofstream   file(file_path);
    Eigen::IOFormat format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",");
    file << X.format(format);
}
