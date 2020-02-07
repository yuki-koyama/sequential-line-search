#ifndef SEQUENTIAL_LINE_SEARCH_UTILS_HPP
#define SEQUENTIAL_LINE_SEARCH_UTILS_HPP

#include <Eigen/Core>
#include <Eigen/LU>
#include <cassert>
#include <cmath>
#include <string>

namespace sequential_line_search
{
    namespace utils
    {
        ////////////////////////////////////////////////
        // Random
        ////////////////////////////////////////////////

        // Uniform sampling from [0, 1]^n
        Eigen::VectorXd GenerateRandomVector(unsigned n);

        ////////////////////////////////////////////////
        // Bradley-Terry-Luce Model
        ////////////////////////////////////////////////

        inline double CalcBtl(const Eigen::VectorXd& f, double scale = 1.0)
        {
            const auto exp_rep = ((1.0 / scale) * f).array().exp();
            return exp_rep(0) / exp_rep.sum();
        }

        inline Eigen::VectorXd CalcBtlDerivative(const Eigen::VectorXd& f, double scale = 1.0)
        {
            const unsigned dim = f.rows();
            const double   btl = CalcBtl(f, scale);
            const double   tmp = -btl * btl / scale;

            Eigen::VectorXd d(dim);

            double sum = 0.0;
            for (unsigned i = 1; i < dim; ++i)
            {
                sum += std::exp((f(i) - f(0)) / scale);
            }
            d(0) = -sum;

            for (unsigned i = 1; i < dim; ++i)
            {
                d(i) = std::exp((f(i) - f(0)) / scale);
            }

            return tmp * d;
        }

        ////////////////////////////////////////////////
        // File IO
        ////////////////////////////////////////////////

        void ExportMatrixToCsv(const std::string& file_path, const Eigen::MatrixXd& X);
    } // namespace utils
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_UTILS_HPP
