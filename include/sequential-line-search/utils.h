#ifndef utils_h
#define utils_h

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
        Eigen::VectorXd generateRandomVector(unsigned n);

        ////////////////////////////////////////////////
        // Math
        ////////////////////////////////////////////////

        inline double derivative_sigmoid(double d)
        {
            const double exp_d     = std::exp(d);
            const double exp_d_one = exp_d + 1.0;
            return -exp_d / (exp_d_one * exp_d_one);
        }

        inline double sigmoid(double d) { return 1.0 / (1.0 + std::exp(-d)); }

        inline double gauss(const Eigen::VectorXd& x,
                            const Eigen::VectorXd& mu,
                            const Eigen::MatrixXd& Sigma_inv,
                            const double           Sigma_det)
        {
            const unsigned n = x.rows();
            return (1.0 / (pow(2.0 * M_PI, 0.5 * n) * std::sqrt(Sigma_det))) *
                   std::exp(-0.5 * (x - mu).transpose() * Sigma_inv * (x - mu));
        }

        inline double gauss(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::MatrixXd& Sigma)
        {
            const unsigned n = x.rows();
            return (1.0 / (pow(2.0 * M_PI, 0.5 * n) * std::sqrt(Sigma.determinant()))) *
                   std::exp(-0.5 * (x - mu).transpose() * Sigma.inverse() * (x - mu));
        }

        ////////////////////////////////////////////////
        // Bradley-Terry Model
        ////////////////////////////////////////////////

        // Probability that f_1 > f_2 (f_1 is preferable to f_2)
        inline double BT(double f_1, double f_2, double scale = 1.0) { return sigmoid((f_1 - f_2) / scale); }

        inline double derivative_BT_f_1(double f_1, double f_2, double scale = 1.0)
        {
            return (1.0 / scale) * derivative_sigmoid((f_1 - f_2) / scale);
        }

        inline double derivative_BT_f_2(double f_1, double f_2, double scale = 1.0)
        {
            return -(1.0 / scale) * derivative_sigmoid((f_1 - f_2) / scale);
        }

        ////////////////////////////////////////////////
        // Bradley-Terry-Luce Model
        ////////////////////////////////////////////////

        inline double CalcBtl(const Eigen::VectorXd& f, double scale = 1.0)
        {
            const unsigned dim = f.rows();
            double         sum = 0.0;
            for (unsigned i = 0; i < dim; ++i)
            {
                sum += std::exp(f(i) / scale);
            }
            return std::exp(f(0) / scale) / sum;
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

        void exportMatrixToCsv(const std::string& filePath, const Eigen::MatrixXd& X);
    } // namespace utils
} // namespace sequential_line_search

#endif // utils_h
