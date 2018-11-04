#ifndef utils_h
#define utils_h

#include <cmath>
#include <cassert>
#include <string>
#include <Eigen/Core>
#include <Eigen/LU>

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
            return - exp_d / (exp_d_one * exp_d_one);
        }
        
        inline double sigmoid(double d)
        {
            return 1.0 / (1.0 + std::exp(- d));
        }
        
        inline double gauss(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::MatrixXd& Sigma_inv, const double Sigma_det)
        {
            const unsigned n = x.rows();
            return (1.0 / (pow(2.0 * M_PI, 0.5 * n) * std::sqrt(Sigma_det))) * std::exp(- 0.5 * (x - mu).transpose() * Sigma_inv * (x - mu));
        }
        
        inline double gauss(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::MatrixXd& Sigma)
        {
            const unsigned n = x.rows();
            return (1.0 / (pow(2.0 * M_PI, 0.5 * n) * std::sqrt(Sigma.determinant()))) * std::exp(- 0.5 * (x - mu).transpose() * Sigma.inverse() * (x - mu));
        }
        
        ////////////////////////////////////////////////
        // Gaussian Processes
        ////////////////////////////////////////////////
        
        inline double ARD_squared_exponential_kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const double a, const Eigen::VectorXd& r)
        {
            const unsigned D = x1.rows();
            
            double sum = 0.0;
            for (unsigned i = 0; i < D; ++ i)
            {
                sum += (x1(i) - x2(i)) * (x1(i) - x2(i)) / (r(i) * r(i));
            }
            
            return a * std::exp(- 0.5 * sum);
        }
        
        inline double ARD_squared_exponential_kernel_derivative_a(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const double /*a*/, const Eigen::VectorXd& r)
        {
            const unsigned D = x1.rows();
            
            double sum = 0.0;
            for (unsigned i = 0; i < D; ++ i)
            {
                sum += (x1(i) - x2(i)) * (x1(i) - x2(i)) / (r(i) * r(i));
            }
            
            return std::exp(- 0.5 * sum);
        }
        
        inline Eigen::VectorXd ARD_squared_exponential_kernel_derivative_r(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const double a, const Eigen::VectorXd& r)
        {
            Eigen::VectorXd deriv(r.rows());
            for (unsigned i = 0; i < r.rows(); ++ i)
            {
                deriv(i) = (x1(i) - x2(i)) * (x1(i) - x2(i)) / (r(i) * r(i) * r(i));
            }
            deriv *= ARD_squared_exponential_kernel(x1, x2, a, r);
            return deriv;
        }
        
        ////////////////////////////////////////////////
        // Bradley-Terry Model
        ////////////////////////////////////////////////
        
        // Probability that f_1 > f_2 (f_1 is preferable to f_2)
        inline double BT(double f_1, double f_2, double scale = 1.0)
        {
            return sigmoid((f_1 - f_2) / scale);
        }
        
        inline double derivative_BT_f_1(double f_1, double f_2, double scale = 1.0)
        {
            return (1.0 / scale) * derivative_sigmoid((f_1 - f_2) / scale);
        }
        
        inline double derivative_BT_f_2(double f_1, double f_2, double scale = 1.0)
        {
            return - (1.0 / scale) * derivative_sigmoid((f_1 - f_2) / scale);
        }
        
        ////////////////////////////////////////////////
        // Bradley-Terry-Luce Model
        ////////////////////////////////////////////////
        
        inline double BTL(Eigen::VectorXd f, double scale = 1.0)
        {
            const unsigned dim = f.rows();
            double sum = 0.0;
            for (unsigned i = 0; i < dim; ++ i)
            {
                sum += std::exp(f(i) / scale);
            }
            return std::exp(f(0) / scale) / sum;
        }
        
        inline Eigen::VectorXd derivative_BTL(const Eigen::VectorXd& f, double scale = 1.0)
        {
            const unsigned dim = f.rows();
            const double   btl = BTL(f, scale);
            const double   tmp = - btl * btl / scale;
            
            Eigen::VectorXd d(dim);
            
            double sum = 0.0;
            for (unsigned i = 1; i < dim; ++ i)
            {
                sum += std::exp((f(i) - f(0)) / scale);
            }
            d(0) = - sum;
            
            for (unsigned i = 1; i < dim; ++ i)
            {
                d(i) = std::exp((f(i) - f(0)) / scale);
            }
            
            return tmp * d;
        }
        
        ////////////////////////////////////////////////
        // Normal distribution
        ////////////////////////////////////////////////
        
        // N(x; mu, sigma^2)
        inline double normal(double x, double mu, double sigma_squared)
        {
            const double x_mu            = x - mu;
            const double sigma_squared_2 = sigma_squared * 2.0;
            return (1.0 / std::sqrt(M_PI * sigma_squared_2)) * std::exp(- (x_mu * x_mu) / sigma_squared_2);
        }
        
        ////////////////////////////////////////////////
        // Log-normal distribution
        ////////////////////////////////////////////////
        
        // LN(x; mu, sigma^2)
        inline double log_normal(double x, double mu, double sigma_squared)
        {
            assert(x > 0);
            const double log_x    = std::log(x);
            const double log_x_mu = log_x - mu;
            return 1.0 / (x * std::sqrt(2.0 * M_PI * sigma_squared)) * std::exp(- log_x_mu * log_x_mu / (2.0 * sigma_squared));
        }
        
        ////////////////////////////////////////////////
        // File IO
        ////////////////////////////////////////////////
        
        void exportMatrixToCsv(const std::string& filePath, const Eigen::MatrixXd& X);
        
        ////////////////////////////////////////////////
        // Clock
        ////////////////////////////////////////////////
        
        std::string getCurrentTimeInString();
    }
}

#endif // utils_h
