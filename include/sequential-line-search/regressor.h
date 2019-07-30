#ifndef REGRESSOR_H
#define REGRESSOR_H

#include <Eigen/Core>

namespace sequential_line_search
{
    class Regressor
    {
    public:
        virtual ~Regressor() {}

        virtual double PredictMu(const Eigen::VectorXd& x) const    = 0;
        virtual double PredictSigma(const Eigen::VectorXd& x) const = 0;

        virtual Eigen::VectorXd PredictMuDerivative(const Eigen::VectorXd& x) const    = 0;
        virtual Eigen::VectorXd PredictSigmaDerivative(const Eigen::VectorXd& x) const = 0;

        virtual const Eigen::MatrixXd& getX() const = 0;
        virtual const Eigen::VectorXd& gety() const = 0;
        virtual double                 geta() const = 0;
        virtual double                 getb() const = 0;
        virtual const Eigen::VectorXd& getr() const = 0;

        Eigen::VectorXd PredictMaximumPointFromData() const;

        static Eigen::MatrixXd
        calc_C(const Eigen::MatrixXd& X, const double a, const double b, const Eigen::VectorXd& r);
        static Eigen::MatrixXd
        calc_C_grad_a(const Eigen::MatrixXd& X, const double a, const double /*b*/, const Eigen::VectorXd& r);
        static Eigen::MatrixXd
                               calc_C_grad_b(const Eigen::MatrixXd& X, const double /*a*/, const double /*b*/, const Eigen::VectorXd& /*r*/);
        static Eigen::MatrixXd calc_C_grad_r_i(const Eigen::MatrixXd& X,
                                               const double           a,
                                               const double /*b*/,
                                               const Eigen::VectorXd& r,
                                               const unsigned         index);
        static Eigen::VectorXd calc_k(const Eigen::VectorXd& x,
                                      const Eigen::MatrixXd& X,
                                      const double           a,
                                      const double /*b*/,
                                      const Eigen::VectorXd& r);

        // partial k / partial x
        static Eigen::MatrixXd CalcSmallKSmallXDerivative(const Eigen::VectorXd& x,
                                                          const Eigen::MatrixXd& X,
                                                          const double           a,
                                                          const double           b,
                                                          const Eigen::VectorXd& r);
    };
} // namespace sequential_line_search

#endif // REGRESSOR_H
