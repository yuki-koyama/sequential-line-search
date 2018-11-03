#ifndef REGRESSOR_H
#define REGRESSOR_H

#include <Eigen/Core>

class Regressor
{
public:
    virtual ~Regressor() {}
    
    virtual double estimate_y(const Eigen::VectorXd& x) const = 0;
    virtual double estimate_s(const Eigen::VectorXd& x) const = 0;

    virtual const Eigen::MatrixXd& getX() const = 0;
    virtual const Eigen::VectorXd& gety() const = 0;
    virtual double geta() const = 0;
    virtual double getb() const = 0;
    virtual const Eigen::VectorXd& getr() const = 0;

    static Eigen::MatrixXd calc_C(const Eigen::MatrixXd& X, const double a, const double b, const Eigen::VectorXd& r);
    static Eigen::MatrixXd calc_C_grad_a(const Eigen::MatrixXd& X, const double a, const double /*b*/, const Eigen::VectorXd& r);
    static Eigen::MatrixXd calc_C_grad_b(const Eigen::MatrixXd& X, const double /*a*/, const double /*b*/, const Eigen::VectorXd& /*r*/);
    static Eigen::MatrixXd calc_C_grad_r_i(const Eigen::MatrixXd& X, const double a, const double /*b*/, const Eigen::VectorXd& r, const unsigned index);
    static Eigen::VectorXd calc_k(const Eigen::VectorXd& x, const Eigen::MatrixXd& X, const double a, const double /*b*/, const Eigen::VectorXd& r);
};

#endif // REGRESSOR_H
