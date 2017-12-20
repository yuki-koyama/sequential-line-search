#ifndef PREFERENCEREGRESSOR_H
#define PREFERENCEREGRESSOR_H

#include <vector>
#include <string>
#include <utility>
#include <Eigen/Core>
#include "regressor.h"
#include "preference.h"

///////////////////////////////////////
// Chu+, ICML 2005; Brochu+, NIPS 2007
///////////////////////////////////////

class PreferenceRegressor : public Regressor
{
public:

    PreferenceRegressor(const Eigen::MatrixXd &X, const std::vector<Preference>& D, bool use_MAP_hyperparameters = false);
    PreferenceRegressor(const Eigen::MatrixXd &X, const std::vector<Preference>& D, const Eigen::VectorXd& w, bool use_MAP_hyperparameters = false);
    PreferenceRegressor(const Eigen::MatrixXd &X, const std::vector<Preference>& D, const Eigen::VectorXd& w, bool use_MAP_hyperparameters, const PreferenceRegressor* previous);

    double estimate_y(const Eigen::VectorXd& x) const override;
    double estimate_s(const Eigen::VectorXd& x) const override;

    const bool use_MAP_hyperparameters;

    Eigen::VectorXd find_arg_max();

    // Data
    Eigen::MatrixXd         X;
    std::vector<Preference> D;
    Eigen::VectorXd         w; // Weights for calculating the scales in the BTL model (default = ones), used in crowdsourcing settings

    // Derived by MAP estimation
    Eigen::VectorXd         y;
    double                  a;
    double                  b;
    Eigen::VectorXd         r;

    // Can be derived after MAP estimation
    Eigen::MatrixXd         C;
    Eigen::MatrixXd         C_inv;

    // IO
    void dampData(const std::string& dirPath) const;

    // Getter
    const Eigen::MatrixXd& getX() const override { return X; }
    const Eigen::VectorXd& gety() const override { return y; }
    double geta() const override { return a; }
    double getb() const override { return b; }
    const Eigen::VectorXd& getr() const override { return r; }

    struct Params
    {
        Params() {}
        double a         = 0.500;
        double r         = 0.500;
        double b         = 0.005;
        double variance  = 0.250; // Used when hyperparameters are estimated via MAP
        double btl_scale = 0.010;
        static Params& getInstance() { static Params p; return p; }
    };

private:
    void compute_MAP(const PreferenceRegressor* = nullptr);
};

#endif // PREFERENCEREGRESSOR_H
