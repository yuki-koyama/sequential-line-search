#ifndef GAUSSIANPROCESSREGRESSOR_H
#define GAUSSIANPROCESSREGRESSOR_H

#include <Eigen/Core>
#include <sequential-line-search/regressor.h>

namespace sequential_line_search
{
    class GaussianProcessRegressor : public Regressor
    {
    public:
        /// \details Hyperparameters will be set via MAP estimation.
        GaussianProcessRegressor(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

        /// \details Specified hyperparameters will be used.
        GaussianProcessRegressor(
            const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double a, double b, const Eigen::VectorXd& r);

        double estimate_y(const Eigen::VectorXd& x) const override;
        double estimate_s(const Eigen::VectorXd& x) const override;

        /// \brief Data points.
        Eigen::MatrixXd X;

        /// \brief Values on data points.
        Eigen::VectorXd y;

        /// \brief A hyperparameter about signal level of ARD.
        /// \details Derived from MAP or specified directly.
        double a;

        /// \brief A hyperparameter about noise level of ARD.
        /// \details Derived from MAP or specified directly.
        double b;

        /// \brief A hyperparameter about length scales of ARD.
        /// \details Derived from MAP or specified directly.
        Eigen::VectorXd r;

        // Can be derived after MAP
        Eigen::MatrixXd C;
        Eigen::MatrixXd C_inv;

        // Getter
        const Eigen::MatrixXd& getX() const override { return X; }
        const Eigen::VectorXd& gety() const override { return y; }
        double                 geta() const override { return a; }
        double                 getb() const override { return b; }
        const Eigen::VectorXd& getr() const override { return r; }

    private:
        void compute_MAP();
    };
} // namespace sequential_line_search

#endif // GAUSSIANPROCESSREGRESSOR_H
