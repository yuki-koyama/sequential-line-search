#ifndef GAUSSIAN_PROCESS_REGRESSOR_H
#define GAUSSIAN_PROCESS_REGRESSOR_H

#include <Eigen/Core>
#include <sequential-line-search/regressor.hpp>

namespace sequential_line_search
{
    class GaussianProcessRegressor : public Regressor
    {
    public:
        /// \details Hyperparameters will be set via MAP estimation.
        GaussianProcessRegressor(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

        /// \details Specified hyperparameters will be used.
        GaussianProcessRegressor(const Eigen::MatrixXd& X,
                                 const Eigen::VectorXd& y,
                                 double                 a,
                                 double                 b,
                                 const Eigen::VectorXd& r);

        double PredictMu(const Eigen::VectorXd& x) const override;
        double PredictSigma(const Eigen::VectorXd& x) const override;

        Eigen::VectorXd PredictMuDerivative(const Eigen::VectorXd& x) const override;
        Eigen::VectorXd PredictSigmaDerivative(const Eigen::VectorXd& x) const override;

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
        double                 getb() const override { return b; }

        Eigen::VectorXd GetKernelHyperparams() const override
        {
            return (Eigen::VectorXd(r.size() + 1) << a, r).finished();
        }

    private:
        void PerformMapEstimation();
    };
} // namespace sequential_line_search

#endif // GAUSSIAN_PROCESS_REGRESSOR_H
