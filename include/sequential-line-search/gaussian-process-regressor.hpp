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
                                 const Eigen::VectorXd& kernel_hyperparams,
                                 double                 noise_hyperparam);

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
        double m_noise_hyperparam;

        /// \brief A hyperparameter about length scales of ARD.
        /// \details Derived from MAP or specified directly.
        Eigen::VectorXd r;

        // Can be derived after MAP
        Eigen::MatrixXd C;
        Eigen::MatrixXd C_inv;

        // Getter
        const Eigen::MatrixXd& getX() const override { return X; }
        const Eigen::VectorXd& gety() const override { return y; }

        Eigen::VectorXd GetKernelHyperparams() const override
        {
            return (Eigen::VectorXd(r.size() + 1) << a, r).finished();
        }

        double GetNoiseHyperparam() const override { return m_noise_hyperparam; }

    private:
        void PerformMapEstimation();
    };
} // namespace sequential_line_search

#endif // GAUSSIAN_PROCESS_REGRESSOR_H
