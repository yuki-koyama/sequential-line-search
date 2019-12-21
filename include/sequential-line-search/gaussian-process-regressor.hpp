#ifndef SEQUENTIAL_LINE_SEARCH_GAUSSIAN_PROCESS_REGRESSOR_HPP
#define SEQUENTIAL_LINE_SEARCH_GAUSSIAN_PROCESS_REGRESSOR_HPP

#include <Eigen/Core>
#include <sequential-line-search/regressor.hpp>

namespace sequential_line_search
{
    class GaussianProcessRegressor : public Regressor
    {
    public:
        /// \details Hyperparameters will be set via MAP estimation.
        GaussianProcessRegressor(const Eigen::MatrixXd& X,
                                 const Eigen::VectorXd& y,
                                 const KernelType       kernel_type = KernelType::ArdMatern52Kernel);

        /// \details Specified hyperparameters will be used.
        GaussianProcessRegressor(const Eigen::MatrixXd& X,
                                 const Eigen::VectorXd& y,
                                 const Eigen::VectorXd& kernel_hyperparams,
                                 double                 noise_hyperparam,
                                 const KernelType       kernel_type = KernelType::ArdMatern52Kernel);

        double PredictMu(const Eigen::VectorXd& x) const override;
        double PredictSigma(const Eigen::VectorXd& x) const override;

        Eigen::VectorXd PredictMuDerivative(const Eigen::VectorXd& x) const override;
        Eigen::VectorXd PredictSigmaDerivative(const Eigen::VectorXd& x) const override;

        // Can be derived after MAP
        Eigen::MatrixXd m_K_y;
        Eigen::MatrixXd m_K_y_inv;

        // Getter
        const Eigen::MatrixXd& GetLargeX() const override { return m_X; }
        const Eigen::VectorXd& GetSmallY() const override { return m_y; }

        const Eigen::VectorXd& GetKernelHyperparams() const override { return m_kernel_hyperparams; }
        double                 GetNoiseHyperparam() const override { return m_noise_hyperparam; }

    private:
        void PerformMapEstimation();

        /// \brief Data points.
        Eigen::MatrixXd m_X;

        /// \brief Values on data points.
        Eigen::VectorXd m_y;

        /// \brief Kernel hyperparameters
        ///
        /// \details Derived from MAP or specified directly.
        Eigen::VectorXd m_kernel_hyperparams;

        /// \brief A hyperparameter about noise level of ARD.
        ///
        /// \details Derived from MAP or specified directly.
        double m_noise_hyperparam;
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_GAUSSIAN_PROCESS_REGRESSOR_HPP
