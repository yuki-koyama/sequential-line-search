#ifndef SEQUENTIAL_LINE_SEARCH_PREFERENCE_REGRESSOR_HPP
#define SEQUENTIAL_LINE_SEARCH_PREFERENCE_REGRESSOR_HPP

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <sequential-line-search/preference.hpp>
#include <sequential-line-search/regressor.hpp>
#include <string>
#include <utility>
#include <vector>

namespace sequential_line_search
{
    /// \brief Class for performing regression based on preference data.
    ///
    /// \details See [Chu+, ICML 2005; Brochu+, NIPS 2007].
    class PreferenceRegressor : public Regressor
    {
    public:
        PreferenceRegressor(const Eigen::MatrixXd&         X,
                            const std::vector<Preference>& D,
                            const bool                     use_map_hyperparams          = false,
                            const double                   default_kernel_signal_var    = 0.500,
                            const double                   default_kernel_length_scale  = 0.500,
                            const double                   default_noise_level          = 0.005,
                            const double                   kernel_hyperparams_prior_var = 0.250,
                            const double                   btl_scale                    = 0.010,
                            const unsigned                 num_map_estimation_iters     = 100,
                            const KernelType               kernel_type = KernelType::ArdMatern52Kernel);

        double PredictMu(const Eigen::VectorXd& x) const override;
        double PredictSigma(const Eigen::VectorXd& x) const override;

        Eigen::VectorXd PredictMuDerivative(const Eigen::VectorXd& x) const override;
        Eigen::VectorXd PredictSigmaDerivative(const Eigen::VectorXd& x) const override;

        const bool m_use_map_hyperparams;

        /// \brief Find the data point that is likely to have the largest value from the so-far observed data points.
        Eigen::VectorXd FindArgMax() const;

        // Data
        Eigen::MatrixXd         m_X;
        std::vector<Preference> m_D;

        /// \brief Noise level hyperparameter
        ///
        /// \details This value is either derived by the MAP estimation or copied from the default values
        double m_noise_hyperparam;

        /// \brief Kernel hyperparameters
        ///
        /// \details These values are either derived by the MAP estimation or copied from the default values
        Eigen::VectorXd m_kernel_hyperparams;

        /// \brief Kernel matrix calculated in the MAP estimation procedure.
        Eigen::MatrixXd m_K;

        /// \brief Kernel matrix stored as a Cholesky-decomposed form.
        Eigen::LLT<Eigen::MatrixXd> m_K_llt;

        // IO
        void DampData(const std::string& dir_path, const std::string& prefix = "") const;

        // Getter
        const Eigen::MatrixXd& GetLargeX() const override { return m_X; }
        const Eigen::VectorXd& GetSmallY() const override { return m_y; }

        const Eigen::VectorXd& GetKernelHyperparams() const override { return m_kernel_hyperparams; }
        double                 GetNoiseHyperparam() const override { return m_noise_hyperparam; }

        // Default hyperparameters; when MAP is enabled, they are used as initial guesses.
        const double m_default_kernel_signal_var;
        const double m_default_kernel_length_scale;
        const double m_default_noise_level;

        /// \brief Variance of the prior distribution. Used only when MAP is enabled.
        const double m_kernel_hyperparams_prior_var;

        /// \brief Scale parameter in the BTL model
        const double m_btl_scale;

    private:
        /// \brief Goodness values derived by the MAP estimation.
        Eigen::VectorXd m_y;

        void PerformMapEstimation(const unsigned num_iters);
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_PREFERENCE_REGRESSOR_HPP
