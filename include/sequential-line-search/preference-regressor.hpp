#ifndef PREFERENCE_REGRESSOR_H
#define PREFERENCE_REGRESSOR_H

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
                            const bool                     use_map_hyperparameters = false,
                            const double                   default_a               = 0.500,
                            const double                   default_r               = 0.500,
                            const double                   default_b               = 0.005,
                            const double                   variance                = 0.250,
                            const double                   btl_scale               = 0.010);

        double PredictMu(const Eigen::VectorXd& x) const override;
        double PredictSigma(const Eigen::VectorXd& x) const override;

        Eigen::VectorXd PredictMuDerivative(const Eigen::VectorXd& x) const override;
        Eigen::VectorXd PredictSigmaDerivative(const Eigen::VectorXd& x) const override;

        const bool m_use_map_hyperparameters;

        /// \brief Find the data point that is likely to have the largest value from the so-far observed data points.
        Eigen::VectorXd FindArgMax() const;

        // Data
        Eigen::MatrixXd         m_X;
        std::vector<Preference> m_D;

        // Hyperparameters either derived by the MAP estimation or copied from the default values
        double          a; ///< ARD hyperparameter about signal level.
        double          b; ///< ARD hyperparameter about noise level.
        Eigen::VectorXd r; ///< ARD hyperparameter about length scales.

        /// \brief Kernel matrix calculated in the MAP estimation procedure.
        Eigen::MatrixXd m_K;

        /// \brief Kernel matrix stored as a Cholesky-decomposed form.
        Eigen::LLT<Eigen::MatrixXd> m_K_llt;

        // IO
        void DampData(const std::string& dir_path, const std::string& prefix = "") const;

        // Getter
        const Eigen::MatrixXd& getX() const override { return m_X; }
        const Eigen::VectorXd& gety() const override { return m_y; }
        double                 getb() const override { return b; }

        Eigen::VectorXd GetKernelHyperparams() const override
        {
            return (Eigen::VectorXd(r.size() + 1) << a, r).finished();
        }

        // Default hyperparameters; when MAP is enabled, they are used as initial guesses.
        const double m_default_a;
        const double m_default_r;
        const double m_default_b;

        /// \brief Variance of the prior distribution. Used only when MAP is enabled.
        const double m_variance;

        /// \brief Scale parameter in the BTL model
        const double m_btl_scale;

    private:
        /// \brief Goodness values derived by the MAP estimation.
        Eigen::VectorXd m_y;

        void PerformMapEstimation();
    };
} // namespace sequential_line_search

#endif // PREFERENCE_REGRESSOR_H
