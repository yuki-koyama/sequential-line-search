#ifndef PREFERENCE_REGRESSOR_H
#define PREFERENCE_REGRESSOR_H

#include <Eigen/Core>
#include <sequential-line-search/preference.h>
#include <sequential-line-search/regressor.h>
#include <string>
#include <utility>
#include <vector>

namespace sequential_line_search
{
    /// \brief Class for performing regression based on preference data.
    /// \details See [Chu+, ICML 2005; Brochu+, NIPS 2007].
    class PreferenceRegressor : public Regressor
    {
    public:
        PreferenceRegressor(const Eigen::MatrixXd&         X,
                            const std::vector<Preference>& D,
                            const Eigen::VectorXd&         w                       = Eigen::VectorXd(),
                            const bool                     use_MAP_hyperparameters = false,
                            const double                   default_a               = 0.500,
                            const double                   default_r               = 0.500,
                            const double                   default_b               = 0.005,
                            const double                   variance                = 0.250,
                            const double                   btl_scale               = 0.010);

        double estimate_y(const Eigen::VectorXd& x) const override;
        double estimate_s(const Eigen::VectorXd& x) const override;

        const bool use_MAP_hyperparameters;

        /// \brief Find the data point that is likely to have the largest value from the so-far observed data points.
        Eigen::VectorXd find_arg_max();

        // Data
        Eigen::MatrixXd         X;
        std::vector<Preference> D;
        Eigen::VectorXd
            w; ///< Weights for calculating the scales in the BTL model (default = ones), used in crowdsourcing settings

        // Derived by MAP estimation
        Eigen::VectorXd y;
        double          a; ///< ARD hyperparameter about signal level.
        double          b; ///< ARD hyperparameter about noise level.
        Eigen::VectorXd r; ///< ARD hyperparameter about length scales.

        // Can be derived after MAP estimation
        Eigen::MatrixXd C;
        Eigen::MatrixXd C_inv;

        // IO
        void dampData(const std::string& dirPath) const;

        // Getter
        const Eigen::MatrixXd& getX() const override { return X; }
        const Eigen::VectorXd& gety() const override { return y; }
        double                 geta() const override { return a; }
        double                 getb() const override { return b; }
        const Eigen::VectorXd& getr() const override { return r; }

        // Default hyperparameters; when MAP is enabled, they are used as initial guesses.
        const double m_default_a;
        const double m_default_r;
        const double m_default_b;

        /// \brief Variance of the prior distribution. Used only when MAP is enabled.
        const double m_variance;

        /// \brief Scale parameter in the BTL model
        const double m_btl_scale;

    private:
        void compute_MAP(const PreferenceRegressor* = nullptr);
    };
} // namespace sequential_line_search

#endif // PREFERENCE_REGRESSOR_H
