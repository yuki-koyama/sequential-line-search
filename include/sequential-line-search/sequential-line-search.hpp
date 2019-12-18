#ifndef SEQUENTIAL_LINE_SEARCH_SEQUENTIAL_LINE_SEARCH_HPP
#define SEQUENTIAL_LINE_SEARCH_SEQUENTIAL_LINE_SEARCH_HPP

#include <Eigen/Core>
#include <memory>
#include <utility>

namespace sequential_line_search
{
    class PreferenceRegressor;
    class Slider;
    class PreferenceDataManager;

    std::pair<Eigen::VectorXd, Eigen::VectorXd> GenerateRandomSliderEnds(const int num_dims);
    std::pair<Eigen::VectorXd, Eigen::VectorXd> GenerateCenteredFixedLengthRandomSliderEnds(const int num_dims);

    /// \brief Optimizer class for performing sequential line search
    ///
    /// \details This class assumes that the search space is [0, 1]^{D}
    class SequentialLineSearchOptimizer
    {
    public:
        /// \brief Construct an optimizer instance.
        ///
        /// \param use_slider_enlargement When this is set true, slider spaces will be enlarged in post-processing. See
        /// [Koyama+17] for details.
        ///
        /// \param use_map_hyperparams When this is set true, the optimizer always perform the MAP estimation for the
        /// GPR kernel hyperparameters. When this is set false, the optimizer performs the MAP estimation only for
        /// goodness values.
        SequentialLineSearchOptimizer(const int  num_dims,
                                      const bool use_slider_enlargement = false,
                                      const bool use_map_hyperparams    = true,
                                      const std::function<std::pair<Eigen::VectorXd, Eigen::VectorXd>(const int)>&
                                          initial_slider_generator = GenerateRandomSliderEnds);

        /// \brief Specify (kernel and other) hyperparameter values.
        ///
        /// \details When the MAP estimation is enabled, the specified kernel hyperparameters will be used as the median
        /// of the prior distribution and the initial solution of the estimation. When the MAP estimation is not
        /// enabled, these values will be directly used.
        void SetHyperparams(const double kernel_signal_var            = 0.500,
                            const double kernel_length_scale          = 0.500,
                            const double noise_level                  = 0.005,
                            const double kernel_hyperparams_prior_var = 0.250,
                            const double btl_scale                    = 0.010);

        /// \brief Submit the result of user-performed line search and go to the next iteration step.
        ///
        /// \param slider_position The position of the chosen point in the slider space. It should be represented in [0,
        /// 1], where 0 means that the chosen position is the first end-point of the slider and 1 means that it is the
        /// second end-point.
        void SubmitLineSearchResult(const double slider_position);

        /// \brief Get the slider end-points.
        ///
        /// \details When the slider enlargement is not enabled, the first end-point is the maximizer among the observed
        /// points, and the second one is the maximizer of the acquisition function.
        std::pair<Eigen::VectorXd, Eigen::VectorXd> GetSliderEnds() const;

        /// \brief Calculate data point from a slider position.
        Eigen::VectorXd CalcPointFromSliderPosition(const double slider_position) const;

        /// \brief Get the point that has the highest value among the observed points.
        Eigen::VectorXd GetMaximizer() const;

        double GetPreferenceValueMean(const Eigen::VectorXd& point) const;
        double GetPreferenceValueStdev(const Eigen::VectorXd& point) const;
        double GetExpectedImprovementValue(const Eigen::VectorXd& point) const;

        const Eigen::MatrixXd& GetRawDataPoints() const;

        void DampData(const std::string& directory_path) const;

    private:
        const bool m_use_slider_enlargement;
        const bool m_use_map_hyperparams;

        std::shared_ptr<PreferenceRegressor>   m_regressor;
        std::shared_ptr<Slider>                m_slider;
        std::shared_ptr<PreferenceDataManager> m_data;

        double m_kernel_signal_var;
        double m_kernel_length_scale;
        double m_noise_level;
        double m_kernel_hyperparams_prior_var;
        double m_btl_scale;
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_SEQUENTIAL_LINE_SEARCH_HPP
