#ifndef SEQUENTIAL_LINE_SEARCH_SEQUENTIAL_LINE_SEARCH_HPP
#define SEQUENTIAL_LINE_SEARCH_SEQUENTIAL_LINE_SEARCH_HPP

#include <Eigen/Core>
#include <memory>
#include <sequential-line-search/acquisition-function.hpp>
#include <sequential-line-search/current-best-selection-strategy.hpp>
#include <sequential-line-search/kernel-type.hpp>
#include <utility>

namespace sequential_line_search
{
    class PreferenceRegressor;
    class Slider;
    class PreferenceDataManager;

    std::pair<Eigen::VectorXd, Eigen::VectorXd> GenerateRandomSliderEnds(const int num_dims);
    std::pair<Eigen::VectorXd, Eigen::VectorXd> GenerateCenteredFixedLengthRandomSliderEnds(const int num_dims);

    /// \brief Optimizer class for performing sequential line search.
    ///
    /// \details This class assumes that the search space is [0, 1]^{D}.
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
        SequentialLineSearchOptimizer(
            const int                 num_dims,
            const bool                use_slider_enlargement = true,
            const bool                use_map_hyperparams    = true,
            const KernelType          kernel_type            = KernelType::ArdMatern52Kernel,
            const AcquisitionFuncType acquisition_func_type  = AcquisitionFuncType::ExpectedImprovement,
            const std::function<std::pair<Eigen::VectorXd, Eigen::VectorXd>(const int)>& initial_query_generator =
                GenerateRandomSliderEnds,
            const CurrentBestSelectionStrategy current_best_selection_strategy =
                CurrentBestSelectionStrategy::LargestExpectValue);

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
        ///
        /// \details The computational effort for finding the global maximizer of the acquisition function will be
        /// automatically set by a naive heuristics.
        void SubmitFeedbackData(const double slider_position);

        /// \brief Submit the result of user-performed line search and go to the next iteration step.
        void SubmitFeedbackData(const double slider_position,
                                const int    num_map_estimation_iters,
                                const int    num_global_search_iters,
                                const int    num_local_search_iters);

        /// \brief Get the slider end-points.
        ///
        /// \details When the slider enlargement is not enabled, the first end-point is the maximizer among the observed
        /// points, and the second one is the maximizer of the acquisition function.
        std::pair<Eigen::VectorXd, Eigen::VectorXd> GetSliderEnds() const;

        /// \brief Calculate data point from a slider position.
        Eigen::VectorXd CalcPointFromSliderPosition(const double slider_position) const;

        /// \brief Get the point that has the highest value among the observed points.
        ///
        /// \details The point is selected according to `CurrentBestSelectionStrategy`.
        Eigen::VectorXd GetMaximizer() const;

        double GetPreferenceValueMean(const Eigen::VectorXd& point) const;
        double GetPreferenceValueStdev(const Eigen::VectorXd& point) const;
        double GetAcquisitionFuncValue(const Eigen::VectorXd& point) const;

        const Eigen::MatrixXd& GetRawDataPoints() const;

        void DampData(const std::string& directory_path) const;

        /// \brief Set the hyperparameter in the GP-UCB algorithm.
        ///
        /// \details This hyperparameter controls the trade-off of exploration and exploitation. Specifically, this
        /// hyperparameter corresponds to the square root of the beta in [Srinivas et al. ICML '10].
        ///
        /// If the acquisition function is not GP-UCB, this value will not be used.
        void SetGaussianProcessUpperConfidenceBoundHyperparam(const double hyperparam)
        {
            m_gaussian_process_upper_confidence_bound_hyperparam = hyperparam;
        }

    private:
        const bool m_use_slider_enlargement;
        const bool m_use_map_hyperparams;

        const CurrentBestSelectionStrategy m_current_best_selection_strategy;

        std::shared_ptr<PreferenceRegressor>   m_regressor;
        std::shared_ptr<Slider>                m_slider;
        std::shared_ptr<PreferenceDataManager> m_data;

        double m_kernel_signal_var;
        double m_kernel_length_scale;
        double m_noise_level;
        double m_kernel_hyperparams_prior_var;
        double m_btl_scale;

        const KernelType          m_kernel_type;
        const AcquisitionFuncType m_acquisition_func_type;

        double m_gaussian_process_upper_confidence_bound_hyperparam;
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_SEQUENTIAL_LINE_SEARCH_HPP
