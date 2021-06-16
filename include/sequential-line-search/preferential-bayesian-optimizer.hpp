#ifndef SEQUENTIAL_LINE_SEARCH_PREFERENTIAL_BAYESIAN_OPTIMIZER_HPP
#define SEQUENTIAL_LINE_SEARCH_PREFERENTIAL_BAYESIAN_OPTIMIZER_HPP

#include <Eigen/Core>
#include <memory>
#include <sequential-line-search/acquisition-function.hpp>
#include <sequential-line-search/current-best-selection-strategy.hpp>
#include <sequential-line-search/kernel-type.hpp>
#include <utility>
#include <vector>

namespace sequential_line_search
{
    class PreferenceRegressor;
    class PreferenceDataManager;

    std::vector<Eigen::VectorXd> GenerateRandomPoints(const int num_dims);

    /// \brief Optimizer class for performing preferential Bayesian optimization with discrete choice.
    ///
    /// \details This optimizer requests discrete choice queries. In the current implementation, the number of choices
    /// for each query is fixed to two (i.e., pairwise comparison). This class assumes that the search space is [0,
    /// 1]^{D}.
    class PreferentialBayesianOptimizer
    {
    public:
        /// \brief Construct an optimizer instance.
        ///
        /// \param use_map_hyperparams When this is set true, the optimizer always perform the MAP estimation for the
        /// GPR kernel hyperparameters. When this is set false, the optimizer performs the MAP estimation only for
        /// goodness values.
        PreferentialBayesianOptimizer(
            const int                 num_dims,
            const bool                use_map_hyperparams   = true,
            const KernelType          kernel_type           = KernelType::ArdMatern52Kernel,
            const AcquisitionFuncType acquisition_func_type = AcquisitionFuncType::ExpectedImprovement,
            const std::function<std::vector<Eigen::VectorXd>(const int)>& initial_query_generator =
                GenerateRandomPoints,
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

        /// \brief Submit the result of the user's selection and go to the next iteration step.
        ///
        /// \details The computational effort for finding the global maximizer of the acquisition function will be
        /// automatically set by a naive heuristics.
        void SubmitFeedbackData(const int option_index);

        /// \brief Submit the result of the user's selection and go to the next iteration step.
        void SubmitFeedbackData(const int option_index,
                                const int num_map_estimation_iters,
                                const int num_global_search_iters,
                                const int num_local_search_iters);

        /// \brief Get the current options.
        const std::vector<Eigen::VectorXd>& GetCurrentOptions() const { return m_current_options; }

        /// \brief Get the point that has the highest value among the observed points.
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
        const bool m_use_map_hyperparams;

        const CurrentBestSelectionStrategy m_current_best_selection_strategy;

        std::shared_ptr<PreferenceRegressor>   m_regressor;
        std::shared_ptr<PreferenceDataManager> m_data;

        /// \details In the case of using pairwise comparison, the number of options is always two.
        std::vector<Eigen::VectorXd> m_current_options;

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

#endif // SEQUENTIAL_LINE_SEARCH_PREFERENTIAL_BAYESIAN_OPTIMIZER_HPP
