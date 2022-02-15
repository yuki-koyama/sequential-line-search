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

    /// \brief A function type for specifying how initial query should be generated.
    ///
    /// \details The first parameter is the number of dimensions of the target problem. The second parameter is the
    /// number of options in each iteration (e.g., two in case of pairwise comparison).
    using InitialQueryGenerator = std::function<std::vector<Eigen::VectorXd>(const int, const int)>;

    /// \brief A utility function for initial query generation.
    ///
    /// \details This function is compatible with `InitialQueryGenerator`.
    std::vector<Eigen::VectorXd> GenerateRandomPoints(const int num_dims, const int num_options);

    /// \brief Optimizer class for performing preferential Bayesian optimization with discrete choice.
    ///
    /// \details This optimizer requests discrete choice queries. In the current implementation, the number of choices
    /// for each query is two by default (i.e., pairwise comparison). This class assumes that the search space is [0,
    /// 1]^{D}.
    class PreferentialBayesianOptimizer
    {
    public:
        /// \brief Construct an optimizer instance.
        ///
        /// \param use_map_hyperparams When this is set true, the optimizer always perform the MAP estimation for the
        /// GPR kernel hyperparameters. When this is set false, the optimizer performs the MAP estimation only for
        /// goodness values.
        ///
        /// \param num_options The number of discrete choices in each iteration. The default value is two (i.e.,
        /// pairwise comparison). The value should be no less than two.
        PreferentialBayesianOptimizer(
            const int                          num_dims,
            const bool                         use_map_hyperparams     = true,
            const KernelType                   kernel_type             = KernelType::ArdMatern52Kernel,
            const AcquisitionFuncType          acquisition_func_type   = AcquisitionFuncType::ExpectedImprovement,
            const InitialQueryGenerator&       initial_query_generator = GenerateRandomPoints,
            const CurrentBestSelectionStrategy current_best_selection_strategy =
                CurrentBestSelectionStrategy::LargestExpectValue,
            const int num_options = 2);

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

        /// \brief Submit the result of the user's selection and update the internal surrogate model.
        ///
        /// \param option_index The index of the chosen option, starting at zero.
        ///
        /// \param num_map_estimation_iters The number of iterations for the MAP estimation. When a non-positive value
        /// (e.g., 0) is specified, this is heuristically set.
        void SubmitFeedbackData(const int option_index, const int num_map_estimation_iters = 0);

        /// \brief Submit the result of the user's selection from custom arbitrary options and update the internal
        /// surrogate model.
        ///
        /// \details This method can be used intead of the `SubmitFeedbackData` method. While `SubmitFeedbackData`
        /// requires to use the options retrieved by `GetCurrentOptions`, this method allows to use arbitrary options.
        /// This may be useful for researchers who explore new variants of the PBO framework.
        ///
        /// \param chosen_option The option that is chosen.
        ///
        /// \param other_options The other options that are not chosen.
        ///
        /// \param num_map_estimation_iters The number of iterations for the MAP estimation. When a non-positive value
        /// (e.g., 0) is specified, this is heuristically set.
        void SubmitCustomFeedbackData(const Eigen::VectorXd&              chosen_option,
                                      const std::vector<Eigen::VectorXd>& other_options,
                                      const int                           num_map_estimation_iters = 0);

        /// \brief Determine the preferential query for the next iteration by using an acquisition function.
        ///
        /// \details This method is expected to be called right after `SubmitFeedbackData` is called.
        void DetermineNextQuery(const int num_global_search_iters = 0, const int num_local_search_iters = 0);

        /// \brief Get the current options.
        ///
        /// \details This getter method returns a list of `num_options` points. The first option is always the "current
        /// best" option, and the others are the new options determined by maximizing the acquisition function.
        const std::vector<Eigen::VectorXd>& GetCurrentOptions() const { return m_current_options; }

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
        const bool m_use_map_hyperparams;
        const int  m_num_options;

        const CurrentBestSelectionStrategy m_current_best_selection_strategy;

        std::shared_ptr<PreferenceRegressor>   m_regressor;
        std::shared_ptr<PreferenceDataManager> m_data;

        /// \details The number of options is always equivalent to `m_num_options`. For example, the size of this list
        /// is two in case of using pairwise comparison.
        std::vector<Eigen::VectorXd> m_current_options;

        double m_kernel_signal_var;
        double m_kernel_length_scale;
        double m_noise_level;
        double m_kernel_hyperparams_prior_var;
        double m_btl_scale;

        const KernelType          m_kernel_type;
        const AcquisitionFuncType m_acquisition_func_type;

        double m_gaussian_process_upper_confidence_bound_hyperparam;

        /// \brief Peform MAP estimation of the latent goodness values (and optionally the kernel hyperparameters).
        ///
        /// \details This private method is called by `SubmitFeedbackData` and `SubmitCustomFeedbackData`.
        ///
        /// \param num_map_estimation_iters The number of iterations for the MAP estimation. When a non-positive value
        /// (e.g., 0) is specified, this is heuristically set.
        void PerformMapEstimation(const int num_map_estimation_iters);
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_PREFERENTIAL_BAYESIAN_OPTIMIZER_HPP
