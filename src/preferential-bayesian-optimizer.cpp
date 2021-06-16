#include <sequential-line-search/acquisition-function.hpp>
#include <sequential-line-search/preference-data-manager.hpp>
#include <sequential-line-search/preference-regressor.hpp>
#include <sequential-line-search/preferential-bayesian-optimizer.hpp>
#include <sequential-line-search/utils.hpp>
#include <stdexcept>

using Eigen::VectorXd;

std::vector<VectorXd> sequential_line_search::GenerateRandomPoints(const int num_dims)
{
    // Note: Currently, the number of options is always fixed to two.
    constexpr int num_options = 2;

    std::vector<VectorXd> options;
    for (int i = 0; i < num_options; ++i)
    {
        options.push_back(0.5 * (VectorXd::Random(num_dims) + VectorXd::Ones(num_dims)));
    }

    return options;
}

sequential_line_search::PreferentialBayesianOptimizer::PreferentialBayesianOptimizer(
    const int                                              num_dims,
    const bool                                             use_map_hyperparams,
    const KernelType                                       kernel_type,
    const AcquisitionFuncType                              acquisition_func_type,
    const std::function<std::vector<VectorXd>(const int)>& initial_query_generator,
    const CurrentBestSelectionStrategy                     current_best_selection_strategy)
    : m_use_map_hyperparams(use_map_hyperparams),
      m_current_best_selection_strategy(current_best_selection_strategy),
      m_kernel_signal_var(0.500),
      m_kernel_length_scale(0.500),
      m_noise_level(0.005),
      m_kernel_hyperparams_prior_var(0.250),
      m_btl_scale(0.010),
      m_kernel_type(kernel_type),
      m_acquisition_func_type(acquisition_func_type),
      m_gaussian_process_upper_confidence_bound_hyperparam(1.0)
{
    m_data            = std::make_shared<PreferenceDataManager>();
    m_regressor       = nullptr;
    m_current_options = initial_query_generator(num_dims);
}

void sequential_line_search::PreferentialBayesianOptimizer::SetHyperparams(const double kernel_signal_var,
                                                                           const double kernel_length_scale,
                                                                           const double noise_level,
                                                                           const double kernel_hyperparams_prior_var,
                                                                           const double btl_scale)
{
    m_kernel_signal_var            = kernel_signal_var;
    m_kernel_length_scale          = kernel_length_scale;
    m_noise_level                  = noise_level;
    m_kernel_hyperparams_prior_var = kernel_hyperparams_prior_var;
    m_btl_scale                    = btl_scale;
}

void sequential_line_search::PreferentialBayesianOptimizer::SubmitFeedbackData(const int option_index)
{
    // A heuristics to set the computational effort for solving the maximization of the acquisition function. This is
    // not justified or validated.
    const int num_dims                 = GetMaximizer().size();
    const int num_map_estimation_iters = 100;
#ifdef SEQUENTIAL_LINE_SEARCH_USE_PARALLELIZED_MULTI_START_SEARCH
    const int num_global_search_iters = 10;
#else
    const int num_global_search_iters = 50 * num_dims;
#endif
    const int num_local_search_iters = 10 * num_dims;

    SubmitFeedbackData(option_index, num_map_estimation_iters, num_global_search_iters, num_local_search_iters);
}

void sequential_line_search::PreferentialBayesianOptimizer::SubmitFeedbackData(const int option_index,
                                                                               const int num_map_estimation_iters,
                                                                               const int num_global_search_iters,
                                                                               const int num_local_search_iters)
{
    assert(option_index >= 0 && option_index < m_current_options.size());

    const auto x_chosen = m_current_options[option_index];

    std::vector<VectorXd> x_others = m_current_options;
    x_others.erase(x_others.begin() + option_index);

    // Update the data
    m_data->AddNewPoints(x_chosen, x_others, true);

    // Perform the MAP estimation
    m_regressor = std::make_shared<PreferenceRegressor>(m_data->m_X,
                                                        m_data->m_D,
                                                        m_use_map_hyperparams,
                                                        m_kernel_signal_var,
                                                        m_kernel_length_scale,
                                                        m_noise_level,
                                                        m_kernel_hyperparams_prior_var,
                                                        m_btl_scale,
                                                        num_map_estimation_iters,
                                                        m_kernel_type);

    // Find the next search space
    const auto x_plus = [&]() -> VectorXd
    {
        switch (m_current_best_selection_strategy)
        {
            case CurrentBestSelectionStrategy::LargestExpectValue:
                return m_regressor->FindArgMax();
            case CurrentBestSelectionStrategy::LastSelection:
                return x_chosen;
        }
    }();
    const auto x_acquisition = acquisition_func::FindNextPoint(*m_regressor,
                                                               num_global_search_iters,
                                                               num_local_search_iters,
                                                               m_acquisition_func_type,
                                                               m_gaussian_process_upper_confidence_bound_hyperparam);

    m_current_options = {x_plus, x_acquisition};
}

VectorXd sequential_line_search::PreferentialBayesianOptimizer::GetMaximizer() const
{
    // This code assumes that the first option always represents the current-best data point
    return m_current_options[0];
}

double sequential_line_search::PreferentialBayesianOptimizer::GetPreferenceValueMean(const VectorXd& point) const
{
    return (m_regressor == nullptr) ? 0.0 : m_regressor->PredictMu(point);
}

double sequential_line_search::PreferentialBayesianOptimizer::GetPreferenceValueStdev(const VectorXd& point) const
{
    return (m_regressor == nullptr) ? 0.0 : m_regressor->PredictSigma(point);
}

double sequential_line_search::PreferentialBayesianOptimizer::GetAcquisitionFuncValue(const VectorXd& point) const
{
    return (m_regressor == nullptr)
               ? 0.0
               : acquisition_func::CalcAcqusitionValue(*m_regressor,
                                                       point,
                                                       m_acquisition_func_type,
                                                       m_gaussian_process_upper_confidence_bound_hyperparam);
}

const Eigen::MatrixXd& sequential_line_search::PreferentialBayesianOptimizer::GetRawDataPoints() const
{
    return m_data->m_X;
}

void sequential_line_search::PreferentialBayesianOptimizer::DampData(const std::string& directory_path) const
{
    if (m_regressor == nullptr)
    {
        return;
    }

    m_regressor->DampData(directory_path);
}
