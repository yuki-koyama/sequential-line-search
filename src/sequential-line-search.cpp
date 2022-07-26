#include <sequential-line-search/acquisition-function.hpp>
#include <sequential-line-search/preference-data-manager.hpp>
#include <sequential-line-search/preference-regressor.hpp>
#include <sequential-line-search/sequential-line-search.hpp>
#include <sequential-line-search/slider.hpp>
#include <sequential-line-search/utils.hpp>
#include <stdexcept>

using Eigen::VectorXd;

std::pair<VectorXd, VectorXd> sequential_line_search::GenerateRandomSliderEnds(const int num_dims)
{
    return {0.5 * (VectorXd::Random(num_dims) + VectorXd::Ones(num_dims)),
            0.5 * (VectorXd::Random(num_dims) + VectorXd::Ones(num_dims))};
}

std::pair<VectorXd, VectorXd> sequential_line_search::GenerateCenteredFixedLengthRandomSliderEnds(const int num_dims)
{
    const VectorXd x_center   = VectorXd::Constant(num_dims, 0.50);
    const VectorXd random_dir = 0.5 * VectorXd::Random(num_dims);

    return {x_center + random_dir, x_center - random_dir};
}

sequential_line_search::SequentialLineSearchOptimizer::SequentialLineSearchOptimizer(
    const int                                                      num_dims,
    const bool                                                     use_slider_enlargement,
    const bool                                                     use_map_hyperparams,
    const KernelType                                               kernel_type,
    const AcquisitionFuncType                                      acquisition_func_type,
    const std::function<std::pair<VectorXd, VectorXd>(const int)>& initial_query_generator,
    const CurrentBestSelectionStrategy                             current_best_selection_strategy)
    : m_use_slider_enlargement(use_slider_enlargement),
      m_use_map_hyperparams(use_map_hyperparams),
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
    const auto slider_ends = initial_query_generator(num_dims);

    m_data      = std::make_shared<PreferenceDataManager>();
    m_regressor = nullptr;
    m_slider    = std::make_shared<Slider>(std::get<0>(slider_ends), std::get<1>(slider_ends), false);
}

void sequential_line_search::SequentialLineSearchOptimizer::SetHyperparams(const double kernel_signal_var,
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

void sequential_line_search::SequentialLineSearchOptimizer::SubmitFeedbackData(const double slider_position)
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

    SubmitFeedbackData(slider_position, num_map_estimation_iters, num_global_search_iters, num_local_search_iters);
}

void sequential_line_search::SequentialLineSearchOptimizer::SubmitFeedbackData(const double slider_position,
                                                                               const int    num_map_estimation_iters,
                                                                               const int    num_global_search_iters,
                                                                               const int    num_local_search_iters)
{
    const auto  x_chosen   = CalcPointFromSliderPosition(slider_position);
    const auto& x_prev_max = m_slider->original_end_0;
    const auto& x_prev_ei  = m_slider->original_end_1;

    // Update the data
    m_data->AddNewPoints(x_chosen, {x_prev_max, x_prev_ei}, true);

    // Perform the MAP estimation
    m_regressor = std::make_shared<PreferenceRegressor>(m_data->GetX(),
                                                        m_data->GetD(),
                                                        m_use_map_hyperparams,
                                                        m_kernel_signal_var,
                                                        m_kernel_length_scale,
                                                        m_noise_level,
                                                        m_kernel_hyperparams_prior_var,
                                                        m_btl_scale,
                                                        num_map_estimation_iters,
                                                        m_kernel_type);

    // Find the next search subspace
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

    m_slider = std::make_shared<Slider>(x_plus, x_acquisition, m_use_slider_enlargement);
}

std::pair<VectorXd, VectorXd> sequential_line_search::SequentialLineSearchOptimizer::GetSliderEnds() const
{
    return {m_slider->end_0, m_slider->end_1};
}

VectorXd
sequential_line_search::SequentialLineSearchOptimizer::CalcPointFromSliderPosition(const double slider_position) const
{
    return m_slider->GetValue(slider_position);
}

VectorXd sequential_line_search::SequentialLineSearchOptimizer::GetMaximizer() const
{
    return m_slider->original_end_0;
}

double sequential_line_search::SequentialLineSearchOptimizer::GetPreferenceValueMean(const VectorXd& point) const
{
    return (m_regressor == nullptr) ? 0.0 : m_regressor->PredictMu(point);
}

double sequential_line_search::SequentialLineSearchOptimizer::GetPreferenceValueStdev(const VectorXd& point) const
{
    return (m_regressor == nullptr) ? 0.0 : m_regressor->PredictSigma(point);
}

double sequential_line_search::SequentialLineSearchOptimizer::GetAcquisitionFuncValue(const VectorXd& point) const
{
    return (m_regressor == nullptr)
               ? 0.0
               : acquisition_func::CalcAcquisitionValue(*m_regressor,
                                                        point,
                                                        m_acquisition_func_type,
                                                        m_gaussian_process_upper_confidence_bound_hyperparam);
}

const Eigen::MatrixXd& sequential_line_search::SequentialLineSearchOptimizer::GetRawDataPoints() const
{
    return m_data->GetX();
}

void sequential_line_search::SequentialLineSearchOptimizer::DampData(const std::string& directory_path) const
{
    if (m_regressor == nullptr)
    {
        return;
    }

    m_regressor->DampData(directory_path);
}
