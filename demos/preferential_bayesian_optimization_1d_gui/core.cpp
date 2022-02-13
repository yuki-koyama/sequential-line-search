#include "core.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <mathtoolbox/data-normalization.hpp>
#include <sequential-line-search/preferential-bayesian-optimizer.hpp>

using namespace sequential_line_search;
using Eigen::MatrixXd;
using Eigen::VectorXd;

Core::Core()
{
    reset();
}

void Core::reset()
{
    // Define a generator that generates the initial options
    auto generator = [](int num_dims) {
        return std::vector<VectorXd>{VectorXd::Constant(1, 0.3), VectorXd::Constant(1, 0.6)};
    };

    constexpr bool   use_map      = false;
    constexpr double signal_var   = 0.5;
    constexpr double length_scale = 0.5;
    constexpr double noise_level  = 0.0;
    constexpr double prior        = 0.0;

    // Note: This app is for a one-dimensional problem.
    constexpr int num_dims = 1;

    // Note: This app uses pairwise comparison.
    constexpr int num_options = 2;

    m_optimizer = std::make_shared<PreferentialBayesianOptimizer>(num_dims,
                                                                  use_map,
                                                                  KernelType::ArdMatern52Kernel,
                                                                  AcquisitionFuncType::ExpectedImprovement,
                                                                  generator,
                                                                  CurrentBestSelectionStrategy::LastSelection,
                                                                  num_options);
    m_optimizer->SetHyperparams(signal_var, length_scale, noise_level, prior);

    m_y = VectorXd::Zero(0);

    m_x_max = VectorXd::Zero(0);
    m_y_max = NAN;
}

void Core::proceedOptimization()
{
    // Retrieve the current options
    const auto options = m_optimizer->GetCurrentOptions();

    // Simulate human response
    std::vector<double> values(options.size());
    for (int i = 0; i < options.size(); ++i)
    {
        values[i] = evaluateObjectiveFunction(options[i]);
    }
    const int max_index = std::distance(values.begin(), std::max_element(values.begin(), values.end()));

    // Submit the human's feedback and let the optimizer calculate a new preference model
    m_optimizer->SubmitFeedbackData(max_index);

    // Update internal data according to the new preference model
    const auto data_points     = m_optimizer->GetRawDataPoints();
    const int  num_data_points = data_points.cols();

    VectorXd f(num_data_points);
    for (int i = 0; i < num_data_points; ++i)
    {
        f(i) = m_optimizer->GetPreferenceValueMean(data_points.col(i));
    }

    m_normalizer = std::make_shared<mathtoolbox::DataNormalizer>(f.transpose());
    m_y          = VectorXd::Constant(f.size(), 1.0) + m_normalizer->GetNormalizedDataPoints().transpose();

    int best_index;
    m_y_max = m_y.maxCoeff(&best_index);
    m_x_max = data_points.col(best_index);

    // Determine the next pairwise comparison query
    m_optimizer->DetermineNextQuery();
}

double Core::evaluateObjectiveFunction(const Eigen::VectorXd& x) const
{
    return 1.0 - 1.5 * x(0) * std::sin(x(0) * 13.0);
}
