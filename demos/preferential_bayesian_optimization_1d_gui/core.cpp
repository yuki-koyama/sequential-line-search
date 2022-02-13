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

    m_optimizer = std::make_shared<PreferentialBayesianOptimizer>(1,
                                                                  use_map,
                                                                  KernelType::ArdMatern52Kernel,
                                                                  AcquisitionFuncType::ExpectedImprovement,
                                                                  generator,
                                                                  CurrentBestSelectionStrategy::LastSelection);
    m_optimizer->SetHyperparams(signal_var, length_scale, noise_level, prior);

    m_y = VectorXd::Zero(0);

    m_x_max = VectorXd::Zero(0);
    m_y_max = NAN;
}

void Core::proceedOptimization()
{
    const auto options = m_optimizer->GetCurrentOptions();

    // The current implementation assumes pairwise comparison queries
    assert(options.size() == 2);

    // Simulate human response
    int    max_index;
    double max_value = -std::numeric_limits<double>::max();
    for (int i = 0; i < options.size(); ++i)
    {
        double value = evaluateObjectiveFunction(options[i]);

        if (max_value < value)
        {
            max_index = i;
            max_value = value;
        }
    }
    m_optimizer->SubmitFeedbackData(max_index);

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
