#include "core.hpp"
#include <cmath>
#include <iostream>
#include <mathtoolbox/data-normalization.hpp>
#include <sequential-line-search/acquisition-function.hpp>
#include <sequential-line-search/preference-data-manager.hpp>
#include <sequential-line-search/preference-regressor.hpp>
#include <sequential-line-search/utils.hpp>

using namespace sequential_line_search;
using Eigen::MatrixXd;
using Eigen::VectorXd;

Core::Core()
{
    m_data = std::make_shared<PreferenceDataManager>();

    m_data->m_X = MatrixXd::Zero(0, 0);
    m_y         = VectorXd::Zero(0);

    m_x_max = VectorXd::Zero(0);
    m_y_max = NAN;
}

void Core::proceedOptimization()
{
    VectorXd x_plus;
    VectorXd x_acquisition;

    if (m_data->m_X.cols() == 0)
    {
        x_plus        = VectorXd::Constant(1, 0.3); // utils::GenerateRandomVector(1);
        x_acquisition = VectorXd::Constant(1, 0.6); // utils::GenerateRandomVector(1);
    }
    else
    {
        int x_index;
        m_y.maxCoeff(&x_index);

        x_plus        = m_data->m_X.col(x_index);
        x_acquisition = acquisition_func::FindNextPoint(*m_regressor);
    }

    const double y_plus        = evaluateObjectiveFunction(x_plus);
    const double y_acquisition = evaluateObjectiveFunction(x_acquisition);

    const VectorXd& x_chosen = y_plus > y_acquisition ? x_plus : x_acquisition;
    const VectorXd& x_other  = y_plus > y_acquisition ? x_acquisition : x_plus;

    m_data->AddNewPoints(x_chosen, {x_other});

    computeRegression();

    const int num_data_points = m_data->m_X.cols();

    VectorXd f(num_data_points);
    for (int i = 0; i < m_data->m_X.cols(); ++i)
    {
        f(i) = m_regressor->PredictMu(m_data->m_X.col(i));
    }

    m_normalizer = std::make_shared<mathtoolbox::DataNormalizer>(f.transpose());
    m_y          = VectorXd::Constant(f.size(), 1.0) + m_normalizer->GetNormalizedDataPoints().transpose();

    int best_index;
    m_y_max = m_y.maxCoeff(&best_index);
    m_x_max = m_data->m_X.col(best_index);
}

double Core::evaluateObjectiveFunction(const Eigen::VectorXd& x) const
{
    return 1.0 - 1.5 * x(0) * std::sin(x(0) * 13.0);
}

void Core::computeRegression()
{
    constexpr bool   use_map      = false;
    constexpr double signal_var   = 0.5;
    constexpr double length_scale = 0.5;
    constexpr double noise_level  = 0.0;
    constexpr double prior        = 0.0;

    m_regressor = std::make_shared<PreferenceRegressor>(
        m_data->m_X, m_data->m_D, use_map, signal_var, length_scale, noise_level, prior);
}
