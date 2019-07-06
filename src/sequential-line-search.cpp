#include <sequential-line-search/acquisition-function.h>
#include <sequential-line-search/data.h>
#include <sequential-line-search/preference-regressor.h>
#include <sequential-line-search/sequential-line-search.h>
#include <sequential-line-search/slider.h>
#include <sequential-line-search/utils.h>
#include <stdexcept>

sequential_line_search::SequentialLineSearchOptimizer::SequentialLineSearchOptimizer(const int  dimension,
                                                                                     const bool use_slider_enlargement,
                                                                                     const bool use_MAP_hyperparameters)
    : m_dimension(dimension), m_use_slider_enlargement(use_slider_enlargement),
      m_use_MAP_hyperparameters(use_MAP_hyperparameters)
{
    m_data      = std::make_shared<Data>();
    m_regressor = nullptr;
    m_slider    = std::make_shared<Slider>(
        utils::generateRandomVector(m_dimension), utils::generateRandomVector(m_dimension), false);
}

void sequential_line_search::SequentialLineSearchOptimizer::setHyperparameters(
    const double a, const double r, const double b, const double variance, const double btl_scale)
{
    PreferenceRegressor::Params::getInstance().a         = a;
    PreferenceRegressor::Params::getInstance().r         = r;
    PreferenceRegressor::Params::getInstance().b         = b;
    PreferenceRegressor::Params::getInstance().variance  = variance;
    PreferenceRegressor::Params::getInstance().btl_scale = btl_scale;
}

void sequential_line_search::SequentialLineSearchOptimizer::submit(const double slider_position)
{
    const auto x_chosen       = getParameters(slider_position);
    const auto xs_slider_ends = getSliderEnds();

    m_data->AddNewPoints(x_chosen, {xs_slider_ends.first, xs_slider_ends.second}, true);

    m_regressor = std::make_shared<PreferenceRegressor>(m_data->X, m_data->D, m_use_MAP_hyperparameters);

    const auto x_max = getMaximizer();
    const auto x_ei  = acquisition_function::FindNextPoint(*m_regressor);

    m_slider = std::make_shared<Slider>(x_max, x_ei, m_use_slider_enlargement);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> sequential_line_search::SequentialLineSearchOptimizer::getSliderEnds() const
{
    return {m_slider->orig_0, m_slider->orig_1};
}

Eigen::VectorXd sequential_line_search::SequentialLineSearchOptimizer::getParameters(const double slider_position) const
{
    return m_slider->getValue(slider_position);
}

Eigen::VectorXd sequential_line_search::SequentialLineSearchOptimizer::getMaximizer() const
{
    if (m_regressor == nullptr)
    {
        throw std::runtime_error("");
    }

    return m_regressor->find_arg_max();
}

double sequential_line_search::SequentialLineSearchOptimizer::getPreferenceValue(const Eigen::VectorXd& parameter) const
{
    return (m_regressor == nullptr) ? 0.0 : m_regressor->estimate_y(parameter);
}

double sequential_line_search::SequentialLineSearchOptimizer::getPreferenceValueStandardDeviation(const Eigen::VectorXd& parameter) const
{
    return (m_regressor == nullptr) ? 0.0 : m_regressor->estimate_s(parameter);
}

double sequential_line_search::SequentialLineSearchOptimizer::getExpectedImprovementValue(const Eigen::VectorXd& parameter) const
{
    return (m_regressor == nullptr) ? 0.0 : acquisition_function::CalculateAcqusitionValue(*m_regressor, parameter);
}

const Eigen::MatrixXd& sequential_line_search::SequentialLineSearchOptimizer::getRawDataPoints() const
{
    return m_data->X;
}
