#ifndef sequential_line_search_h
#define sequential_line_search_h

#include <Eigen/Core>
#include <memory>
#include <utility>

namespace sequential_line_search
{
    class PreferenceRegressor;
    class Slider;
    class Data;

    class SequentialLineSearchOptimizer
    {
    public:
        SequentialLineSearchOptimizer(const int  dimension,
                                      const bool use_slider_enlargement  = true,
                                      const bool use_MAP_hyperparameters = true);

        void setHyperparameters(
            const double a, const double r, const double b, const double variance, const double btl_scale);

        void submit(const double slider_position);

        std::pair<Eigen::VectorXd, Eigen::VectorXd> getSliderEnds() const;
        Eigen::VectorXd                             getParameters(const double slider_position) const;

        Eigen::VectorXd getMaximizer() const;

    private:
        const int  m_dimension;
        const bool m_use_slider_enlargement;
        const bool m_use_MAP_hyperparameters;

        std::shared_ptr<PreferenceRegressor> m_regressor;
        std::shared_ptr<Slider>              m_slider;
        std::shared_ptr<Data>                m_data;
    };
} // namespace sequential_line_search

#endif // sequential_line_search_h
