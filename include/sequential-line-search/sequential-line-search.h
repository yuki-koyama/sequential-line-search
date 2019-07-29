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
                                      const bool use_map_hyperparameters = true);

        void SetHyperparameters(
            const double a, const double r, const double b, const double variance, const double btl_scale);

        void SubmitLineSearchResult(const double slider_position);

        std::pair<Eigen::VectorXd, Eigen::VectorXd> GetSliderEnds() const;
        Eigen::VectorXd                             GetParameters(const double slider_position) const;

        Eigen::VectorXd GetMaximizer() const;

        double GetPreferenceValueMean(const Eigen::VectorXd& parameter) const;
        double GetPreferenceValueStandardDeviation(const Eigen::VectorXd& parameter) const;
        double GetExpectedImprovementValue(const Eigen::VectorXd& parameter) const;

        const Eigen::MatrixXd& GetRawDataPoints() const;

        void DampData(const std::string& directory_path) const;

    private:
        const int  m_dimension;
        const bool m_use_slider_enlargement;
        const bool m_use_map_hyperparameters;

        std::shared_ptr<PreferenceRegressor> m_regressor;
        std::shared_ptr<Slider>              m_slider;
        std::shared_ptr<Data>                m_data;

        double m_a;
        double m_r;
        double m_b;
        double m_variance;
        double m_btl_scale;
    };
} // namespace sequential_line_search

#endif // sequential_line_search_h
