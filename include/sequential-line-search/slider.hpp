#ifndef SEQUENTIAL_LINE_SEARCH_SLIDER_HPP
#define SEQUENTIAL_LINE_SEARCH_SLIDER_HPP

#include <Eigen/Core>

namespace sequential_line_search
{
    class Slider
    {
    public:
        Slider(const Eigen::VectorXd& end_0,
               const Eigen::VectorXd& end_1,
               const bool             enlarge,
               const double           scale          = 1.25,
               const double           minimum_length = 0.25);

        Eigen::VectorXd GetValue(const double t) const { return (1.0 - t) * end_0 + t * end_1; }

        Eigen::VectorXd end_0;
        Eigen::VectorXd end_1;

        Eigen::VectorXd original_end_0;
        Eigen::VectorXd original_end_1;
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_SLIDER_HPP
