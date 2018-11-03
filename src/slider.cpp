#include <sequential-line-search/slider.h>
#include <sequential-line-search/sliderutility.h>

Slider::Slider(const Eigen::VectorXd& end_0, const Eigen::VectorXd& end_1, const bool enlarge, const double scale, const double minimum_length) :
    enlarge(enlarge),
    minimum_length(minimum_length),
    orig_0(end_0),
    orig_1(end_1)
{
    if (enlarge)
    {
        const auto ends = SliderUtility::enlargeSliderEnds(orig_0, orig_1, scale, minimum_length);
        this->end_0 = ends.first;
        this->end_1 = ends.second;
    }
    else
    {
        this->end_0 = orig_0;
        this->end_1 = orig_1;
    }
}
