#ifndef SLIDERUTILITY_H
#define SLIDERUTILITY_H

#include <utility>
#include <Eigen/Core>
#include "preference.h"

namespace SliderUtility
{
    /// \brief Given a pair of slider ends, this function enlarges the slider by a specified scale
    ///        but considering the bounding box constraint.
    std::pair<Eigen::VectorXd, Eigen::VectorXd> enlargeSliderEnds(const Eigen::VectorXd& x_1,
                                                                  const Eigen::VectorXd& x_2,
                                                                  const double scale = 1.25,
                                                                  const double minimum_length = 1e-10);
    
    /// \brief Merge sampled points that are sufficiently closer.
    /// \param epsilon The threshold of the distance between sampled points to be merged
    void mergeData(Eigen::MatrixXd& X, std::vector<Preference>& D, const double epsilon = 5e-03);
}

#endif // SLIDERUTILITY_H
