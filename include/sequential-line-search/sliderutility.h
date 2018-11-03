#ifndef SLIDERUTILITY_H
#define SLIDERUTILITY_H

#include <Eigen/Core>
#include <vector>
#include <sequential-line-search/preference.h>

namespace sequential_line_search
{
    /// \brief Merge sampled points that are sufficiently closer.
    /// \param epsilon The threshold of the distance between sampled points to be merged
    void mergeData(Eigen::MatrixXd& X, std::vector<Preference>& D, const double epsilon = 5e-03);
}

#endif // SLIDERUTILITY_H
