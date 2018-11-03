#ifndef EXPECTEDIMPROVEMENTMAXIMIZER_H
#define EXPECTEDIMPROVEMENTMAXIMIZER_H

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <sequential-line-search/regressor.h>

namespace acquisition_function
{
    double calculateExpectedImprovedment(const Regressor& regressor, const Eigen::VectorXd &x);
    Eigen::VectorXd findNextPoint(Regressor &regressor);
    
    /// \brief Find the next n sampled points that should be observed. These points are determined
    ///        using the method by Schonlau et al. (1997)
    /// \param n The number of the sampled points.
    std::vector<Eigen::VectorXd> findNextPoints(const Regressor &regressor, unsigned n);
}

#endif // EXPECTEDIMPROVEMENTMAXIMIZER_H
