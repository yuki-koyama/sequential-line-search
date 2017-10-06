#ifndef EXPECTEDIMPROVEMENTMAXIMIZER_H
#define EXPECTEDIMPROVEMENTMAXIMIZER_H

#include <memory>
#include <vector>
#include <Eigen/Core>

class Regressor;

namespace ExpectedImprovement
{
double calculateExpectedImprovedment(const Regressor& regressor, const Eigen::VectorXd &x);
Eigen::VectorXd findNextPoint(Regressor &regressor);
std::vector<Eigen::VectorXd> findNextPoints(const Regressor &regressor, unsigned n);
}

#endif // EXPECTEDIMPROVEMENTMAXIMIZER_H
