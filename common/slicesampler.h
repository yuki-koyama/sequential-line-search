#ifndef SLICESAMPLER_H
#define SLICESAMPLER_H

#include <Eigen/Core>

namespace SliceSampler
{
typedef double (*Func)(const Eigen::VectorXd& x, const void* data);

Eigen::VectorXd sampling(Func func, const void* data, const Eigen::VectorXd& x_last, const Eigen::VectorXd& bracket_size);
}

#endif // SLICESAMPLER_H
