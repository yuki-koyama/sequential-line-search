#ifndef SEQUENTIAL_LINE_SEARCH_KERNEL_TYPE_HPP
#define SEQUENTIAL_LINE_SEARCH_KERNEL_TYPE_HPP

#include <Eigen/Core>

namespace sequential_line_search
{
    enum class KernelType
    {
        ArdSquaredExponentialKernel,
        ArdMatern52Kernel,
    };

    using Kernel                   = double (*)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&);
    using KernelThetaDerivative    = Eigen::VectorXd (*)(const Eigen::VectorXd&,
                                                      const Eigen::VectorXd&,
                                                      const Eigen::VectorXd&);
    using KernelFirstArgDerivative = Eigen::VectorXd (*)(const Eigen::VectorXd&,
                                                         const Eigen::VectorXd&,
                                                         const Eigen::VectorXd&);
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_KERNEL_TYPE_HPP
