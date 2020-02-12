#ifndef SEQUENTIAL_LINE_SEARCH_ACQUISITION_FUNCTION_HPP
#define SEQUENTIAL_LINE_SEARCH_ACQUISITION_FUNCTION_HPP

#include <Eigen/Core>
#include <memory>
#include <sequential-line-search/regressor.hpp>
#include <vector>

namespace sequential_line_search
{
    enum class AcquisitionFuncType
    {
        ExpectedImprovement,
        GaussianProcessUpperConfidenceBound,
    };

    namespace acquisition_function
    {
        /// \brief Calculate the value of the acquisition function value.
        ///
        /// \param function_type Type of the acquisition function.
        double CalcAcqusitionValue(const Regressor&          regressor,
                                   const Eigen::VectorXd&    x,
                                   const AcquisitionFuncType function_type);

        Eigen::VectorXd CalcAcquisitionValueDerivative(const Regressor&          regressor,
                                                       const Eigen::VectorXd&    x,
                                                       const AcquisitionFuncType function_type);

        /// \param num_global_search_iters The number of trials of acquisition value maximization. Specifying a large
        /// number is helpful for finding the global maximizer while it increases the computational cost proportional to
        /// it.
        Eigen::VectorXd
        FindNextPoint(const Regressor&          regressor,
                      const unsigned            num_global_search_iters = 100,
                      const unsigned            num_local_search_iters  = 50,
                      const AcquisitionFuncType function_type           = AcquisitionFuncType::ExpectedImprovement);

        /// \brief Find the next n sampled points that should be observed.
        ///
        /// \details The points will be determined by the method by Schonlau et al. (1997)
        ///
        /// \param num_points The number of the sampled points.
        ///
        /// \param num_global_search_iters The number of trials of acquisition value maximization. Specifying a large
        /// number is helpful for finding the global maximizer while it increases the computational cost proportional to
        /// it.
        std::vector<Eigen::VectorXd>
        FindNextPoints(const Regressor&          regressor,
                       const unsigned            num_points,
                       const unsigned            num_global_search_iters = 100,
                       const unsigned            num_local_search_iters  = 50,
                       const AcquisitionFuncType function_type           = AcquisitionFuncType::ExpectedImprovement);
    } // namespace acquisition_function
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_ACQUISITION_FUNCTION_HPP
