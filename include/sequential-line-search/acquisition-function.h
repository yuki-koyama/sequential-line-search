#ifndef acquisition_function_h
#define acquisition_function_h

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <sequential-line-search/regressor.h>

namespace sequential_line_search
{
    namespace acquisition_function
    {
        enum class FunctionType
        {
            ExpectedImprovement
        };
        
        /// \brief Calculate the value of the acquisition function value.
        /// \param function_type Type of the acquisition function.
        double CalculateAcqusitionValue(const Regressor&       regressor,
                                        const Eigen::VectorXd& x,
                                        const FunctionType     function_type = FunctionType::ExpectedImprovement);
        
        Eigen::VectorXd findNextPoint(Regressor &regressor);
        
        /// \brief Find the next n sampled points that should be observed. These points are determined
        ///        using the method by Schonlau et al. (1997)
        /// \param n The number of the sampled points.
        std::vector<Eigen::VectorXd> findNextPoints(const Regressor &regressor, unsigned n);
    }
}

#endif // acquisition_function_h
