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

    namespace acquisition_func
    {
        /// \brief Calculate the value of the acquisition function value.
        ///
        /// \param function_type Type of the acquisition function.
        ///
        /// \param gaussian_process_upper_confidence_bound_hyperparam The hyperparameter in the GP-UCB algorithm, which
        /// controls the trade-off of exploration and exploitation. If the acquisition function is not GP-UCB, this
        /// value will not be used.
        double CalcAcquisitionValue(const Regressor&          regressor,
                                    const Eigen::VectorXd&    x,
                                    const AcquisitionFuncType func_type,
                                    const double              gaussian_process_upper_confidence_bound_hyperparam = 1.0);

        /// \param gaussian_process_upper_confidence_bound_hyperparam The hyperparameter in the GP-UCB algorithm, which
        /// controls the trade-off of exploration and exploitation. If the acquisition function is not GP-UCB, this
        /// value will not be used.
        Eigen::VectorXd
        CalcAcquisitionValueDerivative(const Regressor&          regressor,
                                       const Eigen::VectorXd&    x,
                                       const AcquisitionFuncType func_type,
                                       const double gaussian_process_upper_confidence_bound_hyperparam = 1.0);

        /// \param num_global_search_iters The number of trials of acquisition value maximization. Specifying a large
        /// number is helpful for finding the global maximizer while it increases the computational cost proportional to
        /// it.
        ///
        /// \param gaussian_process_upper_confidence_bound_hyperparam The hyperparameter in the GP-UCB algorithm, which
        /// controls the trade-off of exploration and exploitation. If the acquisition function is not GP-UCB, this
        /// value will not be used.
        Eigen::VectorXd FindNextPoint(const Regressor&          regressor,
                                      const unsigned            num_global_search_iters = 100,
                                      const unsigned            num_local_search_iters  = 50,
                                      const AcquisitionFuncType func_type = AcquisitionFuncType::ExpectedImprovement,
                                      const double gaussian_process_upper_confidence_bound_hyperparam = 1.0);

        /// \brief Find the next n sampled points that should be observed.
        ///
        /// \details The points will be determined by Schonlau et al.'s method [1998]. This method determines the points
        /// one by one sequentially by maximizing the acquisition function. In each maximization, the variance of the
        /// surrogate function (or the covariance matrix of the underlying GP model) is updated using the newly sampled
        /// point, by which it avoids sampling similar points.
        ///
        /// Matthias Schonlau, William J. Welch, and Donald R. Jones. 1998. Global versus local search in constrained
        /// optimization of computer models. Institute of Mathematical Statistics Lecture Notes - Monograph Series,
        /// 1998: 11-25 (1998). DOI: https://doi.org/10.1214/lnms/1215456182
        ///
        /// \param num_points The number of the sampled points.
        ///
        /// \param num_global_search_iters The number of trials of acquisition value maximization. Specifying a large
        /// number is helpful for finding the global maximizer while it increases the computational cost proportional to
        /// it.
        ///
        /// \param gaussian_process_upper_confidence_bound_hyperparam The hyperparameter in the GP-UCB algorithm, which
        /// controls the trade-off of exploration and exploitation. If the acquisition function is not GP-UCB, this
        /// value will not be used.
        std::vector<Eigen::VectorXd>
        FindNextPoints(const Regressor&          regressor,
                       const unsigned            num_points,
                       const unsigned            num_global_search_iters = 100,
                       const unsigned            num_local_search_iters  = 50,
                       const AcquisitionFuncType func_type               = AcquisitionFuncType::ExpectedImprovement,
                       const double              gaussian_process_upper_confidence_bound_hyperparam = 1.0);
    } // namespace acquisition_func
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_ACQUISITION_FUNCTION_HPP
