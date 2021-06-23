#ifndef SEQUENTIAL_LINE_SEARCH_PREFERENCE_DATA_MANAGER_HPP
#define SEQUENTIAL_LINE_SEARCH_PREFERENCE_DATA_MANAGER_HPP

#include <Eigen/Core>
#include <sequential-line-search/preference.hpp>
#include <vector>

namespace sequential_line_search
{
    /// \brief Utility class for managing preferential data observed during optimization.
    class PreferenceDataManager
    {
    public:
        Eigen::MatrixXd         m_X;
        std::vector<Preference> m_D;

        /// \brief Add a new preference observation.
        ///
        /// \details If merge_close_points is true, this method will merge data points (including both existing and new
        /// ones) that are sufficiently close to each other with the threshold of epsilon.
        void AddNewPoints(const Eigen::VectorXd&              x_preferable,
                          const std::vector<Eigen::VectorXd>& xs_other,
                          const bool                          merge_close_points = true,
                          const double                        epsilon            = 1e-04);

        /// \brief Get the last preferential feedback data sample
        const Preference& GetLastDataSample() const { return m_D.back(); }

        /// \brief Get the number of data points
        int GetNumDataPoints() const { return m_X.cols(); }
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_PREFERENCE_DATA_MANAGER_HPP
