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
        /// \brief Add a new preference observation.
        ///
        /// \details If merge_close_points is true, this method will merge data points (including both existing and new
        /// ones) that are sufficiently close to each other with the threshold of epsilon.
        void AddNewPoints(const Eigen::VectorXd&              x_preferable,
                          const std::vector<Eigen::VectorXd>& xs_other,
                          const bool                          merge_close_points = true,
                          const double                        epsilon            = 1e-04);

        /// \brief Get the data point that was selected in the last preferential data observation
        const Eigen::VectorXd GetLastSelectedDataPoint() const { return m_X.col(GetLastDataSample()[0]); }

        /// \brief Get the number of data points
        int GetNumDataPoints() const { return m_X.cols(); }

        /// \brief Get the raw data points in a matrix format
        const Eigen::MatrixXd& GetX() const { return m_X; }

        /// \brief Get the list of preferential observations
        const std::vector<Preference>& GetD() const { return m_D; }

    private:
        /// \brief Get the last preferential feedback data sample
        const Preference& GetLastDataSample() const { return m_D.back(); }

        Eigen::MatrixXd         m_X;
        std::vector<Preference> m_D;
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_PREFERENCE_DATA_MANAGER_HPP
