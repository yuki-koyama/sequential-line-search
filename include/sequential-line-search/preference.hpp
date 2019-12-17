#ifndef SEQUENTIAL_LINE_SEARCH_PREFERENCE_HPP
#define SEQUENTIAL_LINE_SEARCH_PREFERENCE_HPP

#include <vector>

namespace sequential_line_search
{
    /// \brief Struct for storing preference among 2 or more data points
    struct Preference : public std::vector<unsigned>
    {
        /// \details A pair (i, j) means that the i-th data point is preferable to the j-th data point
        Preference(unsigned i, unsigned j) : std::vector<unsigned>{i, j} {}

        /// \details A tuple (i, j, k) means that the i-th data point is preferable to both the j-th and k-th data
        /// points
        Preference(unsigned i, unsigned j, unsigned k) : std::vector<unsigned>{i, j, k} {}

        /// \details A list (i, j, k, ...) means that the i-th data point is preferable to any other data points
        Preference(const std::vector<unsigned>& indices) : std::vector<unsigned>{indices} {}
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_PREFERENCE_HPP
