#ifndef SEQUENTIAL_LINE_SEARCH_PREFERENCE_HPP
#define SEQUENTIAL_LINE_SEARCH_PREFERENCE_HPP

#include <vector>

namespace sequential_line_search
{
    /// \brief Struct for storing preference among 2 or more data points
    struct Preference : public std::vector<unsigned>
    {
        /// \details A pair (i, j) means that the data point i is preferable to the data point j.
        Preference(unsigned i, unsigned j) : std::vector<unsigned>{i, j} {}

        /// \details A tuple (i, j, k) means that the data point i is preferable to both the data points j and k.
        Preference(unsigned i, unsigned j, unsigned k) : std::vector<unsigned>{i, j, k} {}

        /// \details A list (i, j, k, ...) means that data point i is preferable to any other data points.
        Preference(const std::vector<unsigned>& indices) : std::vector<unsigned>{indices} {}
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_PREFERENCE_HPP
