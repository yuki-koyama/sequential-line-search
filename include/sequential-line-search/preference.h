#ifndef PREFERENCE_H
#define PREFERENCE_H

#include <vector>

namespace sequential_line_search
{
    /// \brief Struct for storing preference among 2 or more data points
    /// \details A pair (i, j) means that the i-th data point is preferable to the j-th data point
    struct Preference : public std::vector<unsigned>
    {
        Preference(unsigned i, unsigned j) : std::vector<unsigned>{ i, j } {}
        Preference(unsigned i, unsigned j, unsigned k) : std::vector<unsigned>{ i, j, k } {}
        Preference(const std::vector<unsigned>& indices) : std::vector<unsigned>{ indices } {}
    };
}

#endif // PREFERENCE_H
