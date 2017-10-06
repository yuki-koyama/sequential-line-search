#ifndef PREFERENCE_H
#define PREFERENCE_H

#include <utility>
#include <vector>

// A pair (i, j) means that the i-th data point is preferable to the j-th (and k-th) data point
struct Preference : public std::vector<unsigned>
{
    Preference(unsigned i, unsigned j) : std::vector<unsigned>{i, j} {}
    Preference(unsigned i, unsigned j, unsigned k) : std::vector<unsigned>{i, j, k} {}
    Preference(const std::vector<unsigned>& indices) : std::vector<unsigned>{indices} {}
};

#endif // PREFERENCE_H
