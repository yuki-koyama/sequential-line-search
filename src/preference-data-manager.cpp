#include <map>
#include <sequential-line-search/preference-data-manager.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace internal
{
    /// \brief Merge sampled points that are sufficiently closer.
    /// \param epsilon The threshold of the distance between sampled points to be merged
    void MergeClosePoints(const double epsilon, Eigen::MatrixXd& X, std::vector<sequential_line_search::Preference>& D);
} // namespace internal

void internal::MergeClosePoints(const double                                     epsilon,
                                Eigen::MatrixXd&                                 X,
                                std::vector<sequential_line_search::Preference>& D)
{
    const double eps_squared = epsilon * epsilon;

    while (true)
    {
        bool is_dirty = false;

        const unsigned N = X.rows();
        const unsigned M = X.cols();

        // Distance matrix (upper triangle only)
        MatrixXd distance_matrix(M, M);
        for (unsigned i = 0; i < M; ++i)
        {
            for (unsigned j = i + 1; j < M; ++j)
            {
                distance_matrix(i, j) = (X.col(i) - X.col(j)).squaredNorm();
            }
        }

        for (unsigned i = 0; i < M; ++i)
        {
            for (unsigned j = i + 1; j < M; ++j)
            {
                if (!is_dirty && distance_matrix(i, j) < eps_squared)
                {
                    // Construct a mapping from the old indices to the new one
                    std::map<unsigned, unsigned> mapping;
                    unsigned                     new_index = 0;
                    for (unsigned old_index = 0; old_index < M; ++old_index)
                    {
                        if (old_index != i && old_index != j)
                        {
                            mapping[old_index] = new_index++;
                        }
                    }
                    mapping[i] = M - 2;
                    mapping[j] = M - 2;

                    // Construct a new matrix
                    MatrixXd new_X(N, M - 1);
                    for (unsigned old_index = 0; old_index < M; ++old_index)
                    {
                        new_X.col(mapping[old_index]) = X.col(old_index);
                    }
                    new_X.col(M - 2) = 0.5 * (X.col(i) + X.col(j));

                    // Replace the data
                    X = new_X;

                    // Update the indices in the preference pairs
                    for (sequential_line_search::Preference& p : D)
                    {
                        for (unsigned i = 0; i < p.size(); ++i)
                        {
                            p[i] = mapping[p[i]];
                        }
                    }

                    is_dirty = true;
                }
            }
        }

        if (!is_dirty)
        {
            return;
        }
    }
}

void sequential_line_search::PreferenceDataManager::AddNewPoints(const Eigen::VectorXd&              x_preferable,
                                                                 const std::vector<Eigen::VectorXd>& xs_other,
                                                                 const bool                          merge_close_points,
                                                                 const double                        epsilon)
{
    if (m_X.rows() == 0)
    {
        // X
        const unsigned d = xs_other[0].rows();
        m_X              = MatrixXd(d, xs_other.size() + 1);
        m_X.col(0)       = x_preferable;
        for (unsigned i = 0; i < xs_other.size(); ++i)
        {
            m_X.col(i + 1) = xs_other[i];
        }

        // D
        std::vector<unsigned> indices(xs_other.size() + 1);
        for (unsigned i = 0; i < xs_other.size() + 1; ++i)
        {
            indices[i] = i;
        }
        m_D.push_back(Preference(indices));

        return;
    }

    const unsigned d = m_X.rows();
    const unsigned N = m_X.cols();

    // X
    MatrixXd new_X(d, N + xs_other.size() + 1);
    new_X.block(0, 0, d, N) = m_X;
    new_X.col(N)            = x_preferable;
    for (unsigned i = 0; i < xs_other.size(); ++i)
    {
        new_X.col(N + i + 1) = xs_other[i];
    }
    m_X = new_X;

    // D
    std::vector<unsigned> indices(xs_other.size() + 1);
    for (unsigned i = 0; i < xs_other.size() + 1; ++i)
    {
        indices[i] = N + i;
    }
    m_D.push_back(Preference(indices));

    // Merge
    if (merge_close_points)
    {
        internal::MergeClosePoints(epsilon, m_X, m_D);
    }
}
