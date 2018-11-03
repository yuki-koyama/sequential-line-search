#include <sequential-line-search/sliderutility.h>
#include <map>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace sequential_line_search
{
    namespace data
    {
        void MergeCloseData(MatrixXd& X, std::vector<Preference>& D, const double epsilon)
        {
            const double eps_squared = epsilon * epsilon;
            
            while (true)
            {
                bool dirty = false;
                
                const unsigned N = X.rows();
                const unsigned M = X.cols();
                
                // Distance matrix (upper triangle only)
                MatrixXd Dist(M, M);
                for (unsigned i = 0; i < M; ++ i)
                {
                    for (unsigned j = i + 1; j < M; ++ j)
                    {
                        Dist(i, j) = (X.col(i) - X.col(j)).squaredNorm();
                    }
                }
                
                for (unsigned i = 0; i < M; ++ i)
                {
                    for (unsigned j = i + 1; j < M; ++ j)
                    {
                        if (!dirty && Dist(i, j) < eps_squared)
                        {
                            // Construct a mapping from the old indices to the new one
                            std::map<unsigned, unsigned> mapping;
                            unsigned new_index = 0;
                            for (unsigned old_index = 0; old_index < M; ++ old_index)
                            {
                                if (old_index != i && old_index != j)
                                {
                                    mapping[old_index] = new_index ++;
                                }
                            }
                            mapping[i] = M - 2;
                            mapping[j] = M - 2;
                            
                            // Update the matrix
                            MatrixXd newX(N, M - 1);
                            for (unsigned old_index = 0; old_index < M; ++ old_index)
                            {
                                newX.col(mapping[old_index]) = X.col(old_index);
                            }
                            newX.col(M - 2) = 0.5 * (X.col(i) + X.col(j));
                            X = newX;
                            
                            // Update the indices in the preference pairs
                            for (Preference& p : D)
                            {
                                for (unsigned i = 0; i < p.size(); ++ i) p[i] = mapping[p[i]];
                            }
                            
                            dirty = true;
                        }
                    }
                }
                if (!dirty) return;
            }
        }
    }
}
