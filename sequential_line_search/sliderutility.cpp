#include "sliderutility.h"
#include <cmath>
#include <map>
#include <memory>
#include "nloptutility.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::pair;
using std::vector;

namespace SliderUtility
{

struct Data
{
    Data(const VectorXd& c, const VectorXd& r, const double s) : c(c), r(r), s(s) {}
    VectorXd c;
    VectorXd r;
    double   s;
};

inline double crop(double x)
{
    double epsilon = 1e-16;
    return (x > epsilon) ? ((x < 1.0 - epsilon) ? (x) : (1.0 - epsilon)) : epsilon;
}

inline VectorXd crop(const VectorXd& x)
{
    VectorXd y(x.rows());
    for (unsigned i = 0; i < x.rows(); ++ i) y(i) = crop(x(i));
    return y;
}

double objective(const std::vector<double> &x, std::vector<double>& /*grad*/, void* data)
{
    const double target = static_cast<Data*>(data)->s;
    return - (x[0] - target) * (x[0] - target);
}

double constraint_p(const std::vector<double> &x, std::vector<double>& /*grad*/, void* data)
{
    const VectorXd& c = static_cast<Data*>(data)->c;
    const VectorXd& r = static_cast<Data*>(data)->r;
    const double&   t = x[0];

    const VectorXd y = c + t * r;

    double sum = 0.0;
    for (unsigned i = 0; i < y.rows(); ++ i)
    {
        sum += (crop(y(i)) - y(i)) * (crop(y(i)) - y(i));
    }
    return sum;
}

double constraint_n(const std::vector<double> &x, std::vector<double>& /*grad*/, void* data)
{
    const VectorXd& c = static_cast<Data*>(data)->c;
    const VectorXd& r = static_cast<Data*>(data)->r;
    const double&   t = x[0];

    const VectorXd y = c - t * r;

    double sum = 0.0;
    for (unsigned i = 0; i < y.rows(); ++ i)
    {
        sum += (crop(y(i)) - y(i)) * (crop(y(i)) - y(i));
    }
    return sum;
}

pair<VectorXd, VectorXd> enlargeSliderEnds(const VectorXd& x_1, const VectorXd& x_2, const double scale, const double minimum_length)
{
    const VectorXd c = 0.5 * (crop(x_1) + crop(x_2));
    const VectorXd r = crop(x_1) - c;

    auto data = std::make_shared<Data>(c, r, scale);

    const double t_1 = nloptutil::solve_with_constraints(VectorXd::Constant(1, 1.0), VectorXd::Constant(1, scale), VectorXd::Constant(1, 0.0), objective, constraint_p, data.get(), nlopt::LN_COBYLA, 1000)(0);
    const double t_2 = nloptutil::solve_with_constraints(VectorXd::Constant(1, 1.0), VectorXd::Constant(1, scale), VectorXd::Constant(1, 0.0), objective, constraint_n, data.get(), nlopt::LN_COBYLA, 1000)(0);

    const VectorXd x_1_new = crop(c + t_1 * r);
    const VectorXd x_2_new = crop(c - t_2 * r);

    const double length = (x_1_new - x_2_new).norm();

    if (length < minimum_length)
    {
        if (std::abs(t_1 - t_2) < 1e-10)
        {
            return pair<VectorXd, VectorXd>(c + (minimum_length / length) * t_1 * r, c - (minimum_length / length) * t_2 * r);
        }
        else if (t_1 > t_2)
        {
            return pair<VectorXd, VectorXd>(c + 2.0 * (minimum_length / length) * t_1 * r, c - t_2 * r);
        }
        else
        {
            return pair<VectorXd, VectorXd>(c + t_1 * r, c - 2.0 * (minimum_length / length) * t_2 * r);
        }
    }
    return pair<VectorXd, VectorXd>(x_1_new, x_2_new);
}

void mergeData(Eigen::MatrixXd &X, std::vector<Preference> &D, const double epsilon)
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
