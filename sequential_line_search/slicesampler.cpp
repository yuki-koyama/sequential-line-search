#include "slicesampler.h"
#include "utility.h"

using Eigen::VectorXd;

namespace SliceSampler
{

VectorXd sampling(Func func, const void* data, const VectorXd &x_last, const Eigen::VectorXd &bracket_size)
{
    const unsigned dim = x_last.rows();
    const double   y   = Utility::generateUniformReal() * func(x_last, data);

    const VectorXd v = bracket_size.cwiseProduct(Utility::generateRandomVector(dim));

    // Set initial hyperrectangle
    VectorXd l = x_last - v;
    VectorXd u = x_last + bracket_size;

    VectorXd x_next;
    while (true)
    {
        const VectorXd x_cand = l + (u - l).cwiseProduct(Utility::generateRandomVector(dim));
        const double   y_cand = func(x_cand, data);

        if (y_cand > y)
        {
            x_next = x_cand;
            break;
        }
        else
        {
            // Narrow hyperrectangle
            for (unsigned i = 0; i < dim; ++ i)
            {
                if (x_last(i) < x_cand(i))
                {
                    u(i) = x_cand(i);
                }
                else
                {
                    l(i) = x_cand(i);
                }
            }
        }
    }
    return x_next;
}

}
