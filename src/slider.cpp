#include <memory>
#include <nlopt-util.hpp>
#include <sequential-line-search/slider.hpp>

namespace
{
    using Eigen::VectorXd;
    using std::pair;

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
        for (unsigned i = 0; i < x.rows(); ++i)
            y(i) = crop(x(i));
        return y;
    }

    double objective(const std::vector<double>& x, std::vector<double>& /*grad*/, void* data)
    {
        const double target = static_cast<Data*>(data)->s;
        return -(x[0] - target) * (x[0] - target);
    }

    double constraint_p(const std::vector<double>& x, std::vector<double>& /*grad*/, void* data)
    {
        const VectorXd& c = static_cast<Data*>(data)->c;
        const VectorXd& r = static_cast<Data*>(data)->r;
        const double&   t = x[0];

        const VectorXd y = c + t * r;

        double sum = 0.0;
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            sum += (crop(y(i)) - y(i)) * (crop(y(i)) - y(i));
        }
        return sum;
    }

    double constraint_n(const std::vector<double>& x, std::vector<double>& /*grad*/, void* data)
    {
        const VectorXd& c = static_cast<Data*>(data)->c;
        const VectorXd& r = static_cast<Data*>(data)->r;
        const double&   t = x[0];

        const VectorXd y = c - t * r;

        double sum = 0.0;
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            sum += (crop(y(i)) - y(i)) * (crop(y(i)) - y(i));
        }
        return sum;
    }

    /// \brief Given a pair of slider ends, this function enlarges the slider by a specified scale but considering the
    /// bounding box constraint.
    pair<VectorXd, VectorXd>
    EnlargeSliderEnds(const VectorXd& x_1, const VectorXd& x_2, const double scale, const double minimum_length)
    {
        const VectorXd c = 0.5 * (crop(x_1) + crop(x_2));
        const VectorXd r = crop(x_1) - c;

        auto data = std::make_shared<Data>(c, r, scale);

        const double t_1 = nloptutil::solve(VectorXd::Constant(1, 1.0),
                                            VectorXd::Constant(1, scale),
                                            VectorXd::Constant(1, 0.0),
                                            objective,
                                            {},
                                            {constraint_p},
                                            nlopt::LN_COBYLA,
                                            data.get(),
                                            1000)(0);
        const double t_2 = nloptutil::solve(VectorXd::Constant(1, 1.0),
                                            VectorXd::Constant(1, scale),
                                            VectorXd::Constant(1, 0.0),
                                            objective,
                                            {},
                                            {constraint_n},
                                            nlopt::LN_COBYLA,
                                            data.get(),
                                            1000)(0);

        const VectorXd x_1_new = crop(c + t_1 * r);
        const VectorXd x_2_new = crop(c - t_2 * r);

        const double length = (x_1_new - x_2_new).norm();

        if (length < minimum_length)
        {
            if (std::abs(t_1 - t_2) < 1e-10)
            {
                return pair<VectorXd, VectorXd>(c + (minimum_length / length) * t_1 * r,
                                                c - (minimum_length / length) * t_2 * r);
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
} // namespace

sequential_line_search::Slider::Slider(const Eigen::VectorXd& end_0,
                                       const Eigen::VectorXd& end_1,
                                       const bool             enlarge,
                                       const double           scale,
                                       const double           minimum_length)
    : original_end_0(end_0), original_end_1(end_1)
{
    if (enlarge)
    {
        const auto ends = EnlargeSliderEnds(original_end_0, original_end_1, scale, minimum_length);
        this->end_0     = std::get<0>(ends);
        this->end_1     = std::get<1>(ends);
    }
    else
    {
        this->end_0 = original_end_0;
        this->end_1 = original_end_1;
    }
}
