#include <Eigen/Core>
#include <memory>
#include <sequential-line-search/sequential-line-search.h>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// C API
///////////////////////////////////////////////////////////////////////////////

void                init(unsigned);
void                proceedOptimization(double);
std::vector<double> getParametersFromSlider(double);
std::vector<double> getXmax();

///////////////////////////////////////////////////////////////////////////////
// unnamed namespace for sealing
///////////////////////////////////////////////////////////////////////////////

using namespace sequential_line_search;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{

    std::shared_ptr<sequential_line_search::PreferenceRegressor> regressor;
    std::shared_ptr<sequential_line_search::Slider>              slider;

    sequential_line_search::Data data;

    unsigned        dimension;
    Eigen::VectorXd x_max;
    double          y_max;

    void clear()
    {
        regressor = nullptr;
        slider    = nullptr;

        data.X = MatrixXd::Zero(0, 0);
        data.D.clear();
        x_max = VectorXd::Zero(0);
        y_max = NAN;
    }

    std::vector<double> convertToSTL(const Eigen::VectorXd& vec_x_Eigen)
    {
        std::vector<double> vec_x_STL(vec_x_Eigen.rows());
        for (int i = 0; i < vec_x_Eigen.rows(); ++i)
        {
            vec_x_STL[i] = vec_x_Eigen(i);
        }
        return vec_x_STL;
    }

    void computeRegression() { regressor = std::make_shared<PreferenceRegressor>(data.X, data.D); }

    void updateSliderEnds()
    {
        // If this is the first time...
        if (x_max.rows() == 0)
        {
            slider = std::make_shared<Slider>(
                utils::generateRandomVector(dimension), utils::generateRandomVector(dimension), true);
            return;
        }

        const VectorXd x_1 = regressor->find_arg_max();
        const VectorXd x_2 = acquisition_function::FindNextPoint(*regressor);

        slider = std::make_shared<Slider>(x_1, x_2, true);
    }

    const VectorXd computeParametersFromSlider(double value)
    {
        return slider->end_0 * (1.0 - value) + slider->end_1 * value;
    }

} // namespace

///////////////////////////////////////////////////////////////////////////////
// C API
///////////////////////////////////////////////////////////////////////////////

void init(unsigned _dimension)
{
    dimension = _dimension;
    clear();
    computeRegression();
    updateSliderEnds();
}

void proceedOptimization(double slider_position)
{
    // Add new preference data
    const VectorXd x = computeParametersFromSlider(slider_position);
    data.AddNewPoints(x, {slider->orig_0, slider->orig_1});

    // Compute regression
    computeRegression();

    // Check the current best
    unsigned index;
    y_max = regressor->y.maxCoeff(&index);
    x_max = regressor->X.col(index);

    // Update slider ends
    updateSliderEnds();
}

std::vector<double> getParametersFromSlider(double value)
{
    return convertToSTL(slider->end_0 * (1.0 - value) + slider->end_1 * value);
}

std::vector<double> getXmax() { return convertToSTL(x_max); }

///////////////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////////////

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(pySequentialLineSearch, m)
{
    m.doc() = R"pbdoc(
    Sequential Line Search Plugin by Pybind11
    -----------------------------------------

    .. currentmodule:: SequentialLineSearch

    .. autosummary::
    :toctree: _generate
    init
    proceedOptimization
    getParametersFromSlider
    getXmax
    )pbdoc";

    m.def("init", &init, R"pbdoc(
          (Re)initializes this module.
          arg0: parameter dimension (int)
          )pbdoc");

    m.def("proceedOptimization", &proceedOptimization, R"pbdoc(
          Proceeds the optimization step by feeding the slider value.
          arg0: slider position (double)
          )pbdoc");

    m.def("getParametersFromSlider", &getParametersFromSlider, R"pbdoc(
          Gets the parameters as list, in the specific slider position.
          arg0: slider position (double)
          )pbdoc");

    m.def("getXmax", &getXmax, R"pbdoc(
          Gets the current best parameters as a list.
          )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
