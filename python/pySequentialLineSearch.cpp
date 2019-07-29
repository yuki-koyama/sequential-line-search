#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <sequential-line-search/sequential-line-search.h>

using sequential_line_search::SequentialLineSearchOptimizer;
namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(pySequentialLineSearch, m)
{
    m.doc() = R"pbdoc(
    Sequential Line Search Python Bindings
    -----------------------------------------

    .. currentmodule:: pySequentialLineSearch

    .. autosummary::
    :toctree: _generate

    SequentialLineSearchOptimizer
    )pbdoc";

    py::class_<SequentialLineSearchOptimizer> optimizer_class(m, "SequentialLineSearchOptimizer");

    optimizer_class.def(py::init<const int, const bool, const bool>(),
                        "dimension"_a,
                        "use_slider_enlargement"_a = true,
                        "use_map_parameters"_a     = true);

    optimizer_class.def("set_hyperparameters",
                        &SequentialLineSearchOptimizer::SetHyperparameters,
                        "a"_a,
                        "r"_a,
                        "b"_a,
                        "variance"_a,
                        "btl_scale"_a);

    optimizer_class.def("submit_line_search_result", &SequentialLineSearchOptimizer::SubmitLineSearchResult, "slider_position"_a);

    optimizer_class.def("get_slider_ends", &SequentialLineSearchOptimizer::GetSliderEnds);

    optimizer_class.def("get_parameters", &SequentialLineSearchOptimizer::GetParameters, "slider_position"_a);

    optimizer_class.def("get_maximizer", &SequentialLineSearchOptimizer::GetMaximizer);

    optimizer_class.def(
        "get_preference_value_mean", &SequentialLineSearchOptimizer::GetPreferenceValueMean, "parameter"_a);

    optimizer_class.def("get_preference_value_standard_deviation",
                        &SequentialLineSearchOptimizer::GetPreferenceValueStandardDeviation,
                        "parameter"_a);

    optimizer_class.def(
        "get_expected_improvement_value", &SequentialLineSearchOptimizer::GetExpectedImprovementValue, "parameter"_a);

    optimizer_class.def("get_raw_data_points", &SequentialLineSearchOptimizer::GetRawDataPoints);

    optimizer_class.def("damp_data", &SequentialLineSearchOptimizer::DampData, "directory_path"_a);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
