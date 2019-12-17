#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <sequential-line-search/sequential-line-search.hpp>

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
                        "num_dims"_a,
                        "use_slider_enlargement"_a = false,
                        "use_map_hyperparams"_a    = true);

    optimizer_class.def("set_hyperparams",
                        &SequentialLineSearchOptimizer::SetHyperparams,
                        "kernel_signal_var"_a            = 0.500,
                        "kernel_length_scale"_a          = 0.500,
                        "noise_level"_a                  = 0.005,
                        "kernel_hyperparams_prior_var"_a = 0.250,
                        "btl_scale"_a                    = 0.010);

    optimizer_class.def(
        "submit_line_search_result", &SequentialLineSearchOptimizer::SubmitLineSearchResult, "slider_position"_a);

    optimizer_class.def("get_slider_ends", &SequentialLineSearchOptimizer::GetSliderEnds);

    optimizer_class.def("calc_point_from_slider_position",
                        &SequentialLineSearchOptimizer::CalcPointFromSliderPosition,
                        "slider_position"_a);

    optimizer_class.def("get_maximizer", &SequentialLineSearchOptimizer::GetMaximizer);

    optimizer_class.def("get_preference_value_mean", &SequentialLineSearchOptimizer::GetPreferenceValueMean, "point"_a);

    optimizer_class.def(
        "get_preference_value_stdev", &SequentialLineSearchOptimizer::GetPreferenceValueStdev, "point"_a);

    optimizer_class.def(
        "get_expected_improvement_value", &SequentialLineSearchOptimizer::GetExpectedImprovementValue, "point"_a);

    optimizer_class.def("get_raw_data_points", &SequentialLineSearchOptimizer::GetRawDataPoints);

    optimizer_class.def("damp_data", &SequentialLineSearchOptimizer::DampData, "directory_path"_a);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
