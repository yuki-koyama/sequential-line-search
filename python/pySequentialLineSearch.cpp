#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <sequential-line-search/sequential-line-search.h>

using sequential_line_search::SequentialLineSearchOptimizer;

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

    pybind11::class_<SequentialLineSearchOptimizer> optimizer_class(m, "SequentialLineSearchOptimizer");

    optimizer_class.def(pybind11::init<const int, const bool, const bool>());

    optimizer_class.def("set_hyperparameters", &SequentialLineSearchOptimizer::setHyperparameters);

    optimizer_class.def("submit", &SequentialLineSearchOptimizer::submit);

    optimizer_class.def("get_slider_ends", &SequentialLineSearchOptimizer::getSliderEnds);

    optimizer_class.def("get_parameters", &SequentialLineSearchOptimizer::getParameters);

    optimizer_class.def("get_maximizer", &SequentialLineSearchOptimizer::getMaximizer);

    optimizer_class.def("get_preference_value_mean", &SequentialLineSearchOptimizer::getPreferenceValueMean);

    optimizer_class.def("get_preference_value_standard_deviation",
                        &SequentialLineSearchOptimizer::getPreferenceValueStandardDeviation);

    optimizer_class.def("get_expected_improvement_value", &SequentialLineSearchOptimizer::getExpectedImprovementValue);

    optimizer_class.def("get_raw_data_points", &SequentialLineSearchOptimizer::getRawDataPoints);

    optimizer_class.def("damp_adata", &SequentialLineSearchOptimizer::dampData);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
