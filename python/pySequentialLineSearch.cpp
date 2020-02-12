#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

    py::enum_<sequential_line_search::KernelType>(m, "KernelType", py::arithmetic())
        .value("ArdSquaredExponentialKernel", sequential_line_search::KernelType::ArdSquaredExponentialKernel)
        .value("ArdMatern52Kernel", sequential_line_search::KernelType::ArdMatern52Kernel);

    py::class_<SequentialLineSearchOptimizer> optimizer_class(m, "SequentialLineSearchOptimizer");

    optimizer_class.def(py::init<const int, const bool, const bool, const sequential_line_search::KernelType>(),
                        "num_dims"_a,
                        "use_slider_enlargement"_a = true,
                        "use_map_hyperparams"_a    = true,
                        "kernel_type"_a            = sequential_line_search::KernelType::ArdMatern52Kernel);

    optimizer_class.def(py::init<const int,
                                 const bool,
                                 const bool,
                                 const sequential_line_search::KernelType,
                                 const std::function<std::pair<Eigen::VectorXd, Eigen::VectorXd>(const int)>&>(),
                        "num_dims"_a,
                        "use_slider_enlargement"_a = true,
                        "use_map_hyperparams"_a    = true,
                        "kernel_type"_a            = sequential_line_search::KernelType::ArdMatern52Kernel,
                        "initial_slider_generator"_a);

    optimizer_class.def("set_hyperparams",
                        &SequentialLineSearchOptimizer::SetHyperparams,
                        "kernel_signal_var"_a            = 0.500,
                        "kernel_length_scale"_a          = 0.500,
                        "noise_level"_a                  = 0.005,
                        "kernel_hyperparams_prior_var"_a = 0.250,
                        "btl_scale"_a                    = 0.010);

    optimizer_class.def("submit_line_search_result",
                        static_cast<void (SequentialLineSearchOptimizer::*)(const double)>(
                            &SequentialLineSearchOptimizer::SubmitLineSearchResult),
                        "slider_position"_a);

    optimizer_class.def(
        "submit_line_search_result",
        static_cast<void (SequentialLineSearchOptimizer::*)(const double, const int, const int, const int)>(
            &SequentialLineSearchOptimizer::SubmitLineSearchResult),
        "slider_position"_a,
        "num_map_estimation_iters"_a,
        "num_global_search_iters"_a,
        "num_local_search_iters"_a);

    optimizer_class.def("get_slider_ends", &SequentialLineSearchOptimizer::GetSliderEnds);

    optimizer_class.def("calc_point_from_slider_position",
                        &SequentialLineSearchOptimizer::CalcPointFromSliderPosition,
                        "slider_position"_a);

    optimizer_class.def("get_maximizer", &SequentialLineSearchOptimizer::GetMaximizer);

    optimizer_class.def("get_preference_value_mean", &SequentialLineSearchOptimizer::GetPreferenceValueMean, "point"_a);

    optimizer_class.def(
        "get_preference_value_stdev", &SequentialLineSearchOptimizer::GetPreferenceValueStdev, "point"_a);

    optimizer_class.def(
        "get_acquisition_func_value", &SequentialLineSearchOptimizer::GetAcquisitionFuncValue, "point"_a);

    optimizer_class.def("get_raw_data_points", &SequentialLineSearchOptimizer::GetRawDataPoints);

    optimizer_class.def("damp_data", &SequentialLineSearchOptimizer::DampData, "directory_path"_a);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
