#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sequential-line-search/preferential-bayesian-optimizer.hpp>
#include <sequential-line-search/sequential-line-search.hpp>

using sequential_line_search::PreferentialBayesianOptimizer;
using sequential_line_search::SequentialLineSearchOptimizer;
namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(pySequentialLineSearch, m)
{
    py::enum_<sequential_line_search::CurrentBestSelectionStrategy>(m, "CurrentBestSelectionStrategy", py::arithmetic())
        .value("LargestExpectValue", sequential_line_search::CurrentBestSelectionStrategy::LargestExpectValue)
        .value("LastSelection", sequential_line_search::CurrentBestSelectionStrategy::LastSelection);

    py::enum_<sequential_line_search::AcquisitionFuncType>(m, "AcquisitionFuncType", py::arithmetic())
        .value("ExpectedImprovement", sequential_line_search::AcquisitionFuncType::ExpectedImprovement)
        .value("GaussianProcessUpperConfidenceBound",
               sequential_line_search::AcquisitionFuncType::GaussianProcessUpperConfidenceBound);

    py::enum_<sequential_line_search::KernelType>(m, "KernelType", py::arithmetic())
        .value("ArdSquaredExponentialKernel", sequential_line_search::KernelType::ArdSquaredExponentialKernel)
        .value("ArdMatern52Kernel", sequential_line_search::KernelType::ArdMatern52Kernel);

    py::class_<SequentialLineSearchOptimizer> seq_opt_class(m, "SequentialLineSearchOptimizer");

    seq_opt_class.def(
        py::init<const int,
                 const bool,
                 const bool,
                 const sequential_line_search::KernelType,
                 const sequential_line_search::AcquisitionFuncType,
                 const std::function<std::pair<Eigen::VectorXd, Eigen::VectorXd>(const int)>&,
                 const sequential_line_search::CurrentBestSelectionStrategy>(),
        "num_dims"_a,
        "use_slider_enlargement"_a  = true,
        "use_map_hyperparams"_a     = true,
        "kernel_type"_a             = sequential_line_search::KernelType::ArdMatern52Kernel,
        "acquisition_func_type"_a   = sequential_line_search::AcquisitionFuncType::ExpectedImprovement,
        "initial_query_generator"_a = std::function<std::pair<Eigen::VectorXd, Eigen::VectorXd>(const int)>(
            sequential_line_search::GenerateRandomSliderEnds),
        "current_best_selection_strategy"_a = sequential_line_search::CurrentBestSelectionStrategy::LargestExpectValue);

    seq_opt_class.def("set_hyperparams",
                      &SequentialLineSearchOptimizer::SetHyperparams,
                      "kernel_signal_var"_a            = 0.500,
                      "kernel_length_scale"_a          = 0.500,
                      "noise_level"_a                  = 0.005,
                      "kernel_hyperparams_prior_var"_a = 0.250,
                      "btl_scale"_a                    = 0.010);

    seq_opt_class.def("submit_feedback_data",
                      static_cast<void (SequentialLineSearchOptimizer::*)(const double)>(
                          &SequentialLineSearchOptimizer::SubmitFeedbackData),
                      "slider_position"_a);

    seq_opt_class.def(
        "submit_feedback_data",
        static_cast<void (SequentialLineSearchOptimizer::*)(const double, const int, const int, const int)>(
            &SequentialLineSearchOptimizer::SubmitFeedbackData),
        "slider_position"_a,
        "num_map_estimation_iters"_a,
        "num_global_search_iters"_a,
        "num_local_search_iters"_a);

    seq_opt_class.def("get_slider_ends", &SequentialLineSearchOptimizer::GetSliderEnds);

    seq_opt_class.def("calc_point_from_slider_position",
                      &SequentialLineSearchOptimizer::CalcPointFromSliderPosition,
                      "slider_position"_a);

    seq_opt_class.def("get_maximizer", &SequentialLineSearchOptimizer::GetMaximizer);

    seq_opt_class.def("get_preference_value_mean", &SequentialLineSearchOptimizer::GetPreferenceValueMean, "point"_a);

    seq_opt_class.def("get_preference_value_stdev", &SequentialLineSearchOptimizer::GetPreferenceValueStdev, "point"_a);

    seq_opt_class.def("get_acquisition_func_value", &SequentialLineSearchOptimizer::GetAcquisitionFuncValue, "point"_a);

    seq_opt_class.def("get_raw_data_points", &SequentialLineSearchOptimizer::GetRawDataPoints);

    seq_opt_class.def("damp_data", &SequentialLineSearchOptimizer::DampData, "directory_path"_a);

    seq_opt_class.def("set_gaussian_process_upper_confidence_bound_hyperparam",
                      &SequentialLineSearchOptimizer::SetGaussianProcessUpperConfidenceBoundHyperparam,
                      "hyperparam"_a);

    py::class_<PreferentialBayesianOptimizer> pref_opt_class(m, "PreferentialBayesianOptimizer");

    pref_opt_class.def(
        py::init<const int,
                 const bool,
                 const sequential_line_search::KernelType,
                 const sequential_line_search::AcquisitionFuncType,
                 const std::function<std::vector<Eigen::VectorXd>(const int)>&,
                 const sequential_line_search::CurrentBestSelectionStrategy,
                 const int>(),
        "num_dims"_a,
        "use_map_hyperparams"_a   = true,
        "kernel_type"_a           = sequential_line_search::KernelType::ArdMatern52Kernel,
        "acquisition_func_type"_a = sequential_line_search::AcquisitionFuncType::ExpectedImprovement,
        "initial_query_generator"_a =
            std::function<std::vector<Eigen::VectorXd>(const int)>(sequential_line_search::GenerateRandomPoints),
        "current_best_selection_strategy"_a = sequential_line_search::CurrentBestSelectionStrategy::LargestExpectValue,
        "num_options"_a                     = 2);

    pref_opt_class.def("set_hyperparams",
                       &PreferentialBayesianOptimizer::SetHyperparams,
                       "kernel_signal_var"_a            = 0.500,
                       "kernel_length_scale"_a          = 0.500,
                       "noise_level"_a                  = 0.005,
                       "kernel_hyperparams_prior_var"_a = 0.250,
                       "btl_scale"_a                    = 0.010);

    pref_opt_class.def("submit_feedback_data",
                       &PreferentialBayesianOptimizer::SubmitFeedbackData,
                       "option_index"_a,
                       "num_map_estimation_iters"_a = 0);

    pref_opt_class.def("submit_custom_feedback_data",
                       &PreferentialBayesianOptimizer::SubmitCustomFeedbackData,
                       "chosen_option"_a,
                       "other_options"_a,
                       "num_map_estimation_iters"_a = 0);

    pref_opt_class.def("determine_next_query",
                       &PreferentialBayesianOptimizer::DetermineNextQuery,
                       "num_global_search_iters"_a = 0,
                       "num_local_search_iters"_a  = 0);

    pref_opt_class.def("get_current_options", &PreferentialBayesianOptimizer::GetCurrentOptions);

    pref_opt_class.def("get_maximizer", &PreferentialBayesianOptimizer::GetMaximizer);

    pref_opt_class.def("get_preference_value_mean", &PreferentialBayesianOptimizer::GetPreferenceValueMean, "point"_a);

    pref_opt_class.def(
        "get_preference_value_stdev", &PreferentialBayesianOptimizer::GetPreferenceValueStdev, "point"_a);

    pref_opt_class.def(
        "get_acquisition_func_value", &PreferentialBayesianOptimizer::GetAcquisitionFuncValue, "point"_a);

    pref_opt_class.def("get_raw_data_points", &PreferentialBayesianOptimizer::GetRawDataPoints);

    pref_opt_class.def("damp_data", &PreferentialBayesianOptimizer::DampData, "directory_path"_a);

    pref_opt_class.def("set_gaussian_process_upper_confidence_bound_hyperparam",
                       &PreferentialBayesianOptimizer::SetGaussianProcessUpperConfidenceBoundHyperparam,
                       "hyperparam"_a);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
