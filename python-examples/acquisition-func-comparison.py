import pySequentialLineSearch
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from joblib import Parallel, delayed
from typing import Optional, Tuple


# A dummy function for testing
def calc_simulated_objective_func(x: np.ndarray) -> float:
    return -np.linalg.norm(x - 0.2)


# A dummy implementation of slider manipulation
def ask_human_for_slider_manipulation(
        slider_ends: Tuple[np.ndarray, np.ndarray]) -> float:
    t_max = 0.0
    f_max = -sys.float_info.max

    for i in range(1000):
        t = float(i) / 999.0
        x = (1.0 - t) * slider_ends[0] + t * slider_ends[1]
        f = calc_simulated_objective_func(x)

        if f_max is None or f_max < f:
            f_max = f
            t_max = t

    return t_max


# A custom generator of a slider for the first iteration
def generate_initial_slider(num_dims: int) -> Tuple[np.ndarray, np.ndarray]:
    end_0 = np.random.uniform(low=0.0, high=1.0, size=(num_dims, ))
    end_1 = np.random.uniform(low=0.0, high=1.0, size=(num_dims, ))
    return end_0, end_1


# A helper function to plot a mean and confidence interval of multiple sequences
def plot_mean_with_errors(axes, data: np.ndarray,
                          label: Optional[str] = None) -> None:
    CONFIDENT_REGION_ALPHA = 0.2

    mean = np.mean(data, axis=1)
    lower = mean - np.std(data, axis=1)
    upper = mean + np.std(data, axis=1)
    axes.plot(mean, label=label)
    axes.fill_between(range(data.shape[0]),
                      lower,
                      upper,
                      alpha=CONFIDENT_REGION_ALPHA)


# Define conditions to be compared
acquisition_func_type_set = [
    (pySequentialLineSearch.AcquisitionFuncType.ExpectedImprovement, 0.0,
     r"Expected Improvedment"),
    (pySequentialLineSearch.AcquisitionFuncType.
     GaussianProcessUpperConfidenceBound, 0.2, r"GP-UCB ($\beta = 0.2$)"),
    (pySequentialLineSearch.AcquisitionFuncType.
     GaussianProcessUpperConfidenceBound, 1.0, r"GP-UCB ($\beta = 1.0$)"),
    (pySequentialLineSearch.AcquisitionFuncType.
     GaussianProcessUpperConfidenceBound, 5.0, r"GP-UCB ($\beta = 5.0$)"),
]

# Define constant values
NUM_TRIALS = 10  # 30
NUM_ITERS = 15  # 30
NUM_DIMS = 10
USE_MAP_HYPERPARAMS = False
OUTPUT_NAME = "acquisition-func-comparison"

# Instantiate a multi-dimensional array for storing results
optimality_gaps = np.ndarray(shape=(NUM_ITERS, NUM_TRIALS,
                                    len(acquisition_func_type_set)))

# Perform sequential line search procedures with various conditions
for condition_index, acquisition_func_type in enumerate(
        acquisition_func_type_set):

    num_dims = NUM_DIMS

    print("Condition No. {}:".format(condition_index + 1))

    def perform_sequential_line_search(trial_index: int) -> np.ndarray:

        optimality_gaps = np.ndarray(NUM_ITERS, )

        # Instantiate the optimizer with acquisition function specification
        optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(
            num_dims=num_dims,
            use_map_hyperparams=USE_MAP_HYPERPARAMS,
            acquisition_func_type=acquisition_func_type[0],
            initial_slider_generator=generate_initial_slider)

        # Specify a hyperparameter in case the acquisition function is GP-UCB
        optimizer.set_gaussian_process_upper_confidence_bound_hyperparam(
            acquisition_func_type[1])

        for i in range(NUM_ITERS):
            slider_ends = optimizer.get_slider_ends()
            slider_position = ask_human_for_slider_manipulation(slider_ends)
            optimizer.submit_line_search_result(slider_position)

            optimality_gap = -calc_simulated_objective_func(
                optimizer.get_maximizer())

            optimality_gaps[i] = optimality_gap

        return optimality_gaps

    results = Parallel(n_jobs=-1)([
        delayed(perform_sequential_line_search)(i) for i in range(NUM_TRIALS)
    ])

    for trial_index, trial_result in enumerate(results):
        optimality_gaps[:, trial_index, condition_index] = trial_result

# Export data
np.save("./" + OUTPUT_NAME + ".npy", optimality_gaps)

# Set up the plot design
sns.set()
sns.set_context()

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [
    r"\usepackage{libertine}",
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
]

# Instantiate a figure object
FIG_SIZE = (4, 4)
DPI = 200

fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

# Plot data
axes = fig.add_subplot(1, 1, 1)

for index, acquisition_func_type in enumerate(acquisition_func_type_set):
    label = acquisition_func_type[2]

    plot_mean_with_errors(axes=axes,
                          data=optimality_gaps[:, :, index],
                          label=label)

axes.set_title("Acquisition Function Comparison")
axes.legend()
axes.set_xlabel(r"\#iterations")
axes.set_ylabel(r"Optimality gap")

# Export figures
fig.tight_layout()
plt.savefig("./" + OUTPUT_NAME + ".pdf")
plt.savefig("./" + OUTPUT_NAME + ".png")
