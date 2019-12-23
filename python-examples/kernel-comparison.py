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
length_scale_set = [
    0.10,
    0.50,
    # 1.00,
]

num_dims_set = [
    5,
    # 10,
    15,
]

kernel_type_set = [
    pySequentialLineSearch.KernelType.ArdSquaredExponentialKernel,
    pySequentialLineSearch.KernelType.ArdMatern52Kernel,
]

condition_pattern = itertools.product(range(len(length_scale_set)),
                                      range(len(kernel_type_set)),
                                      range(len(num_dims_set)))

# Define constant values
NUM_TRIALS = 3  # 10
NUM_ITERS = 15  # 30
USE_MAP_HYPERPARAMS = False
OUTPUT_NAME = "kernel-comparison"

# Instantiate a multi-dimensional array for storing results
optimality_gaps = np.ndarray(shape=(NUM_ITERS,
                                    NUM_TRIALS, len(length_scale_set),
                                    len(kernel_type_set), len(num_dims_set)))

# Perform sequential line search procedures with various conditions
for condition_index, condition in enumerate(condition_pattern):

    length_scale_index = condition[0]
    length_scale = length_scale_set[length_scale_index]

    kernel_type_index = condition[1]
    kernel_type = kernel_type_set[kernel_type_index]

    num_dims_index = condition[2]
    num_dims = num_dims_set[num_dims_index]

    print("Condition No. {}:".format(condition_index + 1))
    print("\tNo. dimensions = {}".format(num_dims))
    print("\tKernel: {}".format(kernel_type))
    print("\tKernel length scale = {}".format(length_scale))

    def perform_sequential_line_search(trial_index: int) -> np.ndarray:

        optimality_gaps = np.ndarray(NUM_ITERS, )

        optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(
            num_dims=num_dims,
            use_map_hyperparams=USE_MAP_HYPERPARAMS,
            kernel_type=kernel_type,
            initial_slider_generator=generate_initial_slider)

        optimizer.set_hyperparams(kernel_signal_var=0.50,
                                  kernel_length_scale=length_scale,
                                  kernel_hyperparams_prior_var=0.10)

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
        optimality_gaps[:, trial_index, length_scale_index, kernel_type_index,
                        num_dims_index] = trial_result

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
num_cols = len(length_scale_set)
num_rows = len(num_dims_set)

FIG_SIZE = (num_cols * 4, num_rows * 4)
DPI = 200

fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

# Plot data
for num_dims_index, num_dims in enumerate(num_dims_set):
    ref_axes = None
    for length_scale_index, length_scale in enumerate(length_scale_set):
        axes = fig.add_subplot(num_rows,
                               num_cols,
                               num_dims_index * num_cols + length_scale_index +
                               1,
                               sharey=ref_axes)
        ref_axes = axes

        for kernel_type_index, kernel_type in enumerate(kernel_type_set):

            label = kernel_type.name

            title_dim = str(num_dims) + "D"
            title_length_scale = r"$\theta_\text{length-scale} = " + str(
                length_scale) + r"$"
            title = title_dim + " / " + title_length_scale

            plot_mean_with_errors(
                axes=axes,
                data=optimality_gaps[:, :, length_scale_index,
                                     kernel_type_index, num_dims_index],
                label=label)

            axes.set_title(title)
            axes.legend()
            axes.set_xlabel(r"\#iterations")
            axes.set_ylabel(r"Optimality gap")

# Export figures
fig.tight_layout()
plt.savefig("./" + OUTPUT_NAME + ".pdf")
plt.savefig("./" + OUTPUT_NAME + ".png")
