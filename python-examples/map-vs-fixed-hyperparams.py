import pySequentialLineSearch
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
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
def plot_mean_with_errors(axes,
                          data: np.ndarray,
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

map_condition_set = [
    {
        "use_map_hyperparams": False,
        "kernel_hyperparams_prior_var": 0.00
    },
    {
        "use_map_hyperparams": True,
        "kernel_hyperparams_prior_var": 0.10
    },
    # {
    #     "use_map_hyperparams": True,
    #     "kernel_hyperparams_prior_var": 0.25
    # },
]

# Define constant values
NUM_TRIALS = 3  # 10
NUM_ITERS = 15  # 30
KERNEL_TYPE = pySequentialLineSearch.KernelType.ArdMatern52Kernel

# Instantiate a multi-dimensional array for storing results
optimality_gaps = np.ndarray(shape=(NUM_ITERS, NUM_TRIALS,
                                    len(length_scale_set),
                                    len(map_condition_set), len(num_dims_set)))

# Perform sequential line search procedures with various conditions
for index_num_dims, num_dims in enumerate(num_dims_set):
    print("Testing on a {}-dimensional function...".format(num_dims))
    for index_map_condition, map_condition in enumerate(map_condition_set):
        print("\t" + "Hyperparameters condition: " + str(map_condition))
        for index_length_scale, length_scale in enumerate(length_scale_set):
            print("\t\t" + "Kernel length scale: " + str(length_scale))
            for trial in range(NUM_TRIALS):

                optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(
                    num_dims=num_dims,
                    use_map_hyperparams=map_condition["use_map_hyperparams"],
                    kernel_type=KERNEL_TYPE,
                    initial_query_generator=generate_initial_slider)

                optimizer.set_hyperparams(
                    kernel_signal_var=0.50,
                    kernel_length_scale=length_scale,
                    kernel_hyperparams_prior_var=map_condition[
                        "kernel_hyperparams_prior_var"])

                for i in range(NUM_ITERS):
                    slider_ends = optimizer.get_slider_ends()
                    slider_position = ask_human_for_slider_manipulation(
                        slider_ends)
                    optimizer.submit_feedback_data(slider_position)

                    optimality_gap = -calc_simulated_objective_func(
                        optimizer.get_maximizer())
                    optimality_gaps[i, trial, index_length_scale,
                                    index_map_condition,
                                    index_num_dims] = optimality_gap

# Set up the plot design
sns.set()
sns.set_context()

plt.rcParams['text.usetex'] = True
plt.rcParams[
    'text.latex.preamble'] = r"\usepackage{libertine} \usepackage{amsmath} \usepackage{amssymb}"

# Instantiate a figure object
num_cols = len(map_condition_set)
num_rows = len(num_dims_set)

FIG_SIZE = (num_cols * 4, num_rows * 4)
DPI = 200

fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

# Plot data
for index_num_dims, num_dims in enumerate(num_dims_set):
    ref_axes = None
    for index_map_condition, map_condition in enumerate(map_condition_set):
        axes = fig.add_subplot(num_rows,
                               num_cols,
                               index_num_dims * num_cols +
                               index_map_condition + 1,
                               sharey=ref_axes)
        ref_axes = axes

        for index_length_scale, length_scale in enumerate(length_scale_set):

            label = r"$\theta_\text{length-scale} = " + str(
                length_scale) + r"$"

            title_dim = str(num_dims) + "D"
            title_map = r"MAP ($\sigma^{2} = " + str(
                map_condition["kernel_hyperparams_prior_var"]
            ) + r"$)" if map_condition["use_map_hyperparams"] else "Fixed"
            title = title_dim + " / " + title_map

            plot_mean_with_errors(axes=axes,
                                  data=optimality_gaps[:, :,
                                                       index_length_scale,
                                                       index_map_condition,
                                                       index_num_dims],
                                  label=label)

            axes.set_title(title)
            axes.legend()
            axes.set_xlabel(r"\#iterations")
            axes.set_ylabel(r"Optimality gap")

# Export figures
fig.tight_layout()
plt.savefig("./map-vs-fixed-hyperparams.pdf")
plt.savefig("./map-vs-fixed-hyperparams.png")
