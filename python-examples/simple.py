import pySequentialLineSearch
import numpy as np
import sys
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


# A typical implementation of sequential line search procedure
def main():
    optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(
        num_dims=5)

    optimizer.set_hyperparams(kernel_signal_var=0.50,
                              kernel_length_scale=0.10,
                              kernel_hyperparams_prior_var=0.10)

    for i in range(30):
        slider_ends = optimizer.get_slider_ends()
        slider_position = ask_human_for_slider_manipulation(slider_ends)
        optimizer.submit_feedback_data(slider_position)

        residual = np.linalg.norm(optimizer.get_maximizer() - 0.2)
        print("[#iters = " + str(i + 1) + "] residual: " + str(residual))


if __name__ == '__main__':
    main()
