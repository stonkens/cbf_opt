"""Defines a gneeric experiment that can be extended to any type of controls problem.

"Experiment" is anything that tests the behavior of a controller, and should only be limited to a single function, e.g., simulating a rollout or plotting the Lyapunov function on a grid



Each experiment should do the following:
    1. Run the experiment on a given controller
    2. Save the results of that experiment to a CSV (tidy data principle) https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html
    3. Plot the results of the experiment and return the plot handle, with an option of displaying the plot
    """

from abc import ABCMeta, abstractmethod
import numpy as np

from typing import List, Tuple
import pandas as pd
from matplotlib.pyplot import figure


class Experiment(metaclass=ABCMeta):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def run(self, controller: "Controller"):
        """
        Run the experiment on a given controller

        returns:
            a pandas Dataframe containing the result of the experiment (each row corresponding to a single observation from the experiment)
        """
        pass
    
    @abstractmethod
    def plot(self, controller: "Controller", display_plots: bool = False) -> List[Tuple[str, figure]]:
        """[summary]

        Args:
            controller (Controller): the controller used to run the experiment
            display_plots (bool, optional): If True, display the plots (block until user responds). Defaults to False.

        Returns:
            List[Tuple[str, figure]]: Contains name of each figures and the figure object
        """
        pass

    def run_and_save_to_csv(self, controller: "Controller", save_dir: str):
        import os
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{self.name}.csv"

        results = self.run(controller)
        results.to_csv(filename, index=False)

    def run_and_plot(self, controller: "Controller", display_plots: bool = False):
        results_df = self.run(controller)
        return self.plot(controller, results_df, display_plots)

class Controller(ABCMeta):
    def __init__(
        self, model: Dynamics, experiment_suite: ExperimentSuite, sampling_rate: float = 0.01
    ):
        super().__init__()
        self.model = model
        self.experiment_suite = experiment_suite
        self.sampling_rate = sampling_rate

    @abstractmethod
    def __self__(self, x: np.ndarray, time: float = 0.0):
        """Run the experiment on a given controller"""
        pass


class PIDController(Controller):
    def __init__(
        self,
        model: Dynamics,
        experiment_suite: ExperimentSuite,
        sampling_rate: float = 0.01,
        K_p: np.ndarray,
        K_i: np.ndarray,
        K_d: np.ndarray,
        target: np.ndarray
    ):
        super().__init__(model, experiment_suite, sampling_rate)
        self.K_p = np.atleast_2d(K_p)
        self.K_i = np.atleast_2d(K_i)
        self.K_d = np.atleast_2d(K_d)

        for K in [self.K_p, self.K_i, self.K_d]:
            assert K.shape == (self.model.n_dims, self.model.control_dims)
        self.sum_error_time = np.zeros(shape=(model.n_dims,))
        self.prev_error = np.zeros(shape=(model.n_dims,))

        self.target = target
        assert self.target.shape == (model.n_dims,)

    def __call__(self, x: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Run the experiment on a given controller"""
        error = self.target - x
        self.sum_error_time += error * self.sampling_rate
        
        u = self.K_p @ error + 1. / self.K_i @ self.sum_error_time + self.K_d / self.sampling_rate @ (error - self.prev_error)

        # TODO: Correct to include a target model (e.g. z = h(x))

        self.prev_error = error
        return np.atleast_1d(u)


#TODO: figure out how to have an argument that is either a torch.Tensor or a np.ndarray
class Rollout2DStateSpaceExperiment(Experiment):
    def __init__(self, name: str, start_x: np.ndarray, plot_x_index: int, plot_x_label: str, plot_y_index: int, plot_y_label: str, scenarios: Optional[ScenarioList] = None, n_sims_per_start: int = 5, t_sim: float = 5.0):
        super().__init__(name)
        self.start_x = start_x
        self.plot_x_index = plot_x_index
        self.plot_y_index = plot_y_index
        self.plot_x_label = plot_x_label
        self.plot_y_label = plot_y_label
        self.scenarios = scenarios
        self.n_sims_per_start = n_sims_per_start  # For stochastic systems
        self.t_sim = t_sim

    def run(self, controller) -> pd.DataFrame:
        """[summary]

        Args:
            controller ([type]): [description]

        Returns:
            pd.DataFrame: [description]
        """
        n_sims = self.n_sims_per_start * self.start_x.shape[0]

        x_current = x_sim_start
        if hasattr(controller, "reset_controller"):
            controller.reset(x_current)
        
        delta_t = controller.dynamics.dt
        num_steps = int(self.t_sim // delta_t)
        