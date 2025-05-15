from stable_baselines3.common.callbacks import BaseCallback
import pickle
import os


class SaveTrajectoryCallback(BaseCallback):
    def __init__(self, num_trajectories: int, output_folder: str, model_name: str, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.num_trajectories = num_trajectories
        self.output_folder = output_folder
        self.model_name = model_name

    def _on_step(self) -> bool:
        assert self.parent is not None, "``SaveTrajectoryCallback`` must be used with ``EvalCallback``"
        eval_env = self.parent.eval_env
        trajectories = []
        for _ in range(self.num_trajectories):
            obs = eval_env.reset()
            terminated = False
            truncated = False
            trajectory = [int(eval_env.envs[0].agent_location)]
            while not (terminated or truncated):
                action, _ = self.model.predict(obs)
                obs, _, terminated, truncated_dict = eval_env.step(action)
                truncated = truncated_dict[0]["TimeLimit.truncated"]
                trajectory.append(int(action))
            trajectories.append(trajectory)

        with open(os.path.join(self.output_folder, f"{self.model_name}_trajectories_{self.n_calls}.pkl"), "wb") as f:
            pickle.dump(trajectories, f)
        return True
