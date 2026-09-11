# Configurations

Everything that Table 2 of the paper lists, in machine-readable form. These files are read by the
scripts; they are not documentation of the settings, they *are* the settings.

| File | Contents |
|---|---|
| `protocol.yaml`  | seed sets, admissibility gate, evaluation windows |
| `training.yaml`  | learner hyperparameters, identical across SAC, TD3 and PPO |
| `preferences.yaml` | the six preference vectors, including the gap-filling preference |
| `baselines.yaml` | PID and draft-loop gains, MPC horizon and weights, fuzzy rule base |
