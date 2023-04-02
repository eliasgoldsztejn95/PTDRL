## PTDRL for simulation

- To train ptdrl run: train_ddqn.py
- To generate recordings for benchmark purposes run: test_ddqn.py
- To visualize scores run: scorify.py
- To run robot for recording purposes for generating synthetic data run: run_robot_for_recording.py

### To change training algorithm, robot and world:
1. In your RL algorithm replace: env=task_env.PtdrlTaskEnv()
2. In task_params.yaml define parameters, local planners and goal points of the world

