
Tetsbench for APPLD.

We reproduce APPLD for benchmark purposeses.

APPLD process consists on 4 stages:

1) Generate lidar recordings of different environments, and tag them using CHAMP or manually.
2) Train a classifier from the recordings.
3) Train a CMA-ES to find the optimal parameters for each environment. This is done by behavioral cloning
on the actions of the person in each recording, using a local planner.

For this testbench, we do not reproduce the whole paper, but take the optimal parameters already found for DWA for the Jackal robot.
Because the tagging of the environments, and the correspondent classifier could not be retrieved from the authors, we generate this labels
ourselves, by moving the robot in different parts of the hospital and tagging the environments manually.

![environments_segmented](https://user-images.githubusercontent.com/75029654/200835629-8390bc0f-685b-4258-b824-f76189a6f353.png)

Blue is Open Space, green is obstalces, red is Corridor, yellow is Curves.
