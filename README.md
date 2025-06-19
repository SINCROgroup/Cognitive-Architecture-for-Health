# Cognitive Architecture for Health

To set up your machine:
- Install python packages running `pip install -r python_packages.txt`.

## Functionalities of the Repository  

Folder `CA_health_scenario` contains the python capable of:
- Connecting with TCP sockets and stream data (by default expects connection by the same local machine).
- Extract phases from 3D positions of end-effector's of human participants .
- Simulate an interaction with 1 trained PPO artificial L3 avatar and an arbitrary number of humans.

## CA for health scenario
To run the application:
- Move in the folder `CA_health_scenario` and launch the python script `main.py` passing the number of participant the L3 has to interact with and the number of the selected exercise and wait for the TCP socket to be online (see prompt messages). Two different exercises are available, whose baselines for motion are contained in the `exercise_baseline` subfolder.

Example usage:  
`python main.py 4 simulation_data 0` : -> run the code for the `Ex02_baseline.csv` with 4 other participants  
`python main.py 5 simulation_data 1` : -> run the code for the `Ex03_baseline.csv` with 5 other participants

Useful information
- The TCP connection consists in the exchange of strings containig 3D end-effector positions and the time interval (delta_t) between each message. The string messages are structured as: `X={1st end-effector x value} Y={1st end-effector y value} Z={1st end-effector z value} / X={2nd end-effector x value} Y={2nd end-effector y value} Z={2nd end-effector z value} / … (repeat for the end-effectors you want) ... ; {delta_t value}`.
- To let the CA receive position data of multiple participants, concatenate them in a unique string and leave the `delta_t` as the last element.  As a good practice, separate each participant by a `;` and each end effector of a single participant by a `/`.
- The python script sends out a message structured as explained before in the following order: `Left Foot (XYZ) / Right Foot (XYZ) / Left Hand (XYZ) / Right Hand (XYZ) / Hip position (XYZ);`

