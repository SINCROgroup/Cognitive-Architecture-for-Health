import socket, select, re, sys, signal, os
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from L3_Agent_PPO import L3, Kuramoto
from RecursiveOnlinePhaseEstimator import RecursiveOnlinePhaseEstimator
from ListeningModeManager import ListeningModeManager
from PhaseIndexer import PhaseIndexer
from low_pass_filters import LowPassFilter
from util import create_folder_if_not_exists, l3_update_theta, get_time_string
from wrap_functions import wrap_to_pi, wrap_to_2pi
from typing import Sequence # For type hinting numpy array
import pandas as pd

class L3_Wrapper():

    def __init__(self, model_path : str, save_path : str, exercise_ID: int = 0, ID : int =0, amplitude : float =15, omega : float =2, n_participants : int =3,
                 omega_parts=np.array([0, 3.4, 4.6]), c_strength : float = 0.25, omega_sat : float = 15):
        self.ID = ID  # Python CA instance ID
        self.current_state = exercise_ID # Exercise ID for the trial
        self.amplitude = amplitude  # Movement amplitude
        self.omega = omega  # Movement frequency
        self.initial_phase = 0
        self.n_participants = n_participants
        self.omega_sat = omega_sat
        self.c_strength = c_strength
        self.listening_manager = ListeningModeManager()

        self.l3_agent = L3(0,omega_parts[0],0,n_actions=1,input_dims=(5,), omega_sat=omega_sat, model_path=model_path)
        self.l3_agent.load_model()

        self.AGENTS = []
        self.AGENTS.append(self.l3_agent)
        for i in range(1, self.n_participants):
            self.AGENTS.append(Kuramoto(i, 0, 0))

        match exercise_ID:
            case 0: 
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/Ex02_baseline.csv", index_col='phase')
                self.baseline_time_signal = np.array(self.baseline_dataframe['Time'])  # Time signal for the baseline data
                columns_input = ['LAnkle.X', 'LAnkle.Y', 'LAnkle.Z', 'RAnkle.X', 'RAnkle.Y', 'RAnkle.Z', 'LWrist.X', 'LWrist.Y', 'LWrist.Z', 'RWrist.X', 'RWrist.Y', 'RWrist.Z']
                self.baseline_pos_loop = np.array(self.baseline_dataframe[columns_input])  # Position loop for the baseline data
                self.end_effector_inputs = 4 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
                self.select_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22] # Select LFoot, RFoot, LHand, RHand, delta_t, exercise_ID from the message
                self.look_behind_pcent = 5
                self.look_ahead_pcent = 40
                self.listening_time = 30
                self.time_const_lowpass_filter_estimand_pos = 0.1
                self.time_const_lowpass_filter_phase = 0.2
                self.is_use_elapsed_time = False
                self.omega_listening = 0.81  # Natural frequency of the L3 agent during the listening phase
            case 1:
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/Ex03_baseline.csv", index_col='phase')
                self.baseline_time_signal = np.array(self.baseline_dataframe['Time'])  # Time signal for the baseline data
                columns_input = ['Hip.X', 'Hip.Y', 'Hip.Z', 'RHip.X', 'RHip.Y', 'RHip.Z', 'LHip.X', 'LHip.Y', 'LHip.Z']
                self.baseline_pos_loop = np.array(self.baseline_dataframe[columns_input])  # Position loop for the baseline data
                self.end_effector_inputs = 3 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
                self.select_list = [0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22] # Select Pelvis, LThigh, RThigh, delta_t, exercise_ID from the message
                self.look_behind_pcent = 5
                self.look_ahead_pcent = 40
                self.listening_time = 15
                self.time_const_lowpass_filter_estimand_pos = 0.1
                self.time_const_lowpass_filter_phase = 0.2
                self.is_use_elapsed_time = False
                self.omega_listening = 1.29  # Natural frequency of the L3 agent during the listening phase
            case 2:
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/20251017_SidePlank_baseline.csv", index_col='phase')
                self.baseline_time_signal = np.array(self.baseline_dataframe['Time'])  # Time signal for the baseline data
                columns_input =  ["pelvis.X", "pelvis.Y", "pelvis.Z", "thigh_l.X", "thigh_l.Y", "thigh_l.Z", "thigh_r.X", "thigh_r.Y", "thigh_r.Z"]
                self.baseline_pos_loop = np.array(self.baseline_dataframe[columns_input])  # Position loop for the baseline data
                self.end_effector_inputs = 3 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
                self.select_list = [0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22] # Select Pelvis, LThigh, RThigh, delta_t, exercise_ID from the message
                self.look_behind_pcent = 5
                self.look_ahead_pcent = 50
                self.listening_time = 18
                self.time_const_lowpass_filter_estimand_pos = 0.1
                self.time_const_lowpass_filter_phase = 0.2
                self.is_use_elapsed_time = True
                self.omega_listening = 0.81  # Natural frequency of the L3 agent during the listening phase
            case _: 
                print(f'Exercise ID {exercise_ID} not recognized, using default baseline.')
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/Ex03_baseline.csv", index_col='phase')
                self.baseline_time_signal = pd.read_csv("./exercises_baselines/Ex03_baseline.csv", index_col='time')
                self.end_effector_inputs = 7 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
        
        self.n_dims_estimand_pos = 3 * self.end_effector_inputs  # Number of dimensions of the position estimand
        self.columns = [
            'LAnkle.X', 'LAnkle.Y', 'LAnkle.Z',
            'RAnkle.X', 'RAnkle.Y', 'RAnkle.Z',
            'LWrist.X', 'LWrist.Y', 'LWrist.Z',
            'RWrist.X', 'RWrist.Y', 'RWrist.Z',
            'Hip.X', 'Hip.Y', 'Hip.Z'
        ]
        self.indexer = PhaseIndexer(self.baseline_dataframe, columns_names=self.columns)  # Create the phase indexer with the baseline data

        omega_filter_time_const = 3  # Time constant for the omega low pass filter
        self.omega_filter = LowPassFilter(self.omega, time_step=0.01, time_const=omega_filter_time_const)  # Low pass filter for the omega of the L3 agent

        self.estimators_live = []
        ref_frame_estimand_points = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])]
        ref_frame_baseline_points = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])]
        for _ in range(self.n_participants):
            self.estimators_live.append(RecursiveOnlinePhaseEstimator(self.n_dims_estimand_pos, self.listening_time, discarded_time=0, min_duration_first_pseudoperiod=0, look_behind_pcent=self.look_behind_pcent, look_ahead_pcent=self.look_ahead_pcent, time_const_lowpass_filter_phase=self.time_const_lowpass_filter_phase,
                                                                       time_const_lowpass_filter_pos=self.time_const_lowpass_filter_estimand_pos, is_use_elapsed_time=self.is_use_elapsed_time, is_use_baseline = True, ref_frame_estimand_points=ref_frame_estimand_points, ref_frame_baseline_points=ref_frame_baseline_points,
                                                                       baseline_time_signal=self.baseline_time_signal, baseline_pos_loop=self.baseline_pos_loop))

        self.n_end_effectors = 5 # Number of end effectors available (hands, feet, hip)
        self.time_history = [0]
        self.time_new_exercise = 0 # Time when the new exercise is set. Needed to restart listening phase of the estimators
        self.initial_phase = 0
        self.current_phase = 0  # Current phase of the L3 agent
        self.time_last_save = 0
        self.phases_history = [np.zeros(self.n_participants)]
        self.positions_history = []
        self.save_path = save_path
        create_folder_if_not_exists(save_path)

    def reset_CA(self):
        # STORE DATA
        self.save_data()
        self.plot_phases(np.stack(self.phases_history))

        # RESET THE PHASE ESTIMATORS
        self.estimators_live = []
        for _ in range(self.n_participants):
            self.estimators_live.append(RecursiveOnlinePhaseEstimator(self.n_dims_estimand_pos, self.listening_time, discarded_time=1, min_duration_first_pseudoperiod=0, look_behind_pcent=self.look_behind_pcent, look_ahead_pcent=self.look_ahead_pcent, time_const_lowpass_filter_phase=self.time_const_lowpass_filter_phase, time_const_lowpass_filter_pos=self.time_const_lowpass_filter_estimand_pos, is_use_elapsed_time=self.is_use_elapsed_time))

        self.phases_history = [np.zeros(self.n_participants)]
        # self.time_history = [0]
        self.time_history = [self.time_history[-1]]
        self.positions_history = []
     
    def parse_TCP_string(self, string : str) -> tuple[bool, Sequence[float]]:
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        flag = len(numbers) == self.n_dims_estimand_pos * self.n_participants + 2  # ADP: Uncomment for the version with state machine
        numbers = numbers[self.select_list] # Filter points of interest from the message, according to the exercise
        # return flag, numbers[0:-1], numbers[-1]  # Use flag for debugging. The last number is the time in seconds
        return flag, numbers[0:-2], numbers[-2], numbers[-1]  # ADP: Uncomment for the version with state machine

    def set_initial_position(self, position : list[list[float]]):
        position = np.reshape(position, (self.n_participants, self.n_dims_estimand_pos)).T
        self.positions_history.append(position)
        self.initial_phase = 0
    
    # Calculates the next position and formats the message to be sent to UE for animation
    def update_position(self, positions : list[list[float]], delta_t : float, time : float) -> str:
        # positions contains the neighbors 3D end effectors
        positions = np.reshape(positions, (self.n_participants, self.n_dims_estimand_pos)).T

        self.current_phase = self.l3_agent.theta  # Get the current phase of the L3 agent
        ic(wrap_to_2pi(self.current_phase))

        ic(time)

        ic(positions)
        phases = [] # Vector of the real phases
        for i in range(self.n_participants):  # Collect phases of all participants. The ones from other participants are estimated
            if self.AGENTS[i].is_virtual == False:
                phases.append(self.estimators_live[i].update_estimator(positions[:, i], time))  # Update the phase estimator with the new position
                phases[-1] = self.listening_manager.mask_none(phases[-1])
                phases[-1] = wrap_to_pi(phases[-1])  # Wrap the phase to [-pi, pi]
            else:
                phases.append(self.current_phase - self.initial_phase)

        self.time_history.append(time)  # Store the time in the history
        self.phases_history.append(np.array(phases))
        
        # observation = self.l3_agent.get_state(np.array(phases), self.n_participants)
        phases_dots = (self.phases_history[-1] - self.phases_history[-2])/delta_t  # Compute the phase dots
        observation = self.l3_agent.get_state_with_theta_dots(np.array(phases), phases_dots, self.n_participants)
        if time > (self.estimators_live[0].listening_time + self.estimators_live[0].discarded_time):  # If the listening time is over, use the phase estimator to compute the omega
            new_omega = self.l3_agent.choose_action_mean(observation)  # Compute the new omega for the virtual agent
            # if new_omega < 0: new_omega = 0
            # self.omega_filter.change_time_step(delta_t)  # Update the time step of the omega filter
            # self.l3_agent.omega = self.omega_filter.update_state(new_omega)  # Apply the low pass filter to the omega
            self.l3_agent.omega = new_omega
        else:
            self.l3_agent.omega = self.omega_listening

        l3_theta_next = l3_update_theta(np.array(phases), self.l3_agent.omega, coupling=self.c_strength, dt=delta_t)
        self.l3_agent.theta = wrap_to_pi(l3_theta_next) # Need wrap_to_pi to plot with other phases

        end_effectors_positions = self.indexer.get_values_at_phase(wrap_to_2pi(l3_theta_next))
        message = self.write_message(end_effectors_positions)  # Format the message to be sent to UE

        return message
    
    def plot_phases(self, phases: list[list[float]]):
        # Plotting
        colors = ['red', 'blue', 'magenta', 'yellow', 'orange', 'olive', 'cyan']
        plt.figure()
        for i in range(self.n_participants):
            if self.AGENTS[i].is_virtual == True: 
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'L3 {i + 1}')
            else:
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'VH {i + 1}')

        plt.title('Phases of Experiment')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Phases (radiants)')
        plt.legend()
        plt.grid(True)

        time_string = get_time_string()

        plt.savefig(f'{self.save_path}\\'+time_string+'phases_plot.png')
        plt.close()

    def save_data(self):
        time_string = get_time_string()
        np.save(f'{self.save_path}/'+time_string+'phases_history.npy',   np.stack(self.phases_history))
        # np.save(f'{self.save_path}/'+time_string+'positions_history.npy', np.array(self.positions_history))
        np.save(f'{self.save_path}/'+time_string+'time_history.npy',     np.stack(self.time_history))
    
    def write_message(self, end_effectors_positions : list[list[float]]) -> str:
        message = ''
        for j in range(self.n_end_effectors):
            message += f'X={end_effectors_positions[3*j]} Y={end_effectors_positions[3*j+1]} Z={end_effectors_positions[3*j+2]}/'
        message = message[:-1]  # Remove the last slash
        message += ';'  # End of participant's end effectors
        ic(message)
        return message
    
    def exercise_setting(self, exercise_ID : int):
        match exercise_ID:
            case 0: 
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/Ex02_baseline.csv", index_col='phase')
                self.baseline_time_signal = np.array(self.baseline_dataframe['Time'])  # Time signal for the baseline data
                columns_input = ['LAnkle.X', 'LAnkle.Y', 'LAnkle.Z', 'RAnkle.X', 'RAnkle.Y', 'RAnkle.Z', 'LWrist.X', 'LWrist.Y', 'LWrist.Z', 'RWrist.X', 'RWrist.Y', 'RWrist.Z']
                self.baseline_pos_loop = np.array(self.baseline_dataframe[columns_input])  # Position loop for the baseline data
                self.end_effector_inputs = 4 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
                self.select_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22] # Select LFoot, RFoot, LHand, RHand, delta_t, exercise_ID from the message
                self.look_behind_pcent = 5
                self.look_ahead_pcent = 40
                self.listening_time = 30
                self.time_const_lowpass_filter_estimand_pos = 0.1
                self.time_const_lowpass_filter_phase = 0.2
                self.is_use_elapsed_time = False
                self.omega_listening = 0.81  # Natural frequency of the L3 agent during the listening phase
            case 1:
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/Ex03_baseline.csv", index_col='phase')
                self.baseline_time_signal = np.array(self.baseline_dataframe['Time'])  # Time signal for the baseline data
                columns_input = ['Hip.X', 'Hip.Y', 'Hip.Z', 'RHip.X', 'RHip.Y', 'RHip.Z', 'LHip.X', 'LHip.Y', 'LHip.Z']
                self.baseline_pos_loop = np.array(self.baseline_dataframe[columns_input])  # Position loop for the baseline data
                self.end_effector_inputs = 3 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
                self.select_list = [0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22] # Select Pelvis, LThigh, RThigh, delta_t, exercise_ID from the message
                self.look_behind_pcent = 5
                self.look_ahead_pcent = 40
                self.listening_time = 15
                self.time_const_lowpass_filter_estimand_pos = 0.1
                self.time_const_lowpass_filter_phase = 0.2
                self.is_use_elapsed_time = False
                self.omega_listening = 1.29  # Natural frequency of the L3 agent during the listening phase
            case 2:
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/20251017_SidePlank_baseline.csv", index_col='phase')
                self.baseline_time_signal = np.array(self.baseline_dataframe['Time'])  # Time signal for the baseline data
                columns_input =  ["pelvis.X", "pelvis.Y", "pelvis.Z", "thigh_l.X", "thigh_l.Y", "thigh_l.Z", "thigh_r.X", "thigh_r.Y", "thigh_r.Z"]
                self.baseline_pos_loop = np.array(self.baseline_dataframe[columns_input])  # Position loop for the baseline data
                self.end_effector_inputs = 3 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)
                self.select_list = [0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22] # Select Pelvis, LThigh, RThigh, delta_t, exercise_ID from the message
                self.look_behind_pcent = 5
                self.look_ahead_pcent = 50
                self.listening_time = 18
                self.time_const_lowpass_filter_estimand_pos = 0.1
                self.time_const_lowpass_filter_phase = 0.2
                self.is_use_elapsed_time = True
                self.omega_listening = 0.81  # Natural frequency of the L3 agent during the listening phase
            case _: 
                print(f'Exercise ID {exercise_ID} not recognized, using default baseline.')
                self.baseline_dataframe = pd.read_csv("./exercises_baselines/Ex03_baseline.csv", index_col='phase')
                self.baseline_time_signal = pd.read_csv("./exercises_baselines/Ex03_baseline.csv", index_col='time')
                self.end_effector_inputs = 5 # End effector in input for the exercise (it is not necessarily equal to the end effectors of the baseline data)

    def change_exercise(self, new_exercise_ID : int, time : float):
        old_state = self.current_state
        self.current_state = new_exercise_ID
        self.time_last_save = time
        if self.current_state == old_state:
            changed = False
            return changed
        else:
            ic(f'Changing exercise from {old_state} to {new_exercise_ID}')
            self.exercise_setting(new_exercise_ID)
            self.indexer = PhaseIndexer(self.baseline_dataframe, columns_names=self.columns)  # Create the phase indexer with the baseline data
            self.time_new_exercise = self.time_history[-1]
            self.n_dims_estimand_pos = 3 * self.end_effector_inputs  # Number of dimensions of the position estimand
            self.reset_CA()
            changed = True
            return changed
        
    @staticmethod
    def start_connection(address : str, port : int) -> tuple[str, int]:
        # Create a TCP sockets
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Bind the socket to the server address and port
            server_socket.bind((address, port))
        except socket.error as e:
            print("Connection error: %s" % e)

        # Listen for incoming connections
        server_socket.listen(1)  # Limit number of connections to L3 socket
        print(f'Server listening on {address}:{port}')

        # Wait for a client connection
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()
        print(f'Connection from {client_address}')

        return connection, client_address