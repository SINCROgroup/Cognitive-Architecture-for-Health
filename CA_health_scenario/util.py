import cmath
import os
import numpy as np
from typing import Sequence
from datetime import datetime


def wrap_angle(angle_rads: float, domain: str) -> float:
    if domain == '-pi to pi':
        return ((angle_rads + np.pi) % (2 * np.pi)) - np.pi
    elif domain == '0 to 2pi':
        return angle_rads % (2 * np.pi)
    else:
        raise ValueError(f"The value of domain '{domain}' is not acceptable.")


def compute_average_phasor(angles: Sequence[float]) -> float:
    return np.mean(np.exp(1j * np.array(angles)))


def compute_average_angle(angles: Sequence[float]) -> float:
    return cmath.phase(compute_average_phasor(angles))


def compute_order_parameter(angles: Sequence[float]) -> float:
    return np.abs(compute_average_phasor(angles))


def create_folder_if_not_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")


def saturate_soft(x: float, thres: float, p: float) -> float:
    return x / ((1 + np.abs(x / thres) ** p) ** (1 / p))


def saturate_hard(x: float, thres: float) -> float:
    if x <= - thres:  return - thres
    elif x >= thres:  return thres
    else:             return x


def bump_fun(x: float) -> float:
    if abs(x) >= 1:  return 0
    else:            return np.exp(1 - 1 / (1 - np.abs(x) ** 2))

def l3_update_theta(theta_old, omega, coupling, dt, virtual_agent_index=0):
    delta_theta = theta_old - theta_old[virtual_agent_index]
    sin_delta_theta = np.sin(delta_theta)
    theta_new = theta_old[virtual_agent_index] + dt * (omega + sum(coupling * sin_delta_theta))
    return theta_new

def get_time_string():
    current_time = datetime.now()
    h = current_time.hour
    m = current_time.minute
    s = current_time.second
    y = current_time.year
    mo = current_time.month
    d = current_time.day
    time_string = str(y)+'_'+str(mo)+'_'+str(d)+'_h'+str(h)+'_m'+str(m)+'_s'+str(s)+'_'

    return time_string