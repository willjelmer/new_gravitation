import numpy as np
from nbodygravity import run_simulation, animate_state
from datetime import timedelta, datetime

# ----- Simulation constants ----- #

epsilon = 0 # force softening constant, stops divisions by zero in force calculation
dt = 0.001 # time interval
n = 10 # number of bodies to simulate

G = 6.6743e-11 * 1.49299e8 # in units where length = 10^7km, mass = 2e28kg, time = 1 day

# ----- Initial conditions for each body ----- #

import json

with open("2025-mar-10.json", "r") as f:
    data = json.load(f)

masses = np.array(data["masses"][:n]) / 2e28
positions = np.array(data["positions"][:n*3]) / 1e7
velocities = np.array(data["velocities"][:n*3]) * (60**2 * 24) / 1e7

# -------------------------------------------- #

# reference start date of the simulation, 10th March 2025
start_date = datetime(2025, 3, 10)
years_to_simulate = 10

# Run simulation and get state vector
state = run_simulation(positions, velocities, masses, 365*years_to_simulate, dt = dt, big_g = G, epsilon= epsilon)

body_names = ["Sun", "Earth", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
animate_state(state, dt, dt_resolution=200, body_names=body_names)