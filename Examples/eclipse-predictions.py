import numpy as np
from nbodygravity import run_simulation
from datetime import timedelta, datetime

# ----- Simulation constants ----- #

epsilon = 0 # force softening constant, stops divisions by zero in force calculation
dt = 0.001 # time interval
n = 3 # number of bodies to simulate

G = 6.6743e-11 * 1.49299e8 # in units where length = 10^7km, mass = 2e28kg, time = 1 day

# ----- Initial conditions for each body ----- #

import json

with open("2025-mar-10.json", "r") as f:
    data = json.load(f)

masses = np.array(data["masses"][:n]) / 2e28 # unit conversions
positions = np.array(data["positions"][:n*3]) / 1e7
velocities = np.array(data["velocities"][:n*3]) * (60**2 * 24) / 1e7

# -------------------------------------------- #

# reference start date of the simulation, 10th March 2025
start_date = datetime(2025, 3, 10)
years_to_simulate = 10

state = run_simulation(positions, velocities, masses, 365*years_to_simulate, dt = dt, big_g = G, epsilon= epsilon)

# -------------------------------------------- #

#animate_state(state, dt, dt_resolution=200, body_names=["Sun", "Earth", "Moon"])

# Set the threshold angle for an eclipse to be recognised
threshold = 1.2 * np.pi/180

# Assigning parts of the state vector to individual variables for clarity
sun = state[:3]
earth = state[3:6]
moon = state[6:9]

# Compute displacement vectors
earth_sun = (sun - earth).T
earth_moon = (moon - earth).T

# Calculate magnitudes (distances) of these vectors
earth_sun_mag = np.linalg.norm(earth_sun, axis = 1)
earth_moon_mag = np.linalg.norm(earth_moon, axis = 1)

dot_prod = np.sum(earth_sun * earth_moon, axis = 1)

# Calculate angle (in radians) between Earth-Sun and Earth-Moon vectors
theta = np.arccos(dot_prod / (earth_moon_mag* earth_sun_mag))

# Find indices of frames where the angle is below the threshold, and convert to number of days
date_numbers = np.where(theta <= threshold)[0] * dt

# Loop through eclipse times and organise by date
eclipses_by_day = {}
for date in date_numbers:
    eclipse_date_time = start_date + timedelta(days = date)
    eclipse_date = eclipse_date_time.date()
    if eclipse_date not in eclipses_by_day:
        eclipses_by_day[eclipse_date] = []
    eclipses_by_day[eclipse_date].append(eclipse_date_time)

# Print out each eclipse and its timespan
for eclipse in eclipses_by_day:
    start_time = eclipses_by_day[eclipse][0].strftime('%H:%M')
    end_time = eclipses_by_day[eclipse][-1].strftime('%H:%M')

    print(f"{eclipse.strftime('%d-%m-%Y')} from {start_time} to {end_time}")
