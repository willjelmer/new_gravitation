import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from datetime import datetime, timedelta

# reference start date of the simulation
reference_data = datetime(2025, 3, 10)
years_to_simulate = 10

# ----- Simulation constants ----- #
epsilon = 0 # force softening constant, stops divisions by zero in force calculation
dt = 0.003 # time interval
n = 4 # number of bodies to simulate
G = 6.6743e-11 * 1.49299e8 # in units where length = 10^7km, mass = 2e28kg, time = 1 day

# ----- Initial conditions for each body ----- #
#body 1 - sun
m_1 = 99.4205
r_10 = np.array([0,0,0])
v_10 = np.array([0,0,0])

#body 2 - earth
m_2 = 2.986095e-4
r_20 = np.array([-1.459847e8,2.754070e7,-1.093196e3])/1e7 # * 10^7 km
v_20 = np.array([-5.996424,-2.937786e1,2.292323e-3])*(60**2*24*1)/1e7 # * (10^7km) month^-1

#body 3 - moon
m_3 = 3.6745e-6
r_30 = np.array([-1.461795e8,2.787552e7,2.889309e4])/1e7
v_30 = np.array([-6.892628,-2.984489e1,-4.503372e-2])*(60**2*24*1)/1e7

#body 4 - jupiter
m_4 = 0.0949095
r_40 = np.array([-8.124191e8,7.594032e8,-4.972164e6])/1e7
v_40 = np.array([-1.315457e1,2.005578,2.860240e-1])*(60**2*24*1)/1e7

#body 5 - random test body
m_5 = 0.0949095
r_50 = np.array([2,-2,0])
v_50 = np.array([1,25,0])

# construct initial state vector and masses
masses = np.array([m_1,m_2,m_3,m_4,m_5][:n])
state0 = np.concatenate(np.concatenate(((r_10,r_20,r_30,r_40,r_50)[:n],
                                        (v_10,v_20,v_30,v_40,v_50)[:n])))


def motion(y,t):
    '''
    This function calculates the derivative of the state vector
    :param y: state vector, y_n
    :param t: current time (not used here but required by odeint)
    :return: next state vector, y_n+dt
    '''

    # Define matrix of radial distances
    positions = y[:n * 3].reshape((n, 3))
    disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_mags = np.linalg.norm(disp, axis=-1)

    # Pull the velocities from the state vector.
    v = y[n * 3:n * 6]

    vdot = np.zeros((n, 3))
    for i in range(n):
        for j in range(n):
            if i != j:
                pos_i = y[3 * i:3 * i + 3]
                pos_j = y[3 * j:3 * j + 3]
                vdot[i] = vdot[i] + G * (masses[j] * (pos_j - pos_i) /
                                         (r_mags[i, j] ** 3 + epsilon **3))

    vdot = np.concatenate(vdot)
    return np.concatenate((v, vdot))


# Time array for integration
timex = np.arange(0, 365*years_to_simulate, dt)

# Integrates motion using scipy's ODE solver
state = odeint(motion, state0, timex).T

# Extract positions of each body over time
positions = []
for i in range(n):
    # Convert state_x and state_y to numpy arrays
    state_x = np.array(state[3 * i])
    state_y = np.array(state[3 * i + 1])
    state_z = np.array(state[3 * i + 2])

    # Transpose to match expected shape (timesteps, n_bodies, 2)
    positions.append(np.stack((state_x.T, state_y.T, state_z.T), axis=-1))  # Shape: (timesteps, n_bodies, 2))
positions = np.array(positions)

thetas = []
dates = []

for i in range(len(positions[0])):
    y = positions[:,i,:]
    sun, earth, moon, jupiter = y

    earth_sun = sun - earth
    earth_moon = moon - earth

    dot_product = np.dot(earth_sun, earth_moon)
    earth_moon_mag = np.linalg.norm(earth_moon)
    earth_sun_mag = np.linalg.norm(earth_sun)

    cos_theta = dot_product / (earth_moon_mag* earth_sun_mag)
    theta = np.arccos(cos_theta)
    thetas.append(theta)
    if theta <= (1.2 * np.pi/180):
        eclipse_date = reference_data + timedelta(days = i * dt)
        dates.append(eclipse_date)

unique_dates = {}
for date in dates:
    date_key = date.date()  # Extract just the date (YYYY-MM-DD)
    if date_key not in unique_dates:
        unique_dates[date_key] = []
    unique_dates[date_key].append(date)

# Print dates and intervals
sorted_dates = sorted(unique_dates.keys())  # Get unique sorted dates

print("Eclipse Dates and Intervals:")
for i, date_key in enumerate(sorted_dates):
    date_str = date_key.strftime("%d-%m-%Y")

    start_time = unique_dates[date_key][0].strftime("%H:%M")
    end_time = unique_dates[date_key][-1].strftime("%H:%M")
    print(f"{date_str} ("+str(start_time)+"-"+str(end_time)+")")

