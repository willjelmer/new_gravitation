import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from datetime import datetime, timedelta


reference_data = datetime(2025, 3, 10)

epsilon = 0 # force softening constant, stops divisions by zero in force calculation
dt = 0.003 # time interval
n = 4 # number of bodies to simulate

G = 6.6743e-11 * 1.49299e8 # in units where length = 10^7km, mass = 2e28kg, time = 1 day

#body1
m_1 = 99.4205
r_10 = np.array([0,0,0])
v_10 = np.array([0,0,0])

#body 2 earth
m_2 = 2.986095e-4
r_20 = np.array([-1.459847e8,2.754070e7,-1.093196e3])/1e7 # * 10^7 km
v_20 = np.array([-5.996424,-2.937786e1,2.292323e-3])*(60**2*24*1)/1e7 # * (10^7km) month^-1

#body 3 moon
m_3 = 3.6745e-6
r_30 = np.array([-1.461795e8,2.787552e7,2.889309e4])/1e7
v_30 = np.array([-6.892628,-2.984489e1,-4.503372e-2])*(60**2*24*1)/1e7

#body 4 jupiter
m_4 = 0.0949095
r_40 = np.array([-8.124191e8,7.594032e8,-4.972164e6])/1e7
v_40 = np.array([-1.315457e1,2.005578,2.860240e-1])*(60**2*24*1)/1e7

#body 5
m_5 = 0.0949095
r_50 = np.array([2,-2,0])
v_50 = np.array([1,25,0])

masses = np.array([m_1,m_2,m_3,m_4,m_5][:n])
state0 = np.concatenate(np.concatenate(((r_10,r_20,r_30,r_40,r_50)[:n],
                                        (v_10,v_20,v_30,v_40,v_50)[:n])))



def motion(y,t):
    '''
    This function calculates the acceleration for each pair of bodies
    :param y: state vector, y_n
    :return: next state vector, y_n+dt
    '''

    # Define matrix of radial distances
    r_mags = np.zeros([n, n])
    for r in range(n):
        for c in range(n):
            r_mags[r, c] = np.sqrt((y[3 * r] - y[3 * c]) ** 2 +
                                   (y[3 * r + 1] - y[3 * c + 1]) ** 2 +
                                   (y[3 * r + 2] - y[3 * c + 2]) ** 2)

    # Pull the velocities from the state vector.
    v = y[n * 3:n * 6]

    # Compute acceleration due to gravity for each pair of bodies. Here we compute the acceleration on each
    # body, i, from each other body, j, and add these accelerations to a 2-D matrix vdot, holding all
    # accelerations. The acceleration magnitude is calculated using Newton's law of gravitation, and the
    # direction is calculated by taking the displacement vector pos_j - pos_i and dividing by the radial
    # distance between these bodies, held in previously defined r_mags

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

def total_energy(y):
    E = 0
    positions = []
    for i in range(n):
        # Convert state_x and state_y to numpy arrays
        state_x = np.array(y[3 * i])
        state_y = np.array(y[3 * i + 1])
        state_z = np.array(y[3 * i + 2])
        # Transpose to match expected shape (timesteps, n_bodies, 2)
        positions.append(np.stack((state_x.T, state_y.T, state_z.T), axis=-1))  # Shape: (timesteps, n_bodies, 2))
    positions = np.array(positions)
    for i in range(n):
        velocity = np.array(y[3*n+3*i: 3*n+3*i+3])
        E += 1/2 * masses[i] * np.linalg.norm(velocity)**2
    for a in range(n):
        for b in range(n):
            if a != b:
                displacement_vector = positions[a]-positions[b]
                E -= G*masses[a]*masses[b] / np.linalg.norm(displacement_vector)
    return E

#print(total_energy(state0))

timex = np.arange(0, 365*10, dt)

# pre-computes a length, specified in trail_length to firstly display
state = odeint(motion, state0, timex).T
#print(total_energy(state[:,-1]))

positions = []
for i in range(n):
    # Convert state_x and state_y to numpy arrays
    state_x = np.array(state[3 * i])
    state_y = np.array(state[3 * i + 1])
    state_z = np.array(state[3 * i + 2])

    # Transpose to match expected shape (timesteps, n_bodies, 2)
    positions.append(np.stack((state_x.T, state_y.T, state_z.T), axis=-1))  # Shape: (timesteps, n_bodies, 2))
positions = np.array(positions)

my_guess = positions[:,-1,:][1]
nasa = np.array([-1.457172e+08,2.883368e+07,-1.313653e+03])/1e7 # nasa coords 2 years from start date

my_guess_mag = np.linalg.norm(my_guess)
difference = np.linalg.norm(my_guess - nasa)


mine = positions[:,:,:][1][:,:2]
mine_x = mine[:,0]
mine_y = mine[:,1]


# Plot the data
plt.figure(figsize=(4,4))

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




x = np.linspace(0,len(thetas)*0.003,len(thetas))
plt.plot(x,thetas)
plt.show()






#plt.scatter(nasa[0],nasa[1], color="blue")
#plt.scatter(my_guess[0],my_guess[1], color = 'red')

# plt.plot(mine_x,mine_y)
#
# plt.xlim(-15,15)
# plt.ylim((-15,15))
# plt.show()

