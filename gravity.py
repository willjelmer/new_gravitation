import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wgd
import matplotlib.animation as animation
from scipy.integrate import odeint

'''
This program simulates the gravitational interactions between n-bodies and visualises their motion over
time. Each body is labelled in order: 0,1,2...n and each list containing information about the bodies are
to be in order, so that info in list[i] corresponds to body_i.
At each time interval, a new state vector, y, is created, which contains each spatial component of
the velocities and accelerations of each body, ordered as such:
y = [x_1,y_1,z_1,x_2,y_2,z_2...x_n,y_n,z_n,vx_1,vy_1,vz_1,vx_2,vy_2,vz_2...vx_n,vy_n,vz_n]
y = [pos_1,pos_2...vel_1,vel_2...]

'''

dt = 0.001 # time interval
n = 5 # number of bodies to simulate

#body1
m_1 = 3
r_10 = np.array([1,3,0])
v_10 = np.array([1,0,0])

#body 2
m_2 = 4
r_20 = np.array([-2,-1,0])
v_20 = np.array([0,0,0])

#body 3
m_3 = 5
r_30 = np.array([1,-1,0])
v_30 = np.array([0,0,0])

#body 4
m_4 = 5
r_40 = np.array([-2,4,0])
v_40 = np.array([-1,0,0])

#body 5
m_5 = 6
r_50 = np.array([2,-2,0])
v_50 = np.array([1,0,0])

masses = np.array([m_1,m_2,m_3,m_4,m_5][:n])
state0 = np.concatenate(np.concatenate(((r_10,r_20,r_30,r_40,r_50)[:n],
                                        (v_10,v_20,v_30,v_40,v_50)[:n])))

G = 4*np.pi**2 # natural units


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
                vdot[i] = vdot[i] + G * (masses[j] * (pos_j - pos_i) / r_mags[i, j] ** 3)

    vdot = np.concatenate(vdot)
    return np.concatenate((v, vdot))


time = np.arange(0, 1, dt)
plot_points = len(time)

state = odeint(motion, state0, time).T

# Create positions, array of all body positions with shape (timesteps, n_bodies, 3 dimensions)
positions = []

for i in range(n):
    # Convert state_x and state_y to numpy arrays
    state_x = np.array(state[3*i])
    state_y = np.array(state[3*i+1])
    state_z = np.array(state[3*i+2])

    # Transpose to match expected shape (timesteps, n_bodies, 2)
    positions.append(np.stack((state_x.T, state_y.T, state_z.T), axis=-1)) # Shape: (timesteps, n_bodies, 2))

positions = np.array(positions)

# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

# Initialize objects
scat = ax.scatter([], [], s=50)  # Scatter plot for current positions

# Initialize trajectory lines (one line for each body)
lines = [ax.plot([], [], lw=1)[0] for _ in range(n)]  # List of lines for each body

# Colours of each body
scat_colors = ['red', 'blue', 'green', 'black', 'purple']

# length of trailing line
trail_length = 250


# Update function for animation
def update(frame):
    # Update the scatter plot (current positions)
    scat.set_offsets(positions[:, frame, :2])
    scat.set_facecolor(scat_colors)
    scat.set_sizes(10*masses)

    # Update the trajectory lines
    for i in range(n):  # Loop through each body
        start_frame = max(0, frame - trail_length)

        lines[i].set_data(positions[i, start_frame:frame + 1, 0],
                          positions[i, start_frame:frame + 1, 1])  # Path for body i up to current frame
        lines[i].set_color(scat_colors[i])

    return scat, *lines  # Return all objects that are updated


def infinite_generator():
    frame = 0
    while True:
        yield frame
        frame += 1  # Keep increasing indefinitely


# Create animation
ani = animation.FuncAnimation(fig, update, frames=plot_points, interval=1, blit=True, repeat=False)

plt.show()