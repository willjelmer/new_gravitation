import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wgd
import matplotlib.animation as animation
from scipy.integrate import odeint
import time


'''
This program simulates the gravitational interactions between n-bodies and visualises their motion over
time. Each body is labelled in order: 0,1,2...n and each list containing information about the bodies are
to be in order, so that info in list[i] corresponds to body_i.
At each time interval, a new matrix, y, is created, which contains each spatial component of
the velocities and accelerations of each body, ordered as such:
y = [x_1,y_1,z_1,x_2,y_2,z_2...x_n,y_n,z_n,vx_1,vy_1,vz_1,vx_2,vy_2,vz_2...vx_n,vy_n,vz_n]
y = [pos_1,pos_2...vel_1,vel_2...]
These matrices are added to another matrix, state, which holds onto the last few states which are then
displayed as a trailing tail.
'''

# used for generating trajectory trails
past_points = np.array([])
relative_trails = [False]

focus = "none" # initial body to focus on

epsilon = 1e-3 # force softening constant, stops divisions by zero in force calculation
dt = 0.1 # time interval
n = 4 # number of bodies to simulate

G = 6.6743e-11 * 1.49299e8 # in units where length = 10^7km, mass = 2e28kg, time = 1 day

# Initialise bodies
#body1 sun
m_1 = 99.4
r_10 = np.array([0,0,0])
v_10 = np.array([0,0,0])

#body 2 earth
m_2 = 3.0e-4
r_20 = np.array([-1.459847e8,2.754070e7,-1.093196e3])/1e7 # * 10^7 km
v_20 = np.array([-5.996424,-2.937786e1,2.292323e-3])*(60**2*24*1)/1e7 # * (10^7km) month^-1

#body 3 moon
m_3 = 3.6e-6
r_30 = np.array([-1.461795e8,2.787552e7,2.889309e4])/1e7
v_30 = np.array([-6.892628,-2.984489e1,-4.503372e-2])*(60**2*24*1)/1e7

#body 4
m_4 = 74
r_40 = [10,10,0]
v_40 = [-0.5,-0.2,0]


# #body 4 jupiter
# m_4 = 0.0949095
# r_40 = np.array([-8.124191e8,7.594032e8,-4.972164e6])/1e7
# v_40 = np.array([-1.315457e1,2.005578,2.860240e-1])*(60**2*24*1)/1e7

#body 5
m_5 = 6
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


# Update function for animation
def update(frame):
    '''
    This function handles all calculations and graph updates every frame.
    :param frame:
    :return: Animation function requires that all updated parameters are returned
    '''

    # Take start time to measure FPS at the end
    start_time = time.time()

    global state

    # Reset axes
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.cla()

    # Create positions list which holds only x,y,z coordinates of the bodies
    positions = []
    for i in range(n):
        state_x = np.array(state[3 * i])
        state_y = np.array(state[3 * i + 1])
        state_z = np.array(state[3 * i + 2])
        # Transpose to match expected shape (timesteps, n_bodies, 2)
        positions.append(np.stack((state_x.T, state_y.T, state_z.T), axis=-1))  # Shape: (timesteps, n_bodies, 2))
    positions = np.array(positions)

    # compute next state, and add this to the state function
    last_state = state[:, -1]
    next_time = np.array([frame * dt, frame * dt + dt])
    next_state = odeint(motion, last_state, next_time).T[:,1:]
    state = np.hstack((state,next_state))

    # for focussing on objects.
    start_frame = max(0, frame - trail_length)
    if focus != "none":
        infocus = int(focus)
        focus_position = positions[infocus, frame, :2] # extract the position of the focussed object
        if relative_trails[0]:
            # past points to calculate trail relative to focus point
            points = positions[infocus, start_frame:frame + 1, :]
        else:
            points = np.tile(focus_position,(frame+1-start_frame,1))
        # reshape the position matrix to match scatter plot offsets, for easier calculation
        focus_position = np.vstack([focus_position] * n)
    else:
        focus_position = np.zeros((n,2))
        points = np.zeros((frame + 1 - start_frame,3))

    # Update the scatter plot (current positions)
    scat.set_offsets(positions[:, frame, :2]-focus_position)
    print((positions[:, frame, :2]-focus_position).shape)

    # Set colour and size of each body
    scat.set_facecolor(scat_colors)
    scat.set_sizes(10*masses)

    # Update the trajectory lines (past positions)
    for i in range(n):  # Loop through each body
        lines[i].set_data(positions[i, start_frame:frame + 1, 0] - points[:,0],
                          positions[i, start_frame:frame + 1, 1] - points[:,1])  # Path for body i up to current frame
        lines[i].set_color(scat_colors[i])

    ax.set_xlim(xlim)  # Keep zoom level
    ax.set_ylim(ylim)

    # Calculate and display number of time units passed, and the frames per second
    elapsed_time = frame * dt  # Convert frames to simulation time
    time_text.set_text(f'Time: {elapsed_time:.2f} days')
    real_time_per_frame = 1/(time.time() - start_time)
    if frame % 5 == 0:
        real_time_text.set_text(f'FPS: {real_time_per_frame:.6f}')

    return scat, *lines, time_text, real_time_text # Return all objects that are updated


def infinite_generator():
    # This function generates an increasing number of frames, so simulation runs endlessly
    frame = 0
    while True:
        yield frame
        frame += 1  # Keep increasing indefinitely


def radio_clicked(val):
    global focus
    focus = val


def toggle_trails(event):
    global relative_trails
    relative_trails[0] = not relative_trails[0]


# length of trailing line
trail_length = 250

# compute first length to initialize state vector
timex = np.arange(0, dt, dt)
state = odeint(motion, state0, timex).T

# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-16,16)
ax.set_ylim(-16,16)

# Set up the radio buttons
radio_ax = plt.axes([0.05, 0.8, 0.15, 0.15])  # Position for the radio buttons
radio = wgd.RadioButtons(radio_ax,  ["none"] + [str(i) for i in range(n)])
radio.on_clicked(radio_clicked)

# Set up relative trails button
button_ax = plt.axes([0.2, 0.85, 0.15, 0.1])
check = wgd.CheckButtons(button_ax, ['RT'], [False])
check.on_clicked(toggle_trails)

# Set up text
time_text = ax.text(0.7, 0.95, '', transform=ax.transAxes, fontsize=10)
real_time_text = ax.text(0.4, 0.95, '', transform=ax.transAxes, fontsize=10)

# Initialize objects
scat = ax.scatter([], [], s=50)  # Scatter plot for current positions

# Initialize trajectory lines (one line for each body)
lines = [ax.plot([], [], lw=1)[0] for _ in range(n)]  # List of lines for each body

# Colours of each body
scat_colors = ['red', 'blue', 'green', 'orange', 'purple']

# Create animation
ani = animation.FuncAnimation(fig, update, frames=infinite_generator(), interval=1, blit=True, repeat=False)

plt.show()