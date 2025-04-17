import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wgd
import matplotlib.animation as animation
from scipy.integrate import odeint
import time


relative_trails = [False]   # Mutable object used to toggle relative trails from the UI
focus = "none"  # Focused object for relative plotting (can be changed via UI)

# ----- Simulation parameters ----- #
epsilon = 1e-3  # Softening constant to prevent division by zero in force calculations
dt = 0.1        # Time step (days)
n = 3           # Number of bodies to simulate
G = 6.6743e-11 * 1.49299e8  # Gravitational constant, scaled to units of 10^7 km, 2e28 kg, and 1 day

# ----- Masses and initial conditions for celestial bodies ----- #

# (Units: position -> 10^7 km, velocity -> 10^7 km/day, time -> day)

# Sun
m_1 = 99.4
r_10 = np.array([0, 0, 0])
v_10 = np.array([0, 0, 0])

# Earth
m_2 = 3.0e-4
r_20 = np.array([-1.459847e8, 2.754070e7, -1.093196e3]) / 1e7
v_20 = np.array([-5.996424, -29.37786, 2.292323e-3]) * (60**2 * 24) / 1e7

# Moon
m_3 = 3.6e-6
r_30 = np.array([-1.461795e8, 2.787552e7, 2.889309e4]) / 1e7
v_30 = np.array([-6.892628, -29.84489, -0.04503372]) * (60**2 * 24) / 1e7

# Jupiter
m_4 = 0.0949095
r_40 = np.array([-8.124191e8, 7.594032e8, -4.972164e6]) / 1e7
v_40 = np.array([-13.15457, 2.005578, 0.2860240]) * (60**2 * 24) / 1e7

# Additional test body (only used if n = 6)
m_5 = 6
r_50 = np.array([2, -2, 0])
v_50 = np.array([1, 25, 0])

# Mass array and initial state vector
masses = np.array([m_1, m_2, m_3, m_4, m_5][:n])
state0 = np.concatenate(np.concatenate(((r_10, r_20, r_30, r_40, r_50)[:n],
                                        (v_10, v_20, v_30, v_40, v_50)[:n])))


def motion(y, t):
    '''
    Calculates the derivative of the state vector for all bodies at time t.

    y: Full state vector (positions + velocities) flattened into 1D
    t: Time (not used explicitly because system is time-invariant)

    Returns: dydt — Flattened derivative of the state (velocities + accelerations)
    '''

    # Extract positions and reshape into (n, 3)
    positions = y[:n * 3].reshape((n, 3))

    # Compute pairwise displacement vectors between all bodies using broadcasting
    disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    # Compute distances between all pairs (shape: n x n)
    r_mags = np.linalg.norm(disp, axis=-1)

    # Extract current velocities (also shape: n x 3)
    v = y[n * 3:n * 6]

    # Initialize acceleration (dv/dt) array
    vdot = np.zeros((n, 3))

    # Loop over each pair of bodies to compute gravitational acceleration
    for i in range(n):
        for j in range(n):
            if i != j:
                pos_i = y[3 * i:3 * i + 3]
                pos_j = y[3 * j:3 * j + 3]
                # Newton's law of gravity, softened by epsilon to avoid blowup
                vdot[i] += G * masses[j] * (pos_j - pos_i) / (r_mags[i, j]**3 + epsilon**3)

    # Flatten and concatenate velocities and accelerations
    vdot = np.concatenate(vdot)
    return np.concatenate((v, vdot))


def update(frame):
    '''
    Updates the positions and trails of all bodies for each frame.

    frame: The current frame number

    Returns: All updated matplotlib artists for blitting
    '''

    start_time = time.time()  # Used for FPS calculation

    global state  # Use the global state matrix

    # Store current axis limits so we can keep zoom constant
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.cla()  # Clear previous frame’s drawings

    # Build up position matrix (n, frames, 3)
    positions = []
    for i in range(n):
        state_x = np.array(state[3 * i])
        state_y = np.array(state[3 * i + 1])
        state_z = np.array(state[3 * i + 2])
        positions.append(np.stack((state_x.T, state_y.T, state_z.T), axis=-1))
    positions = np.array(positions)

    # Integrate next state from the last known state
    last_state = state[:, -1]
    next_time = np.array([frame * dt, frame * dt + dt])
    next_state = odeint(motion, last_state, next_time).T[:, 1:]
    state = np.hstack((state, next_state))  # Add new state to global state matrix

    # Trail calculations
    start_frame = max(0, frame - trail_length)
    if focus != "none":
        infocus = int(focus)
        focus_position = positions[infocus, frame, :2]
        if relative_trails[0]:
            points = positions[infocus, start_frame:frame + 1, :]
        else:
            points = np.tile(focus_position, (frame + 1 - start_frame, 1))
        focus_position = np.vstack([focus_position] * n)
    else:
        focus_position = np.zeros((n, 2))
        points = np.zeros((frame + 1 - start_frame, 3))

    # Update body positions (scatter)
    scat.set_offsets(positions[:, frame, :2] - focus_position)
    scat.set_facecolor(scat_colors)
    scat.set_sizes(10 * masses)

    # Update trail lines
    for i in range(n):
        lines[i].set_data(positions[i, start_frame:frame + 1, 0] - points[:, 0],
                          positions[i, start_frame:frame + 1, 1] - points[:, 1])
        lines[i].set_color(scat_colors[i])

    # Restore axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Display time and real-time FPS
    elapsed_time = frame * dt
    time_text.set_text(f'Time: {elapsed_time:.2f} days')
    real_time_per_frame = 1 / (time.time() - start_time)
    if frame % 5 == 0:
        real_time_text.set_text(f'FPS: {real_time_per_frame:.6f}')

    return scat, *lines, time_text, real_time_text

# optional, calculates total energy of system for error checking
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

# Infinite generator to keep animation running
def infinite_generator():
    frame = 0
    while True:
        yield frame
        frame += 1

# Callback for radio buttons to focus on a body
def radio_clicked(val):
    global focus
    focus = val

# Toggle relative trails
def toggle_trails(event):
    global relative_trails
    relative_trails[0] = not relative_trails[0]


trail_length = 250  # Number of frames to show in trail

# Initialize state with one step
timex = np.arange(0, dt, dt)
state = odeint(motion, state0, timex).T

# Create main figure
fig, ax = plt.subplots()
ax.set_xlim(-16, 16)
ax.set_ylim(-16, 16)

# Radio buttons to switch focus
radio_ax = plt.axes([0.05, 0.8, 0.15, 0.15])
radio = wgd.RadioButtons(radio_ax, ["none"] + [str(i) for i in range(n)])
radio.on_clicked(radio_clicked)

# Checkbox to toggle relative trails
button_ax = plt.axes([0.2, 0.85, 0.15, 0.1])
check = wgd.CheckButtons(button_ax, ['RT'], [False])
check.on_clicked(toggle_trails)

# Time and FPS text boxes
time_text = ax.text(0.7, 0.95, '', transform=ax.transAxes, fontsize=10)
real_time_text = ax.text(0.4, 0.95, '', transform=ax.transAxes, fontsize=10)

# Initialize scatter for bodies
scat = ax.scatter([], [], s=50)

# Initialize lines for trails
lines = [ax.plot([], [], lw=1)[0] for _ in range(n)]

# Define color for each body
scat_colors = ['red', 'blue', 'green', 'orange', 'purple']

# Create the animation
ani = animation.FuncAnimation(fig, update, 3650, interval=1, blit=True, repeat=False)

plt.show()


