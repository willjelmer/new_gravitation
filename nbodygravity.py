import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.widgets as wgd
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.lines import Line2D


def run_simulation(positions, velocities, masses, t_span, dt = 0.001, big_g = 1, epsilon = 0):
    '''
    Runs a simulation for a given set of initial positions, velocities and masses, for a given time range.

    Parameters:
    -----------
    positions: array, shape (n_bodies, 3)
        Initial positions of the bodies in 3D space.

    velocities: array, shape (n_bodies, 3)
        Initial velocities of the bodies in 3D space.

    masses: list, length = n_bodies
        The masses of the bodies

    t_span: float
        The total time span for the simulation.

    dt: float, default = 0.001
        The time step for integration. Smaller value => more precise calculation

    big_g: float, default = 1
        The gravitational constant used in force calculations

    epsilon: float, default = 0
        Force softening constant, prevents divisions by zero or excessive force when bodies are very close

    Returns:
    --------
    state: array, shape (n_bodies * 6, n_frames)
        The state of the system over time.
    '''
    def motion(y, t):
        '''
         Compute the derivatives of position and velocity for an n-body gravitational system.
         This function calculates the time derivatives of the systems state vector "y", which contains
         the positions and velocities of all bodies. It uses vectorised operations to efficiently
         compute the accelerations between each body due to all others
        '''

        # Gather positions and velocities from y vector
        positions = y[:n_bodies * 3].reshape((n_bodies, 3)) # needs reshaping to be used in calculations
        velocities = y[n_bodies * 3:]

        # Compute displacement vectors between pairs of bodies
        disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

        # Compute distances between each pair
        r_mags = np.linalg.norm(disp, axis=-1)
        np.fill_diagonal(r_mags, np.infty)
        inv3 = 1 / (r_mags ** 3 + epsilon ** 3)

        # Calculate magnitudes of forces between each pair
        force_magnitudes = big_g * (mass_matrix @ inv3)
        force_magnitudes = force_magnitudes[:, :, np.newaxis]

        # Compute vector acceleration contributions for each body and sum
        vdot_matrix = force_magnitudes * disp
        column_sums = np.sum(vdot_matrix, axis=0)

        vdot = column_sums.flatten()
        return np.concatenate((velocities, vdot))

    masses = np.array(masses)
    mass_matrix = np.diag(masses) # Used in motion calculation
    n_bodies = len(masses)

    state0 = np.concatenate((positions, velocities)).flatten()
    time_range = np.arange(0, t_span, dt)
    state = odeint(motion, state0, time_range).T
    return state

def animate_state(state, dt, trail_length=250, body_names = None, dt_resolution = 1):
    '''
    This function animates the x and y coordinates of the state vector using matplotlib.animation.FuncAnimation

    :param state: list of numpy arrays
        Contains all simulation data for each frame.

    :param dt: float
        The time step between frames in the simulation.

    :param trail_length: int, default = 250
        The number of frames to display for each body's trail.

    :param body_names: list of str, default = None
        A list of names for each body, if provided, these are used in the UI.

    :param dt_resolution: int, default = 1
        The resolution factor for animated time steps.
    '''

    relative_trails = [False]  # Mutable object used to toggle relative trails from the UI
    focus = ["none"]  # Focused object for relative plotting (can be changed via UI)

    n_frames = int(len(state[0]) / dt_resolution)
    n_bodies = int(len(state) / 6)

    if body_names:
        name_to_index = {name: idx for idx, name in enumerate(body_names)}
    else:
        name_to_index = {str(i): i for i in range(0,n_bodies)}
        body_names = [str(i) for i in range(0,n_bodies)]

    positions = np.zeros((n_bodies, n_frames, 3))  # Initialise empty position matrix

    # Callback for radio buttons to focus on a body
    def radio_clicked(val):
        if val == "None":
            focus[0] = "none"
        else:
            focus[0] = name_to_index[val]

    # Toggle relative trails
    def toggle_trails(event):
        relative_trails[0] = not relative_trails[0]

    def update(frame):
        '''
        Function required by matplotlib.animation to run at each frame
        - Updates global positions matrix with current frame's coordinates
        - Advances simulation by one timestep
        - Updates plot, drawing scatter for current positions and lines for trails
        - Display current time step as text overlay
        '''
        # Store current axis limits so we can keep zoom constant
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.cla()  # Clear previous frame’s drawings


        # Add coordinates to position matrix
        for i in range(n_bodies):
            positions[i, frame] = [
                state[3 * i][::dt_resolution][frame],
                state[3 * i + 1][::dt_resolution][frame],
                state[3 * i + 2][::dt_resolution][frame],
            ]

        # Trail calculations
        start_frame = max(0, frame - trail_length)
        if focus[0] != "none":
            body_in_focus = int(focus[0])
            focus_position = positions[body_in_focus, frame, :2]
            if relative_trails[0]:
                points = positions[body_in_focus, start_frame:frame + 1, :]
            else:
                points = np.tile(focus_position, (frame + 1 - start_frame, 1))
            focus_position = np.vstack([focus_position] * n_bodies)
        else:
            focus_position = np.zeros((n_bodies, 2))
            points = np.zeros((frame + 1 - start_frame, 3))

        # Update body positions (scatter)
        scat.set_offsets(positions[:, frame, :2] - focus_position)
        scat.set_facecolor(scat_colors)
        #scat.set_sizes(10)

        # Update trail lines
        for i in range(n_bodies):
            lines[i].set_data(positions[i, start_frame:frame + 1, 0] - points[:, 0],
                              positions[i, start_frame:frame + 1, 1] - points[:, 1])
            lines[i].set_color(scat_colors[i])

        # Restore axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Display time
        elapsed_time = frame * dt * dt_resolution
        time_text.set_text(f'Time: {elapsed_time:.2f} days')

        # Redraw legend
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                  label=name, markerfacecolor=color, markersize=8)
                           for name, color in zip(body_names, scat_colors)]
        ax.legend(handles=legend_elements, loc='upper right')

        return scat, *lines, time_text

    # Create main figure
    fig, ax = plt.subplots()
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")

    # Define color for each body
    cmap = cm.get_cmap('nipy_spectral', n_bodies)
    scat_colors = [cmap(i) for i in range(n_bodies)]

    # Radio buttons to switch focus
    radio_ax = plt.axes([0.05, 0.8, 0.15, 0.15])
    radio_ax.text(0.5, 1.05, "Focus", transform=radio_ax.transAxes,
                  ha="center", va="center", fontsize=10)
    radio = wgd.RadioButtons(radio_ax, ["None"] + body_names)
    radio.on_clicked(radio_clicked)

    # Checkbox to toggle relative trails
    button_ax = plt.axes([0.2, 0.85, 0.05, 0.05])
    check = wgd.CheckButtons(button_ax, ['RT'], [False])
    check.on_clicked(toggle_trails)

    # Time and FPS text boxes
    time_text = ax.text(0.7, 0.95, '', transform=ax.transAxes, fontsize=10)

    # Initialize scatter for bodies
    scat = ax.scatter([], [], s=50)

    # Initialize lines for trails
    lines = [ax.plot([], [], lw=1)[0] for _ in range(n_bodies)]

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=0, blit=True, repeat=False)
    plt.show()