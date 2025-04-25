# N-Body Gravitational Simulation in Python

This project simulates gravitational interactions between multiple bodies in 2D space using Newtonian physics. It includes an animated visualization of body positions over time, complete with trails, interactive UI controls, and relative motion viewing options.

## Features

- Accurate n-body gravitational dynamics using 'scipy.integrate.odeint'
- 2D animation using 'matplotlib.animation'
- Trail rendering for each body
- UI controls for:
  - Toggling focus on specific bodies
  - Switching between absolute and relative trails
- Mass scaling and colour coding for visual clarity
- Efficient use of 'state' and 'positions' matrices to manage simulation data

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`
- `scipy`
