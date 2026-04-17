# %% IMPORTS

# Fisher-KPP on Star Graph - General n_in, interval, n_out
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.rcParams['figure.dpi'] = 100
print("Fisher-KPP on star graph (general n_in, interval, n_out)")

# %% PARAMETERS

L = 20.0
N = 400
T = 7
dt = 0.001
dx = L / N
r = dt / (dx**2)

print(f"dx = {dx}, dt = {dt}")
print(f"CFL parameter r = {r:.4f}")
if r > 0.5:
    print("WARNING: r > 0.5, might be unstable")
else:
    print("r < 0.5, should be stable")

# %% GRID

n_in = 4
n_interval = 1
n_out = 3

x_incoming = np.linspace(-L, -1, N+1)
x_interval = np.linspace(-1, 0, N+1)
x_outgoing = np.linspace(0, L, N+1)

# %% INITIAL CONDITIONS

def step_initial_condition(n_in, n_interval, n_out):
    u_in = [np.ones(N+1) for _ in range(n_in)]
    u_interval = [np.ones(N+1) for _ in range(n_interval)]
    u_out = [np.zeros(N+1) for _ in range(n_out)]
    return u_in, u_interval, u_out

def exponential_initial_condition(n_in, n_interval, n_out, lam=1.0):
    u_in = [np.ones(N+1) for _ in range(n_in)]
    u_interval = [np.exp(-lam * (x_interval + 1)) for _ in range(n_interval)]
    u_out = [np.exp(-lam * x_outgoing) for _ in range(n_out)]
    return u_in, u_interval, u_out

# %% BOUNDARY & VERTEX CONDITIONS

def apply_neumann_bc(u_in, u_interval, u_out):
    for u in u_in:
        u[0] = u[1]
    for u in u_interval:
        u[0] = u[1]
    for u in u_out:
        u[-1] = u[-2]

def apply_vertex_conditions(u_in, u_interval, u_out):
    """continuity + Kirchhoff (flux balance) at x = -1 and x = 0"""

    # ---- Vertex at x = -1 ----
    # Continuity
    u_vertex_left = (sum(u[-1] for u in u_in) + sum(u[0] for u in u_interval)) / (len(u_in) + len(u_interval))

    for u in u_in:
        u[-1] = u_vertex_left
    for u in u_interval:
        u[0] = u_vertex_left

    # Kirchhoff (flux balance)
    flux_in = sum((u[-1] - u[-2]) / dx for u in u_in)
    flux_out = sum((u[1] - u[0]) / dx for u in u_interval)

    correction = (flux_in - flux_out) / (len(u_in) + len(u_interval))

    for u in u_in:
        u[-2] += correction * dx
    for u in u_interval:
        u[1] -= correction * dx

    # ---- Vertex at x = 0 ----
    # Continuity
    u_vertex_right = (sum(u[-1] for u in u_interval) + sum(u[0] for u in u_out)) / (len(u_interval) + len(u_out))

    for u in u_interval:
        u[-1] = u_vertex_right
    for u in u_out:
        u[0] = u_vertex_right

    # Kirchhoff (flux balance)
    flux_in = sum((u[-1] - u[-2]) / dx for u in u_interval)
    flux_out = sum((u[1] - u[0]) / dx for u in u_out)

    correction = (flux_in - flux_out) / (len(u_interval) + len(u_out))

    for u in u_interval:
        u[-2] += correction * dx
    for u in u_out:
        u[1] -= correction * dx

# %% FTCS STEP

def ftcs_step(u, r, dt):
    u_new = np.zeros_like(u)
    N = len(u)-1
    for j in range(1, N):
        diffusion = r * (u[j+1] - 2*u[j] + u[j-1])
        reaction = dt * u[j] * (1.0 - u[j])
        u_new[j] = u[j] + diffusion + reaction
    return u_new

# %% SIMULATION

def run_simulation(u_in, u_interval, u_out, L=L, N=N, T=T, dt=dt, save_every=50):
    r = dt / (dx**2)
    n_steps = int(T / dt)
    snapshots = [([u.copy() for u in u_in], [u.copy() for u in u_interval], [u.copy() for u in u_out])]
    times = [0.0]

    for n in range(n_steps):
        apply_neumann_bc(u_in, u_interval, u_out)
        apply_vertex_conditions(u_in, u_interval, u_out)

        u_in_new = [ftcs_step(u, r, dt) for u in u_in]
        u_interval_new = [ftcs_step(u, r, dt) for u in u_interval]
        u_out_new = [ftcs_step(u, r, dt) for u in u_out]

        for k in range(len(u_in)):
            u_in[k][1:N] = u_in_new[k][1:N]
        for k in range(len(u_interval)):
            u_interval[k][1:N] = u_interval_new[k][1:N]
        for k in range(len(u_out)):
            u_out[k][1:N] = u_out_new[k][1:N]

        if (n+1) % save_every == 0:
            snapshots.append(([u.copy() for u in u_in], [u.copy() for u in u_interval], [u.copy() for u in u_out]))
            times.append((n+1)*dt)

    return snapshots, times

# %% WAVE SPEED

def calculate_wave_speed(snapshots, times, x_outgoing, threshold=0.5):
    positions, valid_times = [], []
    for i, (u_in_snap, u_interval_snap, u_out_snap) in enumerate(snapshots):
        idx = np.where(u_out_snap[0] >= threshold)[0]
        if len(idx) > 0:
            positions.append(x_outgoing[idx[-1]])
            valid_times.append(times[i])
    if len(positions) > 1:
        speed = np.polyfit(valid_times, positions, 1)[0]
        return speed
    return 0.0

# %% PLOTTING FUNCTIONS

def plot_2d_edges(u_in, u_interval, u_out, x_incoming, x_interval, x_outgoing, t, title=""):
    plt.figure(figsize=(10,4))
    for i, u in enumerate(u_in):
        plt.plot(x_incoming, u, label=f'Edge {i+1} (in)')
    for i, u in enumerate(u_interval):
        plt.plot(x_interval, u, label=f'Interval {i+1}')
    for i, u in enumerate(u_out):
        plt.plot(x_outgoing, u, label=f'Edge {i+1+len(u_in)+len(u_interval)} (out)')
    plt.axvline(-1, color='gray', linestyle='--', alpha=0.5, label='Vertex -1')
    plt.axvline(0, color='k', linestyle='--', alpha=0.5, label='Vertex 0')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'{title} at t={t:.2f}')
    plt.ylim([-0.05, 1.15])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_3d_edges(u_in, u_interval, u_out, x_incoming, x_interval, x_outgoing, t, title=""):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    for i, u in enumerate(u_in):
        y = (i - (len(u_in)-1)/2) * x_incoming / 2
        ax.plot(x_incoming, y, u, linewidth=2)

    for i, u in enumerate(u_interval):
        y = (i - (len(u_interval)-1)/2) * x_interval / 4  # smaller spread
        ax.plot(x_interval, y, u, linewidth=2)

    angles = np.linspace(-np.pi/4, np.pi/4, len(u_out))
    for i, u in enumerate(u_out):
        theta = angles[i]
        x_coords = x_outgoing * np.cos(theta)
        y_coords = x_outgoing * np.sin(theta)
        ax.plot(x_coords, y_coords, u, linewidth=2)

    ax.scatter([0], [0], [np.mean([u[-1] for u in u_interval])], color='k', s=50)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(f'{title} at t={t:.2f}')
    ax.set_zlim([0,1.1])
    ax.view_init(elev=20, azim=45)
    plt.show()

# %% TASK 1: STEP-TYPE IC

u_in, u_interval, u_out = step_initial_condition(n_in, n_interval, n_out)
snapshots_task1, times_task1 = run_simulation(u_in, u_interval, u_out, save_every=50)
speed_task1 = calculate_wave_speed(snapshots_task1, times_task1, x_outgoing)
print(f"Task 1 wave speed: {speed_task1:.4f}")

plot_times = [0, 1.75, 3.50, 5.25, 7.0]
for t_val in plot_times:
    idx = min(range(len(times_task1)), key=lambda i: abs(times_task1[i]-t_val))
    u_in_snap, u_interval_snap, u_out_snap = snapshots_task1[idx]
    plot_2d_edges(u_in_snap, u_interval_snap, u_out_snap, x_incoming, x_interval, x_outgoing, times_task1[idx], title="Task 1 Step IC")

for t_val in [0,3,7]:
    idx = min(range(len(times_task1)), key=lambda i: abs(times_task1[i]-t_val))
    u_in_snap, u_interval_snap, u_out_snap = snapshots_task1[idx]
    plot_3d_edges(u_in_snap, u_interval_snap, u_out_snap, x_incoming, x_interval, x_outgoing, times_task1[idx], title="Task 1 Step IC 3D")

# %% TASK 2: EXPONENTIAL IC

lambda_values = [0.5, 1.0, 2.0]
results = {}
speeds = {}

for lam in lambda_values:
    u_in, u_interval, u_out = exponential_initial_condition(n_in, n_interval, n_out, lam)
    snapshots, times = run_simulation(u_in, u_interval, u_out, save_every=50)
    results[lam] = {'snapshots': snapshots, 'times': times}
    speeds[lam] = calculate_wave_speed(snapshots, times, x_outgoing)
    print(f"λ={lam} wave speed: {speeds[lam]:.4f}")

    for t_val in plot_times:
        idx = min(range(len(times)), key=lambda i: abs(times[i]-t_val))
        u_in_snap, u_interval_snap, u_out_snap = snapshots[idx]
        plot_2d_edges(u_in_snap, u_interval_snap, u_out_snap, x_incoming, x_interval, x_outgoing, times[idx], title=f"Task 2 λ={lam}")

    for t_val in [0,3,7]:
        idx = min(range(len(times)), key=lambda i: abs(times[i]-t_val))
        u_in_snap, u_interval_snap, u_out_snap = snapshots[idx]
        plot_3d_edges(u_in_snap, u_interval_snap, u_out_snap, x_incoming, x_interval, x_outgoing, times[idx], title=f"Task 2 λ={lam} 3D")

# %% ANIMATION λ=1.0 t=0 to 5

lam_anim = 1.0
snapshots_anim = results[lam_anim]['snapshots']
times_anim = results[lam_anim]['times']

max_idx = len([t for t in times_anim if t <= 5.0])
snapshots_anim_5 = snapshots_anim[:max_idx]
times_anim_5 = times_anim[:max_idx]

fig, ax = plt.subplots(figsize=(12,5))

def animate(frame):
    ax.clear()
    u_in_snap, u_interval_snap, u_out_snap = snapshots_anim_5[frame]
    t = times_anim_5[frame]
    for i, u in enumerate(u_in_snap):
        ax.plot(x_incoming, u, label=f'Edge {i+1} (in)')
    for i, u in enumerate(u_interval_snap):
        ax.plot(x_interval, u, label=f'Interval {i+1}')
    for i, u in enumerate(u_out_snap):
        ax.plot(x_outgoing, u, label=f'Edge {i+1+len(u_in_snap)+len(u_interval_snap)} (out)')
    ax.axvline(-1, color='gray', linestyle='--', alpha=0.5, label='Vertex -1')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='Vertex 0')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f'Animation λ={lam_anim} t={t:.3f}')
    ax.set_ylim([-0.05,1.15])
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax,

anim = FuncAnimation(fig, animate, frames=len(snapshots_anim_5), interval=50, blit=False, repeat=True)
plt.close()
HTML(anim.to_jshtml())