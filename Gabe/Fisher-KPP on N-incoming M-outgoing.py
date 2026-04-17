# %% IMPORTS

# Fisher-KPP on Star Graph - General n_in, n_out
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.rcParams['figure.dpi'] = 100
print("Fisher-KPP on star graph (general n_in, n_out)")

# %% PARAMETERS

L = 20.0        # edge length
N = 400         # grid points per edge
T = 7         # total simulation time
dt = 0.001      # time step
dx = L / N
r = dt / (dx**2)

print(f"dx = {dx}, dt = {dt}")
print(f"CFL parameter r = {r:.4f}")
if r > 0.5:
    print("WARNING: r > 0.5, might be unstable")
else:
    print("r < 0.5, should be stable")

# %% GRID

n_in = 4       # number of incoming edges
n_out = 3      # number of outgoing edges

# grids for each edge
x_incoming = np.linspace(-L, 0, N+1)
x_outgoing = np.linspace(0, L, N+1)

# %% INITIAL CONDITIONS

def step_initial_condition(n_in, n_out):
    u_in = [np.ones(N+1) for _ in range(n_in)]
    u_out = [np.zeros(N+1) for _ in range(n_out)]
    return u_in, u_out

def exponential_initial_condition(n_in, n_out, lam=1.0):
    u_in = [np.ones(N+1) for _ in range(n_in)]
    u_out = [np.exp(-lam * x_outgoing) for _ in range(n_out)]
    return u_in, u_out

# %% BOUNDARY & VERTEX CONDITIONS

def apply_neumann_bc(u_in, u_out):
    """no-flux at boundaries"""
    for u in u_in:
        u[0] = u[1]
    for u in u_out:
        u[-1] = u[-2]

def apply_vertex_conditions(u_in, u_out):
    """continuity and flux balance at vertex"""
    u_vertex = (sum([u[-2] for u in u_in]) + sum([u[1] for u in u_out])) / (len(u_in) + len(u_out))
    for u in u_in:
        u[-1] = u_vertex
    for u in u_out:
        u[0] = u_vertex

# %% FTCS STEP

def ftcs_step(u, r, dt):
    """one FTCS step"""
    u_new = np.zeros_like(u)
    N = len(u)-1
    for j in range(1, N):
        diffusion = r * (u[j+1] - 2*u[j] + u[j-1])
        reaction = dt * u[j] * (1.0 - u[j])
        u_new[j] = u[j] + diffusion + reaction
    return u_new

# %% SIMULATION

def run_simulation(u_in, u_out, L=L, N=N, T=T, dt=dt, save_every=50):
    r = dt / (dx**2)
    n_steps = int(T / dt)
    snapshots = [( [u.copy() for u in u_in], [u.copy() for u in u_out] )]
    times = [0.0]

    for n in range(n_steps):
        apply_neumann_bc(u_in, u_out)
        apply_vertex_conditions(u_in, u_out)

        u_in_new = [ftcs_step(u, r, dt) for u in u_in]
        u_out_new = [ftcs_step(u, r, dt) for u in u_out]

        for k in range(len(u_in)):
            u_in[k][1:N] = u_in_new[k][1:N]
        for k in range(len(u_out)):
            u_out[k][1:N] = u_out_new[k][1:N]

        if (n+1) % save_every == 0:
            snapshots.append( ([u.copy() for u in u_in], [u.copy() for u in u_out]) )
            times.append((n+1)*dt)

    return snapshots, times

# %% WAVE SPEED

def calculate_wave_speed(snapshots, times, x_outgoing, threshold=0.5):
    positions, valid_times = [], []
    for i, (u_in_snap, u_out_snap) in enumerate(snapshots):
        # track first outgoing edge only (can average over edges if desired)
        idx = np.where(u_out_snap[0] >= threshold)[0]
        if len(idx) > 0:
            positions.append(x_outgoing[idx[-1]])
            valid_times.append(times[i])
    if len(positions) > 1:
        speed = np.polyfit(valid_times, positions, 1)[0]
        return speed
    return 0.0

# %% PLOTTING FUNCTIONS

def plot_2d_edges(u_in, u_out, x_incoming, x_outgoing, t, title=""):
    plt.figure(figsize=(10,4))
    for i, u in enumerate(u_in):
        plt.plot(x_incoming, u, label=f'Edge {i+1} (in)')
    for i, u in enumerate(u_out):
        plt.plot(x_outgoing, u, label=f'Edge {i+1+n_in} (out)')
    plt.axvline(0, color='k', linestyle='--', alpha=0.5, label='Vertex')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'{title} at t={t:.2f}')
    plt.ylim([-0.05, 1.15])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_3d_edges(u_in, u_out, x_incoming, x_outgoing, t, title=""):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # arrange incoming edges diagonally
    for i, u in enumerate(u_in):
        y = (i - (len(u_in)-1)/2) * x_incoming / 2
        ax.plot(x_incoming, y, u, linewidth=2)

    # arrange outgoing edges horizontally
    angles = np.linspace(-np.pi/4, np.pi/4, len(u_out))  # spread angles

    for i, u in enumerate(u_out):
        theta = angles[i]
        x_coords = x_outgoing * np.cos(theta)
        y_coords = x_outgoing * np.sin(theta)
        ax.plot(x_coords, y_coords, u, linewidth=2)

    # vertex
    ax.scatter([0], [0], [np.mean([u[-1] for u in u_in])], color='k', s=50)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(f'{title} at t={t:.2f}')
    ax.set_zlim([0,1.1])
    ax.view_init(elev=20, azim=45)
    plt.show()


# %% TASK 1: STEP-TYPE IC

u_in, u_out = step_initial_condition(n_in, n_out)
snapshots_task1, times_task1 = run_simulation(u_in, u_out, save_every=50)
speed_task1 = calculate_wave_speed(snapshots_task1, times_task1, x_outgoing)
print(f"Task 1 wave speed: {speed_task1:.4f}")

# 2D plots at specific times
plot_times = [0, 1.75, 3.50, 5.25, 7.0]
for t_val in plot_times:
    idx = min(range(len(times_task1)), key=lambda i: abs(times_task1[i]-t_val))
    u_in_snap, u_out_snap = snapshots_task1[idx]
    plot_2d_edges(u_in_snap, u_out_snap, x_incoming, x_outgoing, times_task1[idx], title="Task 1 Step IC")

# 3D plots at t=0,3,7
for t_val in [0,3,7]:
    idx = min(range(len(times_task1)), key=lambda i: abs(times_task1[i]-t_val))
    u_in_snap, u_out_snap = snapshots_task1[idx]
    plot_3d_edges(u_in_snap, u_out_snap, x_incoming, x_outgoing, times_task1[idx], title="Task 1 Step IC 3D")


# %% TASK 2: EXPONENTIAL IC

lambda_values = [0.5, 1.0, 2.0]
results = {}
speeds = {}

for lam in lambda_values:
    u_in, u_out = exponential_initial_condition(n_in, n_out, lam)
    snapshots, times = run_simulation(u_in, u_out, save_every=50)
    results[lam] = {'snapshots': snapshots, 'times': times}
    speeds[lam] = calculate_wave_speed(snapshots, times, x_outgoing)
    print(f"λ={lam} wave speed: {speeds[lam]:.4f}")

    # 2D plots at specified times
    for t_val in plot_times:
        idx = min(range(len(times)), key=lambda i: abs(times[i]-t_val))
        u_in_snap, u_out_snap = snapshots[idx]
        plot_2d_edges(u_in_snap, u_out_snap, x_incoming, x_outgoing, times[idx], title=f"Task 2 λ={lam}")

    # 3D plots at t=0,3,7
    for t_val in [0,3,7]:
        idx = min(range(len(times)), key=lambda i: abs(times[i]-t_val))
        u_in_snap, u_out_snap = snapshots[idx]
        plot_3d_edges(u_in_snap, u_out_snap, x_incoming, x_outgoing, times[idx], title=f"Task 2 λ={lam} 3D")


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
    u_in_snap, u_out_snap = snapshots_anim_5[frame]
    t = times_anim_5[frame]
    for i, u in enumerate(u_in_snap):
        ax.plot(x_incoming, u, label=f'Edge {i+1} (in)')
    for i, u in enumerate(u_out_snap):
        ax.plot(x_outgoing, u, label=f'Edge {i+1+n_in} (out)')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='Vertex')
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