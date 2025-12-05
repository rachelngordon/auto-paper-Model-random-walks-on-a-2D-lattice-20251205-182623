# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def random_walk(num_steps):
    # steps: 0=up,1=down,2=left,3=right
    moves = np.random.randint(0,4,size=num_steps)
    dx = np.where(moves==2, -1, np.where(moves==3, 1, 0))
    dy = np.where(moves==0, 1, np.where(moves==1, -1, 0))
    x = np.concatenate(([0], np.cumsum(dx)))
    y = np.concatenate(([0], np.cumsum(dy)))
    return np.column_stack((x,y))

def experiment1():
    steps = 500
    traj = random_walk(steps)
    plt.figure(figsize=(6,6))
    plt.plot(traj[:,0], traj[:,1], '-o', markersize=3, linewidth=1, label='Path')
    plt.plot(traj[0,0], traj[0,1], 'go', markersize=8, label='Start')
    plt.plot(traj[-1,0], traj[-1,1], 'ro', markersize=8, label='End')
    plt.title('Single 2D Random Walk (500 steps)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('single_walk_trajectory.png')
    plt.close()

def experiment2():
    n_walks = 1000
    max_steps = 2000
    # generate steps for all walks up to max_steps
    moves = np.random.randint(0,4,size=(n_walks, max_steps))
    dx = np.where(moves==2, -1, np.where(moves==3, 1, 0))
    dy = np.where(moves==0, 1, np.where(moves==1, -1, 0))
    # cumulative positions
    x = np.cumsum(dx, axis=1)
    y = np.cumsum(dy, axis=1)
    # prepend origin
    x = np.hstack((np.zeros((n_walks,1), dtype=int), x))
    y = np.hstack((np.zeros((n_walks,1), dtype=int), y))
    # squared distance
    r2 = x**2 + y**2
    # compute MSD for selected steps (every 10 steps from 10 to 2000)
    step_vals = np.arange(10, max_steps+1, 10)
    msd = np.mean(r2[:, step_vals], axis=0)
    # plot
    plt.figure()
    plt.loglog(step_vals, msd, 'o', label='MSD')
    # reference line with slope 1
    ref = step_vals * (msd[0]/step_vals[0])
    plt.loglog(step_vals, ref, '--', label='Slope 1')
    plt.xlabel('Number of steps')
    plt.ylabel('Mean squared displacement')
    plt.title('Mean Squared Displacement vs Steps')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('msd_vs_steps.png')
    plt.close()
    # fit slope on log‑log data
    coeffs = np.polyfit(np.log(step_vals), np.log(msd), 1)
    slope = coeffs[0]
    return slope

def main():
    experiment1()
    slope = experiment2()
    print('Answer:', slope)

if __name__ == '__main__':
    main()

