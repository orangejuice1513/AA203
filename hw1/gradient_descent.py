import numpy as np
import matplotlib.pyplot as plt

def compute_plot_bounds(paths):
    pts = np.vstack(paths)
    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))

    pad_x = 0.15 * max(1.0, x_max - x_min)
    pad_y = 0.15 * max(1.0, y_max - y_min)
    return (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)

def plot_level_curves(ax, gamma, bounds):
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, 300)
    ys = np.linspace(y_min, y_max, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (X**2 + float(gamma) * Y**2)

    z_min = max(1e-8, float(np.min(Z)))
    z_max = max(z_min * 10.0, float(np.max(Z)))
    levels = np.geomspace(z_min, z_max, 14)
    ax.contour(X, Y, Z, levels=levels, colors="0.75", linewidths=1.0, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

def place_legend_outside(ax):
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )

def solve_grad_desc():
    gammas = [10, 1]
    initial_values = [np.array([5.0, 1.0]), np.array([1.0, 5.0])]
    step_size = 0.15
    num_steps = 30

    for gamma in gammas:
        Q = np.array([[1.0, 0.0], [0.0, float(gamma)]])
        x_star = np.zeros(2)

        for x0 in initial_values:
            path_const = run_grad_desc(x0, Q, step_size, num_steps, is_exact=False)
            path_exact = run_grad_desc(x0, Q, step_size, num_steps, is_exact=True)
            
            # Two ~5x5 panels side-by-side (+ room for outside legends)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            bounds = compute_plot_bounds([path_const, path_exact, x_star[None, :]])
            plot_level_curves(ax1, gamma, bounds)
            plot_level_curves(ax2, gamma, bounds)
            
            ax1.plot(path_const[:, 0], path_const[:, 1], 'ro-', markersize=4, label='Path')
            ax1.plot(x_star[0], x_star[1], 'k*', markersize=10, label='Optimal x*')
            ax1.set_title(f"Constant Step (\u03B3={gamma}, x0={x0.tolist()})")
            ax1.grid(True, linestyle='--', alpha=0.6)
            place_legend_outside(ax1)
            
            ax2.plot(path_exact[:, 0], path_exact[:, 1], 'bo-', markersize=4, label='Path')
            ax2.plot(x_star[0], x_star[1], 'k*', markersize=10, label='Optimal x*') 
            ax2.set_title(f"Exact Line Search (\u03B3={gamma}, x0={x0.tolist()})")
            ax2.grid(True, linestyle='--', alpha=0.6)
            place_legend_outside(ax2)
            
            plt.tight_layout()
            plt.show()

def run_grad_desc(x0, Q, step_size, num_steps, is_exact=False):
    # runs gradient descent and returns trajectory path
    path = [x0.copy()]
    x = x0.copy()

    for _ in range(num_steps):
        grad = Q @ x # b = 0 
        dist_dir = -grad
        
        if np.linalg.norm(grad) < 1e-6: break # we've reached optimal point 
            
        if is_exact:
            eta = float(np.dot(dist_dir, dist_dir) / np.dot(dist_dir, Q @ dist_dir)) 
        else:
            eta = step_size
            
        x = x + eta * dist_dir
        path.append(x.copy())
        
    return np.array(path)

if __name__ == "__main__":
    solve_grad_desc()
