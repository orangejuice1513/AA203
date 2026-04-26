import numpy as np
import matplotlib.pyplot as plt

def solve_grad_desc():
    gammas = [10, 1]
    initial_values = [np.array([5.0, 1.0]), np.array([1.0, 5.0])]
    step_size = 0.5
    num_steps = 8

    for gamma in gammas:
        Q = np.array([[1.0, 0.0], [0.0, float(gamma)]])

        for x0 in initial_values:
            path_const = run_grad_desc(x0, Q, step_size, num_steps, is_exact=False)
            path_exact = run_grad_desc(x0, Q, step_size, num_steps, is_exact=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(path_const[:, 0], path_const[:, 1], 'ro-', markersize=4, label='Path')
            ax1.plot(0, 0, 'k*', markersize=10, label='Optimal x*')
            ax1.set_title(f"Constant Step (\u03B3={gamma}, x0={x0.tolist()})")
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend()
            
            ax2.plot(path_exact[:, 0], path_exact[:, 1], 'bo-', markersize=4, label='Path')
            ax2.plot(0, 0, 'k*', markersize=10, label='Optimal x*')
            ax2.set_title(f"Exact Line Search (\u03B3={gamma}, x0={x0.tolist()})")
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()

def run_grad_desc(x0, Q, step_size, num_steps, is_exact=False):
    # runs gradient descent and returns trajectory path
    path = [x0.copy()]
    x = x0.copy()

    for _ in range(num_steps):
        grad = Q @ x 
        dist_dir = -grad
        
        if np.linalg.norm(grad) < 1e-6: break # we've reached optimal point 
            
        if is_exact:
            numerator = np.dot(dist_dir, dist_dir)
            denominator = np.dot(dist_dir, Q @ dist_dir)
            eta = float(numerator / denominator) 
        else:
            eta = step_size
            
        x = x + eta * dist_dir
        path.append(x.copy())
        
    return np.array(path)

if __name__ == "__main__":
    solve_grad_desc()
