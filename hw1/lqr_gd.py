import numpy as np

T = 20
nx = 2  
nu = 1  

A = np.array([[1, 1], 
              [0, 1]])
B = np.array([[0], 
              [1]])
x0 = np.array([[1], 
               [0]])

Q = np.eye(nx)
QT = 10 * np.eye(nx)
R = np.eye(nu)

A_tilde = np.zeros((nx * T, nx))
B_tilde = np.zeros((nx * T, nu * T))
Q_hat = np.zeros((nx * T, nx * T))
R_hat = np.zeros((nu * T, nu * T))

# fill matrices 
for t in range(1, T + 1):
    r_start = (t - 1) * nx
    r_end = t * nx
    
    A_tilde[r_start:r_end, :] = np.linalg.matrix_power(A, t)
    if t == T:
        Q_hat[r_start:r_end, r_start:r_end] = QT
    else:
        Q_hat[r_start:r_end, r_start:r_end] = Q
        
    for k in range(t):
        c_start = k * nu
        c_end = (k + 1) * nu
        B_tilde[r_start:r_end, c_start:c_end] = np.linalg.matrix_power(A, t - k - 1) @ B

for k in range(T):
    c_start = k * nu
    c_end = (k + 1) * nu
    R_hat[c_start:c_end, c_start:c_end] = R

Q_tilde = 2 * (B_tilde.T @ Q_hat @ B_tilde + R_hat)
b_tilde = -2 * (B_tilde.T @ Q_hat @ A_tilde @ x0)

# gradient descent 
u = np.zeros((nu * T, 1))
max_iter = 1000

for iteration in range(max_iter):
    grad = Q_tilde @ u - b_tilde
    if np.linalg.norm(grad) < 1e-6: break # we've reached optimal point 
    
    d = -grad # descent direction 
    eta = (d.T @ d) / (d.T @ Q_tilde @ d)
    u = u + eta * d # update 

# get optimal cost J(u*)
optimal_cost = (0.5 * u.T @ Q_tilde @ u - b_tilde.T @ u + (x0.T @ Q @ x0) + (A_tilde @ x0).T @ Q_hat @ (A_tilde @ x0))[0, 0] 

print(f"Optimal Cost J(u*): {optimal_cost:.4f}")
print(f"optimal inputs u*: \n{u.flatten()}")
