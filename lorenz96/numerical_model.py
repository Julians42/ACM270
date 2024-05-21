import numpy as np

def lorenz96(t, x, F=8):
    """Lorenz 96 model with constant forcing"""
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

# def lorenz96_twoscale(t, u, N=40, n=5, F=8):
#   dx = np.zeros(N)
#   dy = np.zeros(n, N)

#   u = np.reshape(u, n + 1, N)
#   x = u[1, :]
#   y = u[2:end, :]

#   for i in range(1, N+1):
#     dx[i] = (x[mod(i+1, 1:N)] - x[mod(i-2, 1:N)])*x[mod(i-1, 1:N)] - x[i] + F - p["h"]*p["c"]/p["b"]*sum(y[:, i])

#     for j=1:n
#         if j == n
#           jp1 = 1
#           jp2 = 2
#           jm1 = n - 1
#           ip1 = mod(i + 1, 1:N)
#           ip2 = mod(i + 1, 1:N)
#           im1 = i
#         elseif j == n - 1
#           jp1 = n
#           jp2 = 1
#           jm1 = n - 2
#           ip1 = i
#           ip2 = mod(i + 1, 1:N)
#           im1 = i
#         elseif  j == 1
#           jp1 = 2
#           jp2 = 3
#           jm1 = n
#           ip1 = i
#           ip2 = i
#           im1 = mod(i - 1, 1:N)
#         else
#           jp1 = j + 1
#           jp2 = j + 2
#           jm1 = j - 1
#           ip1 = ip2 = im1 = i
#         end
#         dy[j, i] = p["c"]*p["b"]*y[jp1, ip1]*(y[jm1, im1] - y[jp2, ip2]) - p["c"]*y[j, i] + p["h"]*p["c"]/p["b"]*x[i]
#     end
#   end

#   du = vec([dx, dy])

#   return du
# end

def lorenz96_twoscale(t, u, N=40, n=5, F=8):
    dx = np.zeros(N)
    dy = np.zeros((n, N))  # Corrected shape definition

    u = np.reshape(u, (n + 1, N))  # Corrected shape definition
    x = u[0, :]  # Corrected indexing
    y = u[1:, :]  # Corrected indexing

    for i in range(N):  # Corrected range definition
        dx[i] = (x[(i+1) % N] - x[(i-2) % N]) * x[(i-1) % N] - x[i] + F - p["h"] * p["c"] / p["b"] * sum(y[:, i])

        for j in range(n):  # Corrected range definition
            if j == n - 1:
                jp1 = 0
                jp2 = 1
                jm1 = n - 2
                ip1 = i
                ip2 = (i + 1) % N
                im1 = i
            elif j == n - 2:
                jp1 = n - 1
                jp2 = 0
                jm1 = n - 3
                ip1 = i
                ip2 = (i + 1) % N
                im1 = i
            elif j == 0:
                jp1 = 1
                jp2 = 2
                jm1 = n - 1
                ip1 = i
                ip2 = i
                im1 = (i - 1) % N
            else:
                jp1 = j + 1
                jp2 = j + 2
                jm1 = j - 1
                ip1 = ip2 = im1 = i

            dy[j, i] = p["c"] * p["b"] * y[jp1, ip1] * (y[jm1, im1] - y[jp2, ip2]) - p["c"] * y[j, i] + p["h"] * p["c"] / p["b"] * x[i]

    du = np.reshape([dx, dy], -1)

    return du

