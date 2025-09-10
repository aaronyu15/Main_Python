import matplotlib.pyplot as plt
import numpy as np
import math

def LIF_RC(v_start, v_rest, delta_t, alpha=0.1):
    """
    Parameters
    ----------
    v_start : float
        Initial membrane potential.
    v_rest : float
        Resting membrane potential.
    delta_t : float
        Time since last spike.
    alpha : float, optional
        Decay constant, by default 0.1
    """
    return v_rest + (v_start - v_rest) * math.exp(- alpha * delta_t )

def LIF(v_start, v_rest, delta_t,  I, alpha=0.1):
    """
    Parameters
    ----------
    v_start : float
        Initial membrane potential.
    v_rest : float
        Resting membrane potential.
    delta_t : float
        Timestep of iterative / Euler approximation.
    I : float
        Input current.
    alpha : float, optional
        Decay constant, by default 0.1
        """
    return v_start + delta_t / alpha * (-(v_start - v_rest) + I)

def membrane_potential_evolution(v_start, v_rest, delta_t, I, alpha=0.1, timesteps=100, method='exact'):
    v = v_start
    v_list = [v]

    if not isinstance(I, np.ndarray):
        I = np.full(timesteps, I)
    elif len(I) < timesteps:
        I = np.pad(I, (0, timesteps - len(I)), 'constant', constant_values=0)

    if method == 'exact':
        for t in range(timesteps):
            v = LIF(v_start=v, v_rest=v_rest, delta_t=delta_t, I=I[t], alpha=alpha)
            v_list.append(v)

    elif method == 'approx':
        time_of_last_spike = 0
        for t in range(timesteps):
            delta_t = t - time_of_last_spike
            if I[t] != 0:
                v += alpha * I[t]
                v_start = v
                time_of_last_spike = t
            else:
                v = LIF_RC(v_start=v_start, v_rest=v_rest, delta_t=delta_t, alpha=alpha)
            v_list.append(v)

    return v_list

def points_q(a, b, t, x):
    num_bits = 4
    num_values = 2 ** num_bits

    x_vals = np.array([i*t for i in range(num_values)])
    y_vals = a * x_vals **2 + b * x_vals

    return x_vals, y_vals, y_vals[x]

if __name__ == "__main__":

    hist = {}

    for D in range(16):
        for Q in range(16):
            newQ = Q ^ D
            print(f"D: {D:04b}, Q: {Q:04b}, new Q: {newQ:04b}")
            hist[newQ] = hist.get(newQ, 0) + 1

    print("=========== HISTOGRAM ===========")
    print(hist)



    #timesteps = 100
    ## Example usage of membrane potential evolution
    #v_rest = 0.5
    #v_start = 1.0
    #alpha = 0.1
    #I = np.zeros(100)  # No input current
    #I[20] = 1.0  # Apply a current at timestep 20
    #I[60] = 5.0  # Apply a current at timestep 20
    #I[80] = -5.0  # Apply a current at timestep 20

    #v_list1 = membrane_potential_evolution(v_start=v_start, v_rest=v_rest, delta_t=0.01, I=I, alpha=alpha, timesteps=timesteps, method='exact')

    #v_list2 = membrane_potential_evolution(v_start=v_start, v_rest=v_rest, delta_t=None, I=I, alpha=alpha, timesteps=timesteps, method='approx')

    #plt.plot(v_list1)
    #plt.plot(v_list2, color='red', linestyle='--')
    #plt.title("Membrane Potential Evolution")
    #plt.xlabel("Time Steps")
    #plt.ylabel("Membrane Potential")
    #plt.grid(True)
    #plt.show()

    #exit()

    ## Example usage of points_q
    #x = 10
    #a = 2
    #b = 3
    #t = 0.5

    #x_vals, y_vals, y = points_q(a, b, t, x)
    #fig = plt.figure()

    #plt.scatter(x_vals, y_vals, color='red', label='Point of Interest')
    #plt.xlabel('Time Steps')
    #plt.ylabel('Value')
    #plt.title('Line Plot with Points')
    #plt.grid(True)
    #plt.show()

