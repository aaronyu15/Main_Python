import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
from scipy.linalg import hadamard
import tonic
from torch.utils.data import DataLoader
from NMNIST_Dataset import NMNIST_Dataset  # custom dataset wrapper
from model_train import collate_frames



def walsh_matrix(n_bits):
    """Generate an unnormalized Walsh-Hadamard matrix of size 2^n."""
    H = np.array([[1]])
    for _ in range(n_bits):
        H = np.block([[H, H], [H, -H]])
    return H


# Example
H3 = walsh_matrix(3)


def walsh_transform(f):
    """Compute Walsh–Fourier transform for f (assumes f in {-1,1})."""
    N = len(f)
    H = walsh_matrix(int(np.log2(N)))
    return H @ f / N  # normalized transform


# Example: f(x) = XOR(x0, x1, x2)
f = np.array([1, 0, 1, 0 , 0, 0, 1, 0])  # outputs for all 3-bit inputs
print(walsh_transform(f))


def inverse_walsh_transform(coeffs):
    """Reconstruct function from Walsh coefficients."""
    N = len(coeffs)
    H = walsh_matrix(int(np.log2(N)))
    return H @ coeffs


def parity(x, s):
    """Compute parity (-1)^(x·s) for two bit vectors."""
    return (-1) ** (np.bitwise_xor.reduce(x & s))


# Example
x = np.array([1, 0, 1])  # input bits
s = np.array([1, 1, 0])  # subset bits


def walsh_spectrum(f):
    """Return absolute value spectrum (like FFT magnitude)."""
    coeffs = walsh_transform(f)
    return np.abs(coeffs)


def walsh_fourier_expansion(f_values):
    """
    Compute and display the Walsh–Fourier expansion for a function f:{-1,1}^n -> R.

    Parameters
    ----------
    f_values : list or np.ndarray
        Array of length 2^n containing the real outputs of f(x)
        in lexicographic order of x ∈ {-1,1}^n.
        Example ordering for n=3:
            [+1,+1,+1], [+1,+1,-1], [+1,-1,+1], [+1,-1,-1],
            [-1,+1,+1], [-1,+1,-1], [-1,-1,+1], [-1,-1,-1]
    """

    f_values = np.array(f_values, dtype=float)
    N = len(f_values)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("Length of f_values must be a power of 2.")

    # Generate all inputs x ∈ {-1,1}^n
    X = np.array(list(product([1, -1], repeat=n)))

    # Compute all subsets S of [n]
    subsets = [tuple(S) for r in range(n + 1) for S in combinations(range(n), r)]

    # Compute coefficients
    coeffs = {}
    for S in subsets:
        chi_S = np.prod(X[:, S], axis=1) if len(S) > 0 else np.ones(N)
        coeffs[S] = (1 / N) * np.sum(f_values * chi_S)

    # Display expansion
    print(f"Walsh–Fourier Expansion for n={n}:")
    print("f(x₁,…,xₙ) = ", end="")
    terms = []
    for S, c in coeffs.items():
        if abs(c) < 1e-12:
            continue  # skip zeros
        if len(S) == 0:
            term = f"{c:.4f}"
        else:
            factors = "".join([f"x{i+1}" for i in S])
            term = f"{c:+.4f}·{factors}"
        terms.append(term)
    print(" ".join(terms))

    # Return dict of coefficients
    return coeffs


def walsh_evaluate(x_bits, coeffs):
    """
    Evaluate f(x) using Walsh–Fourier coefficients.

    Parameters
    ----------
    x_bits : list or np.ndarray
        Input vector of length n, with each element in {1, -1}.
    coeffs : dict
        Dictionary of {subset: coefficient} from walsh_fourier_expansion().

    Returns
    -------
    float
        Real-valued result f(x).
    """
    x_bits = np.array(x_bits, dtype=float)
    n = len(x_bits)

    # sanity check
    if not all(abs(v) in (1, 0) or v in (1, -1) for v in x_bits):
        raise ValueError("x_bits must contain ±1 values.")

    result = 0.0
    for S, c in coeffs.items():
        if len(S) == 0:
            result += c
        else:
            result += c * np.prod(x_bits[list(S)])
    return result


f_values = list(range(0, 128))
coeffs = walsh_fourier_expansion(f_values)
res = walsh_evaluate([1, 1, 1, 1, 1, -1, -1], coeffs)
print(res)

new_coeffs = {}
for subset, coef in coeffs.items():
    if abs(coef) > 1e-12:
        new_coeffs[subset] = coef*2
        print(f"Subset {subset}: Coefficient {coef*2:.4f}")

res = walsh_evaluate([1, 1, 1, 1, 1, -1, -1], new_coeffs)
print(res)


#print("XOR")
#coeffs = walsh_fourier_expansion([1, -1, -1, 1])
#print("XNOR")
#coeffs = walsh_fourier_expansion([-1, 1, 1, -1])
#print("AND")
#coeffs = walsh_fourier_expansion([1, 1, 1, -1])
#print("NAND")
#coeffs = walsh_fourier_expansion([-1, -1, -1, 1])
#print("OR")
#coeffs = walsh_fourier_expansion([1, -1, -1, -1])
#print("NOT")
#coeffs = walsh_fourier_expansion([-1, 1])

print("Test")
coeffs = walsh_fourier_expansion([3, 1, 1, -1])

print("Test")
coeffs = walsh_fourier_expansion([-1, 0, 0, 1])

print("Test")
coeffs = walsh_fourier_expansion([0, 1, 1, 2])

print("Test")
coeffs = walsh_fourier_expansion([1, 0, 1, 0, 0, 0, 1, 0])
exit()

def walsh_time_transform(data, normalize=True):
    """
    Compute the Walsh–Hadamard transform along the time axis (axis=0)
    for a tensor shaped (T, P, X, Y).

    Parameters
    ----------
    data : np.ndarray
        Input tensor (T, P, X, Y) — typically float or int.
        T must be a power of 2 for standard Hadamard transform.
    normalize : bool
        If True, divide by sqrt(T) for orthonormal basis.

    Returns
    -------
    transformed : np.ndarray
        Tensor of same shape (T, P, X, Y), containing
        Walsh–Fourier coefficients for each pixel over time.
    """
    T, P, X, Y = data.shape
    if (T & (T - 1)) != 0:
        raise ValueError("Time dimension T must be a power of 2.")

    # Generate Hadamard matrix
    H = hadamard(T, dtype=float)
    if normalize:
        H /= np.sqrt(T)

    # Reshape for matrix multiplication
    flat = data.reshape(T, -1)          # (T, P*X*Y)
    transformed = H @ flat              # apply WHT along time axis
    return transformed.reshape(T, P, X, Y)


def pad_to_power_of_two(data):
    T, P, X, Y = data.shape
    next_pow2 = 1 << (T - 1).bit_length()  # smallest 2^n ≥ T
    if next_pow2 == T:
        return data  # already power of 2
    pad_len = next_pow2 - T
    pad = np.zeros((pad_len, P, X, Y), dtype=data.dtype)
    return np.concatenate([data, pad], axis=0)

train_dataset = tonic.datasets.NMNIST(
    save_to="../dataset/nmnist", train=True, transform=None
)

# Filter dataset to only include specified classes and limit samples
train_dataset = NMNIST_Dataset(
    train_dataset,
    allowed_classes=[0],
    samples_per_class=1,
    use_frames=True,
    frame_time_window=3000,
    frame_clip=True,
)

train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, collate_fn=collate_frames
)

np.random.seed(42)
sample, label = next(iter(train_loader))
sample = pad_to_power_of_two(sample[0])


print(np.argwhere(sample != 0))

wht_data = walsh_time_transform(sample, normalize=True)

p, x, y = 0, 10, 7
T = sample.shape[0]
time = np.arange(T)
signal = sample[:, p, x, y]
spectrum = wht_data[:, p, x, y]

# --- Plot ---
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.stem(time, signal)
plt.title("Pixel Signal Over Time")
plt.xlabel("t")
plt.ylabel("value")

plt.subplot(1,2,2)
plt.stem(time, spectrum)
plt.title("Walsh–Fourier Coefficients")
plt.xlabel("Walsh index (frequency)")
plt.ylabel("amplitude")

plt.tight_layout()
plt.show()