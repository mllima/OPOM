"""
Test script for OPOM and TransferFunction models.

Compares step responses from original transfer functions and OPOM state-space representation.
"""

from opom import OPOM, TransferFunction
from scipy import signal
import matplotlib.pyplot as plt
from typing import List

def create_transfer_functions() -> List[List[TransferFunction]]:
    """Create and return a 2x2 MIMO system of transfer functions."""
    h11 = TransferFunction([0.2**2], [1, 2*0.1*0.2, 0.2**2], delay=10)
    h12 = TransferFunction([1.5], [23*62, 23+62, 1])
    h21 = TransferFunction([-1.4], [30*90, 30+90, 1], delay=0)
    h22 = TransferFunction([2.8], [90, 1])
    return [[h11, h12], [h21, h22]]

def plot_responses(opom_sys, transfer_functions, Ts: float, n_points: int = 3000):
    """Plot and compare responses from OPOM and original transfer functions."""
    system_ss = signal.StateSpace(opom_sys.A, opom_sys.B, opom_sys.C, opom_sys.D, dt=Ts)
    t, y = signal.dimpulse(system_ss, n=n_points)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    titles = [["y11", "y12"], ["y21", "y22"]]
    
    for i in range(2):
        for j in range(2):
            tf = transfer_functions[i][j]
            T, y_tf = signal.step(tf.tf)
            T += tf.delay
            axs[i, j].set_title(titles[i][j])
            axs[i, j].plot(T, y_tf, label="Original")
            axs[i, j].step(t, y[i][:, j], where='post', label="OPOM")
            axs[i, j].legend()
            axs[i, j].set_xlabel("Time (s)")
            axs[i, j].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def main():
    Ts = 0.1
    transfer_functions = create_transfer_functions()
    opom_sys = OPOM(transfer_functions, Ts)
    plot_responses(opom_sys, transfer_functions, Ts)

if __name__ == "__main__":
    main()
