import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = '/home/aaditya/Downloads/tmp'

def plot_forces():
    path = os.path.join(DATA_DIR, 'forces_history.npy')
    if not os.path.exists(path):
        print("No forces history found.")
        return
    
    forces = np.load(path) # (Steps, 3)
    
    steps = np.arange(len(forces))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, forces[:, 0], label='Drag (X)', color='red')
    plt.plot(steps, forces[:, 1], label='Lift (Y)', color='blue')
    
    plt.title('Aerodynamic Forces on NACA Airfoil (10deg AoA)')
    plt.xlabel('Simulation Step')
    plt.ylabel('Integrated Pressure Force (Relative)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(DATA_DIR, 'forces_plot.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_forces()
