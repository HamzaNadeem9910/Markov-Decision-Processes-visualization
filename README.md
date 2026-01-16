# Interactive Grid-World MDP Visualization

This project is a web-based interactive application designed to visualize and analyze **Markov Decision Processes (MDPs)**. It implements two fundamental dynamic programming algorithms—**Value Iteration** and **Policy Iteration**—in a stochastic Grid-World environment.

The application was developed as part of the **Artificial Intelligence (CS-465)** course assignment to demonstrate how optimal policies emerge through iteration and how parameters like the discount factor ($\gamma$) affect an agent's behavior.

### Built With
* **Python 3.10+**
* **Streamlit** (Web Interface)
* **Plotly** (Interactive Grid Visualization)
* **NumPy** (Matrix Operations)

---

## 📸 Screenshots

### 1. Initial State (Before Running)
The agent has no knowledge of the environment. Values are 0.0, and the initial policy is arbitrary.

![Initial State](Screenshot (99).png)

### 2. Converged State (Optimal Policy)
After running iterations, the values propagate from the Goal (+10) and Trap (-10). The arrows show the optimal path, navigating around obstacles and avoiding the trap due to stochastic transitions.

![Converged State](screenshots/converged_state.png)

---

## ✨ Features

* **Grid-World Environment:** A 4x4 grid with a Start state, a fixed Goal state (+10 reward), a Trap state (-10 reward), and Obstacles (walls).
* **Stochastic Transitions:** Movement is not guaranteed. The agent has an 80% chance of moving as intended and a 20% chance of slipping sideways.
* **Algorithm Selection:** Toggle seamlessly between **Value Iteration** and **Policy Iteration** to compare their behaviors.
* **Interactive Controls:**
    * **Discount Factor Slider ($\gamma$):** Adjust $\gamma$ between 0.1 (myopic) and 0.99 (far-sighted) in real-time.
    * **Step-by-Step Execution:** Watch how state values update and policies evolve with every click.
* **Clear Visualizations:**
    * **Heatmap:** Color-coded grid showing the computed Value ($V(s)$) of each state.
    * **Policy Arrows:** Directional indicators showing the optimal action ($\pi(s)$) to take in each state.

---

## 🚀 Installation and Setup

Follow these steps to run the application locally.

### Prerequisites
Ensure you have Python installed.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/MDP-GridWorld-Viz.git](https://github.com/YOUR_USERNAME/MDP-GridWorld-Viz.git)
cd MDP-GridWorld-Viz
