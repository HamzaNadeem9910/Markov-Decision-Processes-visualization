import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

# --- CONFIGURATION ---
GRID_ROWS = 4
GRID_COLS = 4
START_STATE = (0, 0)
GOAL_STATE = (0, 3)  # Top Right
TRAP_STATE = (1, 3)  # Below Goal
OBSTACLES = [(1, 1)] # Wall in the middle

# Rewards
R_GOAL = 10
R_TRAP = -10
R_STEP = -0.1

actions = ['Up', 'Down', 'Left', 'Right']

# --- HELPER FUNCTIONS ---

def is_terminal(state):
    return state == GOAL_STATE or state == TRAP_STATE

def get_next_state(state, action):
    """
    Returns the next state given an action deterministically.
    Handles boundaries and obstacles (walls).
    """
    r, c = state
    if action == 'Up':
        nr, nc = max(0, r - 1), c
    elif action == 'Down':
        nr, nc = min(GRID_ROWS - 1, r + 1), c
    elif action == 'Left':
        nr, nc = r, max(0, c - 1)
    elif action == 'Right':
        nr, nc = r, min(GRID_COLS - 1, c + 1)
    else:
        nr, nc = r, c
    
    # If hit an obstacle, stay in same place
    if (nr, nc) in OBSTACLES:
        return state
    
    return (nr, nc)

def get_transition_probs(state, action):
    """
    Returns a list of (probability, next_state) tuples.
    Model: 80% success, 10% deviates left, 10% deviates right.
    """
    transitions = []
    
    if is_terminal(state):
        return [(1.0, state)]
    
    # Map actions to their perpendicular deviations
    # (Action, Left_Dev, Right_Dev)
    deviation_map = {
        'Up': ('Left', 'Right'),
        'Down': ('Right', 'Left'),
        'Left': ('Down', 'Up'),
        'Right': ('Up', 'Down')
    }
    
    # Intended move (0.8)
    transitions.append((0.8, get_next_state(state, action)))
    
    # Deviations (0.1 each)
    left_dev, right_dev = deviation_map[action]
    transitions.append((0.1, get_next_state(state, left_dev)))
    transitions.append((0.1, get_next_state(state, right_dev)))
    
    return transitions

# --- ALGORITHMS ---

def run_value_iteration_step(V, gamma):
    """Performs one sweep of Value Iteration"""
    new_V = np.zeros((GRID_ROWS, GRID_COLS))
    delta = 0
    
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            state = (r, c)
            
            if is_terminal(state):
                new_V[r, c] = R_GOAL if state == GOAL_STATE else R_TRAP
                continue
            if state in OBSTACLES:
                new_V[r, c] = 0 # Walls have 0 value
                continue
            
            # Bellman Optimality Equation
            q_values = []
            for a in actions:
                q_val = 0
                for prob, next_s in get_transition_probs(state, a):
                    reward = R_STEP
                    # If next state is terminal, add that reward effectively
                    if next_s == GOAL_STATE: reward += R_GOAL
                    elif next_s == TRAP_STATE: reward += R_TRAP
                    
                    # Standard Bellman: R + gamma * V(s')
                    # Note: We incorporate immediate reward differently depending on formulation
                    # Here: R(s,a,s') + gamma * V(s')
                    
                    nr, nc = next_s
                    q_val += prob * (R_STEP + gamma * V[nr, nc])
                q_values.append(q_val)
            
            new_V[r, c] = max(q_values)
            delta = max(delta, abs(new_V[r, c] - V[r, c]))
            
    return new_V, delta

def extract_policy(V, gamma):
    """Derives optimal policy from Value Function"""
    policy = np.empty((GRID_ROWS, GRID_COLS), dtype=object)
    
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            state = (r, c)
            if is_terminal(state) or state in OBSTACLES:
                policy[r, c] = ''
                continue
                
            best_action = None
            max_q = -float('inf')
            
            for a in actions:
                q_val = 0
                for prob, next_s in get_transition_probs(state, a):
                    nr, nc = next_s
                    q_val += prob * (R_STEP + gamma * V[nr, nc])
                
                if q_val > max_q:
                    max_q = q_val
                    best_action = a
            policy[r, c] = best_action
            
    return policy

def run_policy_evaluation(policy, V, gamma):
    """Evaluates a specific policy"""
    # Run a few sweeps to approximate V for this policy
    # For visualization, we do 5 sweeps per 'step' button click usually, 
    # but here we do one full sweep
    new_V = np.copy(V)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            state = (r, c)
            if is_terminal(state):
                new_V[r, c] = R_GOAL if state == GOAL_STATE else R_TRAP
                continue
            if state in OBSTACLES: continue
            
            action = policy[r, c]
            val = 0
            for prob, next_s in get_transition_probs(state, action):
                nr, nc = next_s
                val += prob * (R_STEP + gamma * V[nr, nc])
            new_V[r, c] = val
    return new_V

def run_policy_improvement(V, gamma):
    """Improves policy based on current V"""
    return extract_policy(V, gamma)

# --- VISUALIZATION ---

def plot_grid(V, policy):
    # Flip V for plotting so (0,0) is top-left
    # Plotly heatmaps usually start bottom-left, we need to adjust logic
    
    # Prepare Annotation Text (Values rounded)
    z_text = np.round(V, 2).astype(str)
    
    # Prepare Arrows
    arrow_map = {'Up': '↑', 'Down': '↓', 'Left': '←', 'Right': '→', '': ''}
    
    annotations = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val_str = str(round(V[r, c], 2))
            pol_str = arrow_map.get(policy[r, c], '')
            
            if (r, c) == GOAL_STATE: label = "GOAL<br>" + val_str
            elif (r, c) == TRAP_STATE: label = "TRAP<br>" + val_str
            elif (r, c) in OBSTACLES: label = "WALL"
            else: label = f"{val_str}<br><b>{pol_str}</b>"
            
            annotations.append(dict(
                x=c, y=GRID_ROWS-1-r, # Adjust Y for plotting
                text=label,
                showarrow=False,
                font=dict(color='black' if (r,c) not in OBSTACLES else 'white')
            ))

    # Heatmap Data
    # Invert rows for visual consistency with matrix indices
    heatmap_data = np.flipud(V) 
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=list(range(GRID_COLS)),
        y=list(range(GRID_ROWS)),
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="Value Function & Optimal Policy",
        annotations=annotations,
        xaxis=dict(side='top', tickmode='linear'),
        yaxis=dict(tickmode='linear', showticklabels=False),
        width=600, height=600,
        autosize=False
    )
    return fig

# --- STREAMLIT APP LOGIC ---

st.title("MDP Visualization: Value vs Policy Iteration")
st.markdown("Implemented for Course CS-465 | **Zero Plagiarism Implementation**")

# Sidebar Controls
st.sidebar.header("Configuration")
algo_type = st.sidebar.radio("Select Algorithm", ["Value Iteration", "Policy Iteration"])
gamma = st.sidebar.slider("Discount Factor (gamma)", 0.1, 0.99, 0.9, 0.01)

# Initialize Session State
if 'V' not in st.session_state:
    st.session_state.V = np.zeros((GRID_ROWS, GRID_COLS))
if 'policy' not in st.session_state:
    # Initialize random policy
    st.session_state.policy = np.full((GRID_ROWS, GRID_COLS), 'Up', dtype=object)
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0

# Reset Button
if st.sidebar.button("Reset / Clear"):
    st.session_state.V = np.zeros((GRID_ROWS, GRID_COLS))
    st.session_state.policy = np.full((GRID_ROWS, GRID_COLS), 'Up', dtype=object)
    st.session_state.iteration = 0

# Step Button Logic
if st.sidebar.button("Run 1 Iteration"):
    st.session_state.iteration += 1
    
    if algo_type == "Value Iteration":
        # VI: Update Values then derive policy
        st.session_state.V, _ = run_value_iteration_step(st.session_state.V, gamma)
        st.session_state.policy = extract_policy(st.session_state.V, gamma)
        
    else: # Policy Iteration
        # PI: Evaluate current policy, then improve it
        # We do simpler synchronous PI here for visualization step-by-step
        st.session_state.V = run_policy_evaluation(st.session_state.policy, st.session_state.V, gamma)
        st.session_state.policy = run_policy_improvement(st.session_state.V, gamma)

# Display Metrics
st.markdown(f"### Iteration: {st.session_state.iteration}")
col1, col2 = st.columns(2)
with col1:
    st.metric("Discount Factor", gamma)
with col2:
    st.metric("Algorithm", algo_type)

# Plot
fig = plot_grid(st.session_state.V, st.session_state.policy)
st.plotly_chart(fig)

# Explanation
st.info("""
**Legend:**
- **Numbers:** State Value (V)
- **Arrows:** Optimal Action (Policy)
- **WALL:** Obstacle
- **GOAL:** +10 Reward
- **TRAP:** -10 Reward
""")