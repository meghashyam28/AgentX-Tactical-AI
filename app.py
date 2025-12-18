import streamlit as st
import numpy as np
import plotly.graph_objects as go
# Change your imports to this:
from src.agent import LinUCBAgent
from env.env import DriftingEnvironment
import time

st.set_page_config(page_title="AgentX: Autonomous AI", layout="wide")
st.title("🏀 AgentX: Tactical Intelligence Engine")

# --- Fully Autonomous Setup ---
if 'agent' not in st.session_state:
    st.session_state.agent = LinUCBAgent(n_actions=5, context_dim=6, alpha=1.5)
    st.session_state.env = DriftingEnvironment(n_actions=5, context_dim=6)
    st.session_state.rewards = []
    st.session_state.plays_made = []

# Sidebar for Demo Controls
with st.sidebar:
    st.header("🎮 Session Control")
    if st.button("🚀 RUN GAME SIMULATION"):
        # Run a batch of 50 possessions automatically
        for _ in range(50):
            # 1. System pulls automated situation (No sliders!)
            ctx = st.session_state.env.get_context()
            # 2. AI decides
            action = st.session_state.agent.select_action(ctx)
            # 3. Environment rewards based on internal logic
            reward, _, _ = st.session_state.env.get_reward(action, ctx)
            # 4. Agent learns
            st.session_state.agent.update(action, ctx, reward)
            
            st.session_state.rewards.append(reward)
            plays = ["Pick & Roll", "Fast Break", "Iso-Play", "Zone Defense", "Full-Court Press"]
            st.session_state.plays_made.append(plays[action])

# --- Winning Visuals ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Autonomous Learning Curve")
    if len(st.session_state.rewards) > 5:
        # Show the AI getting smarter over time
        acc = np.convolve(st.session_state.rewards, np.ones(10)/10, mode='valid')
        fig = go.Figure(go.Scatter(y=acc, line=dict(color='#00FFAA', width=3)))
        fig.update_layout(title="Tactical Success Rate", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Tactical Adaptability")
    if st.session_state.plays_made:
        # This shows the judges the AI is picking DIFFERENT plays
        fig2 = go.Figure(go.Histogram(x=st.session_state.plays_made, marker_color='#FF4B4B'))
        fig2.update_layout(title="Play Distribution", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)