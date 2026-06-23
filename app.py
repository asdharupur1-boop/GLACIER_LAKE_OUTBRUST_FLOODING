# ================================================================
# app.py - Plotly-Free Streamlit Dashboard
# Uses Matplotlib + Streamlit Native Charts
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Water Level Monitoring Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# DEVELOPER DETAILS
# ================================================================

DEVELOPER_INFO = {
    "name": "Dr. [Your Name]",
    "role": "Lead AI/ML Engineer",
    "organization": "Water Resources Research Center",
    "email": "your.email@research.org",
    "github": "https://github.com/yourusername",
    "linkedin": "https://linkedin.com/in/yourprofile"
}

# ================================================================
# SESSION STATE
# ================================================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "None"
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None

# ================================================================
# TITLE
# ================================================================

st.title("🌊 IoT Water Level Monitoring & Prediction System")
st.markdown("---")

# ================================================================
# CHECK AND LOAD MODEL
# ================================================================

def load_model_files():
    """Load all model files if they exist."""
    try:
        if not os.path.exists('models/cnn_lstm_model.h5'):
            return None, None, None, None
        
        from tensorflow.keras.models import load_model
        
        model = load_model('models/cnn_lstm_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        scaler = joblib.load('models/scaler.pkl')
        
        with open('models/feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
        
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, feature_cols, metadata
        
    except Exception as e:
        st.warning(f"⚠️ Model loading issue: {str(e)}")
        return None, None, None, None

# Try to load model
model, scaler, feature_cols, metadata = load_model_files()

if model is not None:
    st.success("✅ AI Model Loaded Successfully!")
    SEQUENCE_LENGTH = metadata.get('sequence_length', 24)
else:
    st.info("ℹ️ Model not found. Running in data visualization mode only.")
    SEQUENCE_LENGTH = 24

DANGER_THRESHOLD = 25

# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.header("🎛️ Dashboard Controls")
    
    # Developer Section
    with st.expander("👨‍💻 Developer Details", expanded=False):
        st.markdown(f"**{DEVELOPER_INFO['name']}**")
        st.markdown(f"{DEVELOPER_INFO['role']}")
        st.markdown(f"{DEVELOPER_INFO['organization']}")
        st.markdown("---")
        st.markdown(f"**Email:** {DEVELOPER_INFO['email']}")
        st.markdown(f"**GitHub:** {DEVELOPER_INFO['github']}")
        st.markdown(f"**LinkedIn:** {DEVELOPER_INFO['linkedin']}")
        st.markdown("---")
        st.caption("📅 Version 2.0 | © 2026")
    
    st.markdown("---")
    
    # Data Source
    st.subheader("📂 Data Source")
    
    data_option = st.radio(
        "Select Data Source",
        ["📁 Local File", "📊 Sample Data"],
        index=1
    )
    
    if data_option == "📁 Local File":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
                st.session_state.df = df
                st.session_state.data_source = "Local"
                st.success(f"✅ Loaded {len(df):,} records")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:
        if st.button("🔄 Load Sample Data") or st.session_state.df is None:
            if os.path.exists('data/synthetic_water_data.csv'):
                df = pd.read_csv('data/synthetic_water_data.csv', parse_dates=['timestamp'])
                st.session_state.df = df
                st.session_state.data_source = "Sample"
                st.success(f"✅ Loaded {len(df):,} records")
            else:
                st.warning("⚠️ Sample data not found. Please run train_model.py first.")
    
    st.markdown("---")
    
    # Filter Controls
    if st.session_state.df is not None:
        df = st.session_state.df
        nodes = df['node_id'].unique().tolist()
        selected_node = st.selectbox("📍 Select Node", nodes, index=0)
        
        date_min, date_max = df['timestamp'].min(), df['timestamp'].max()
        date_range = st.date_input(
            "📅 Date Range",
            value=[date_min, date_max],
            min_value=date_min,
            max_value=date_max
        )
        
        time_period = st.selectbox(
            "⏱️ Time Period",
            ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "All"],
            index=4
        )
        
        DANGER_THRESHOLD = st.slider(
            "⚠️ Danger Threshold (m)",
            min_value=5,
            max_value=50,
            value=25,
            step=1
        )
        
        st.markdown("---")
        st.caption(f"📊 Data: {st.session_state.data_source}")
        st.caption(f"📈 Records: {len(df):,}")

# ================================================================
# MAIN CONTENT
# ================================================================

if st.session_state.df is None:
    st.info("👈 Load a dataset from the sidebar to begin.")
    
    # Show a preview of what the dashboard can do
    st.markdown("""
    ### 🚀 Features
    - **📊 Live Monitoring**: Track water levels in real-time
    - **🔮 AI Predictions**: CNN-LSTM model forecasts future levels
    - **🚨 Alert System**: Automatic danger notifications
    - **📈 Analytics**: Historical trends and distributions
    - **📡 Sensor Network**: Multi-node monitoring
    
    ### 📂 How to Get Started
    1. **Load Data**: Use the sidebar to upload your dataset
    2. **Select Node**: Choose a sensor node to monitor
    3. **View Insights**: Explore time series, distributions, and predictions
    """)
    st.stop()

df = st.session_state.df

# Filter data
node_df = df[df['node_id'] == selected_node].copy()

if len(date_range) == 2:
    mask = (node_df['timestamp'].dt.date >= date_range[0]) & (node_df['timestamp'].dt.date <= date_range[1])
    node_df = node_df.loc[mask]

period_map = {
    "1 Day": 24*4,
    "1 Week": 24*4*7,
    "1 Month": 24*4*30,
    "3 Months": 24*4*90,
    "6 Months": 24*4*180,
    "1 Year": 24*4*365,
    "All": len(node_df)
}
node_df = node_df.tail(period_map[time_period])

if len(node_df) == 0:
    st.warning("⚠️ No data available. Please adjust filters.")
    st.stop()

# ================================================================
# METRICS
# ================================================================

st.subheader(f"📊 Live Data - {selected_node}")

current = node_df['water_level'].iloc[-1]
avg = node_df['water_level'].mean()
max_level = node_df['water_level'].max()
min_level = node_df['water_level'].min()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    delta = f"{current - node_df['water_level'].iloc[-2]:+.2f}m" if len(node_df) > 1 else None
    st.metric("🌊 Current", f"{current:.2f}m", delta)

with col2:
    st.metric("📈 Avg", f"{avg:.2f}m")

with col3:
    st.metric("📊 Max", f"{max_level:.2f}m")

with col4:
    st.metric("📉 Min", f"{min_level:.2f}m")

with col5:
    status = "🔴 DANGER" if current > DANGER_THRESHOLD else "🟢 SAFE"
    st.metric("⚠️ Status", status)

# ================================================================
# TABS
# ================================================================

tab1, tab2, tab3 = st.tabs(["📉 Time Series", "📊 Distribution", "🔮 Predictions"])

# ================================================================
# TAB 1: Time Series (Using Matplotlib)
# ================================================================

with tab1:
    st.subheader("📈 Water Level Trend")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot water level
    ax.plot(node_df['timestamp'], node_df['water_level'], 
            color='blue', linewidth=2, label='Water Level')
    
    # Add danger threshold
    ax.axhline(y=DANGER_THRESHOLD, color='red', linestyle='--', 
               linewidth=2, label=f'Danger: {DANGER_THRESHOLD}m')
    
    # Add mean line
    ax.axhline(y=avg, color='green', linestyle=':', 
               linewidth=1.5, label=f'Mean: {avg:.2f}m')
    
    # Add flood events
    flood_data = node_df[node_df['flood_label'] == 1]
    if len(flood_data) > 0:
        ax.scatter(flood_data['timestamp'], flood_data['water_level'], 
                  color='red', s=80, marker='*', label='Flood Event', zorder=5)
    
    # Formatting
    ax.set_xlabel('Date/Time', fontsize=12)
    ax.set_ylabel('Water Level (m)', fontsize=12)
    ax.set_title(f'Water Level Monitoring - {selected_node}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
    plt.close()
    
    # Additional info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Trend", f"{node_df['water_level'].diff().mean():+.4f} m/step")
    with col2:
        st.metric("📊 Volatility", f"{node_df['water_level'].std():.3f} m")
    with col3:
        flood_count = node_df['flood_label'].sum()
        st.metric("⚠️ Flood Events", f"{int(flood_count)}")

# ================================================================
# TAB 2: Distribution (Using Matplotlib)
# ================================================================

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution - Selected Node")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(node_df['water_level'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(x=DANGER_THRESHOLD, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold: {DANGER_THRESHOLD}m')
        ax.axvline(x=avg, color='green', linestyle=':', 
                   linewidth=1.5, label=f'Mean: {avg:.2f}m')
        ax.set_xlabel('Water Level (m)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Water Level Distribution - {selected_node}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📊 Distribution Across Nodes")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Box plot using matplotlib
        nodes = df['node_id'].unique()
        data_by_node = [df[df['node_id'] == node]['water_level'].values for node in nodes]
        
        bp = ax.boxplot(data_by_node, labels=nodes, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for patch, color in zip(bp['boxes'], colors[:len(nodes)]):
            patch.set_facecolor(color)
        
        ax.axhline(y=DANGER_THRESHOLD, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold: {DANGER_THRESHOLD}m')
        ax.set_xlabel('Node', fontsize=12)
        ax.set_ylabel('Water Level (m)', fontsize=12)
        ax.set_title('Water Level Distribution Across All Nodes', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ================================================================
# TAB 3: Predictions
# ================================================================

with tab3:
    st.subheader("🔮 CNN-LSTM Model Predictions")
    
    if model is not None and len(node_df) >= SEQUENCE_LENGTH:
        try:
            # Prepare sequence
            recent = node_df.iloc[-SEQUENCE_LENGTH:][feature_cols].values
            sequence = recent.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
            
            # Predict
            pred_scaled = model.predict(sequence, verbose=0)
            
            # Inverse transform
            dummy = np.zeros((len(pred_scaled), len(feature_cols)))
            dummy[:, 0] = pred_scaled.flatten()
            pred_actual = scaler.inverse_transform(dummy)[0, 0]
            
            # Display predictions
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Current", f"{current:.2f} m")
            
            with col2:
                st.metric(
                    "🔮 Predicted",
                    f"{pred_actual:.2f} m",
                    f"{pred_actual - current:+.2f}m"
                )
            
            with col3:
                error_percent = abs(pred_actual - current) / (current + 0.01) * 100
                confidence = max(0, min(100, 95 - error_percent))
                st.metric("🎯 Confidence", f"{confidence:.1f}%")
            
            with col4:
                if pred_actual > DANGER_THRESHOLD:
                    st.error("🚨 DANGER")
                elif pred_actual > DANGER_THRESHOLD * 0.8:
                    st.warning("⚠️ WARNING")
                else:
                    st.success("✅ SAFE")
            
            # Prediction chart
            st.subheader("📈 Prediction Visualization")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Historical data (last 48 hours)
            hist = node_df.tail(48*4)
            ax.plot(hist['timestamp'], hist['water_level'], 
                    color='blue', linewidth=2, label='Historical')
            
            # Future prediction
            future_time = node_df['timestamp'].iloc[-1] + timedelta(minutes=15)
            ax.scatter([future_time], [pred_actual], 
                      color='red', s=100, marker='D', label='Predicted', zorder=5)
            
            # Add line to prediction
            ax.axvline(x=node_df['timestamp'].iloc[-1], color='gray', 
                      linestyle='--', alpha=0.5)
            
            # Add danger threshold
            ax.axhline(y=DANGER_THRESHOLD, color='red', linestyle='--', 
                       linewidth=2, label=f'Danger: {DANGER_THRESHOLD}m')
            
            # Formatting
            ax.set_xlabel('Date/Time', fontsize=12)
            ax.set_ylabel('Water Level (m)', fontsize=12)
            ax.set_title('Water Level Prediction', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Alert
            if pred_actual > DANGER_THRESHOLD:
                st.error("🚨 **DANGER ALERT!** Immediate action required!")
                st.markdown("""
                **Action Items:**
                - ⚠️ Evacuate low-lying areas
                - 📢 Alert local authorities
                - 📊 Monitor water level continuously
                - 🆘 Prepare emergency response teams
                """)
            elif pred_actual > DANGER_THRESHOLD * 0.8:
                st.warning("⚠️ **WARNING:** Water level approaching danger threshold. Monitor closely.")
            else:
                st.success("✅ **SAFE:** Water level within safe limits.")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check model integrity and try again.")
    else:
        if model is None:
            st.info("💡 Train the model first using train_model.py")
        else:
            st.info(f"Need at least {SEQUENCE_LENGTH} timesteps. Currently have {len(node_df)}.")
            st.progress(len(node_df) / SEQUENCE_LENGTH)

# ================================================================
# ALERTS
# ================================================================

st.markdown("---")
st.subheader("🚨 Recent Alerts")

alert_df = node_df.tail(100).copy()
alert_df['Alert'] = '🟢 SAFE'
alert_df.loc[alert_df['water_level'] > DANGER_THRESHOLD * 0.8, 'Alert'] = '🟡 WARNING'
alert_df.loc[alert_df['water_level'] > DANGER_THRESHOLD, 'Alert'] = '🔴 DANGER'
alert_df.loc[alert_df['flood_label'] == 1, 'Alert'] = '⚫ FLOOD'

alerts = alert_df[alert_df['Alert'] != '🟢 SAFE'].tail(10)

if len(alerts) > 0:
    st.dataframe(
        alerts[['timestamp', 'water_level', 'Alert']].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )
else:
    st.success("✅ No alerts. All systems normal.")

# ================================================================
# SENSOR HEALTH
# ================================================================

st.markdown("---")
st.subheader("📡 Sensor Network Health")

nodes = df['node_id'].unique().tolist()
cols = st.columns(len(nodes))

for i, node in enumerate(nodes):
    node_data = df[df['node_id'] == node]
    if len(node_data) > 0:
        last = node_data.iloc[-1]
        time_diff = (datetime.now() - last['timestamp']).total_seconds()
        if time_diff < 3600:
            status = "🟢 Online"
            color = "green"
        elif time_diff < 7200:
            status = "🟡 Stale"
            color = "orange"
        else:
            status = "🔴 Offline"
            color = "red"
        level = last['water_level']
    else:
        status = "🔴 Offline"
        color = "red"
        level = 0
    
    with cols[i]:
        st.markdown(f"""
        <div style="text-align:center; padding:10px; border-radius:10px; background-color: #f0f2f6;">
            <strong>{node}</strong><br>
            <span style="color:{color}; font-weight:bold;">{status}</span><br>
            <span style="font-size:14px;">Level: {level:.2f}m</span>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.caption(f"📊 Data: {st.session_state.data_source} | Records: {len(df):,}")

with col3:
    st.caption("🌊 IoT Water Level Monitoring | CNN-LSTM-RL")

st.markdown(f"""
<div style="text-align:center; padding:10px; border-top:1px solid #e0e0e0; margin-top:10px; font-size:12px; color:#666;">
    Developed by <strong>{DEVELOPER_INFO['name']}</strong> | 
    <a href="mailto:{DEVELOPER_INFO['email']}">📧 Contact</a> | 
    <a href="{DEVELOPER_INFO['github']}">🐙 GitHub</a> | 
    © 2026 All Rights Reserved
</div>
""", unsafe_allow_html=True)
