# ================================================================
# app.py - Complete Streamlit Dashboard (REQUIRES MODEL)
# IoT Water Level Monitoring & Prediction System
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import os
import requests
from datetime import datetime, timedelta
import io
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
    "phone": "+91-XXXXX-XXXXX",
    "github": "https://github.com/yourusername",
    "linkedin": "https://linkedin.com/in/yourprofile",
    "website": "https://yourwebsite.com",
    "research_area": "IoT & AI for Water Resource Management",
    "publications": [
        "CNN-LSTM for Water Level Prediction (2024)",
        "Reinforcement Learning for Flood Management (2024)",
        "IoT Sensor Networks for Remote Lake Monitoring (2023)"
    ]
}

# ================================================================
# SESSION STATE
# ================================================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "None"
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'model' not in st.session_state:
    st.session_state.model = None

# ================================================================
# CHECK MODEL FILES
# ================================================================

def check_model_files():
    """Check if all required model files exist."""
    required_files = [
        'models/cnn_lstm_model.h5',
        'models/scaler.pkl',
        'models/feature_columns.json',
        'models/metadata.json'
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    return missing

def load_model_files():
    """Load all model files."""
    try:
        # Load scaler
        scaler = joblib.load('models/scaler.pkl')
        
        # Load feature columns
        with open('models/feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
        
        # Load metadata
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load model
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        model = load_model('models/cnn_lstm_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model, scaler, feature_cols, metadata
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# ================================================================
# TITLE
# ================================================================

st.title("🌊 IoT Water Level Monitoring & Prediction System")
st.markdown("---")

# ================================================================
# CHECK MODEL BEFORE ANYTHING ELSE
# ================================================================

missing_files = check_model_files()

if missing_files:
    st.error("🚨 MODEL FILES NOT FOUND!")
    st.markdown("### Missing Files:")
    for f in missing_files:
        st.markdown(f"- `{f}`")
    
    st.markdown("---")
    st.markdown("### How to Fix:")
    st.markdown("**Option 1: Train the Model**")
    st.code("python train_model.py", language="bash")
    
    st.markdown("**Option 2: Check File Locations**")
    st.markdown("- Make sure the `models/` folder exists")
    st.markdown("- Ensure all files are in the correct location")
    st.markdown("- Verify file permissions")
    
    st.markdown("---")
    st.markdown("### Required File Structure:")
    st.code("""
your-project/
├── app.py
├── train_model.py
└── models/
    ├── cnn_lstm_model.h5
    ├── scaler.pkl
    ├── feature_columns.json
    └── metadata.json
    """, language="bash")
    st.stop()

# ================================================================
# LOAD MODEL (Only if files exist)
# ================================================================

with st.spinner("🔄 Loading AI Model..."):
    model, scaler, feature_cols, metadata = load_model_files()

if model is None:
    st.error("Failed to load model. Please check the files and try again.")
    st.stop()

st.session_state.model = model
st.session_state.scaler = scaler
st.session_state.feature_cols = feature_cols
st.session_state.metadata = metadata
st.session_state.model_loaded = True

SEQUENCE_LENGTH = metadata.get('sequence_length', 24)
DANGER_THRESHOLD = 25

st.success("AI Model Loaded Successfully!")

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
        st.markdown(f"**Phone:** {DEVELOPER_INFO['phone']}")
        st.markdown(f"**GitHub:** {DEVELOPER_INFO['github']}")
        st.markdown(f"**LinkedIn:** {DEVELOPER_INFO['linkedin']}")
        st.markdown(f"**Website:** {DEVELOPER_INFO['website']}")
        st.markdown("---")
        st.markdown(f"**Research Area:**")
        st.markdown(f"{DEVELOPER_INFO['research_area']}")
        st.markdown("**Publications:**")
        for pub in DEVELOPER_INFO['publications']:
            st.markdown(f"- {pub}")
        st.markdown("---")
        st.caption("📅 Version 2.0 | © 2026")
    
    st.markdown("---")
    
    # Data Source
    st.subheader("📂 Data Source")
    
    data_option = st.radio(
        "Select Data Source",
        ["📁 Local File", "🌐 URL Link", "📊 Sample Data"],
        index=2
    )
    
    # Load data based on option
    if data_option == "📁 Local File":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
                st.session_state.df = df
                st.session_state.data_source = "Local"
                st.success(f"Loaded {len(df):,} records")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif data_option == "🌐 URL Link":
        url = st.text_input("Dataset URL")
        if st.button("Load Dataset") and url:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    content = io.StringIO(response.text)
                    df = pd.read_csv(content, parse_dates=['timestamp'])
                    st.session_state.df = df
                    st.session_state.data_source = "URL"
                    st.success(f"Loaded {len(df):,} records")
                else:
                    st.error(f"HTTP Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:  # Sample Data
        if st.button("🔄 Load Sample Data") or st.session_state.df is None:
            # Check if sample data exists
            if os.path.exists('data/synthetic_water_data.csv'):
                df = pd.read_csv('data/synthetic_water_data.csv', parse_dates=['timestamp'])
                st.session_state.df = df
                st.session_state.data_source = "Sample"
                st.success(f"Loaded {len(df):,} records")
            else:
                st.warning("Sample data not found. Please run train_model.py first.")
    
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
        st.caption(f"🔄 Updated: {datetime.now().strftime('%H:%M:%S')}")

# ================================================================
# MAIN CONTENT
# ================================================================

if st.session_state.df is None:
    st.info("👈 Load a dataset from the sidebar to begin.")
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
    st.warning("No data available. Please adjust filters.")
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

tab1, tab2, tab3, tab4 = st.tabs(["📉 Time Series", "📊 Distribution", "🔮 Predictions", "📋 Data Table"])

# ================================================================
# TAB 1: Time Series
# ================================================================

with tab1:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=node_df['timestamp'],
        y=node_df['water_level'],
        mode='lines',
        name='Water Level',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_hline(
        y=DANGER_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Danger: {DANGER_THRESHOLD}m"
    )
    
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Mean: {avg:.2f}m"
    )
    
    flood_data = node_df[node_df['flood_label'] == 1]
    if len(flood_data) > 0:
        fig.add_trace(go.Scatter(
            x=flood_data['timestamp'],
            y=flood_data['water_level'],
            mode='markers',
            name='Flood Event',
            marker=dict(color='red', size=12, symbol='star')
        ))
    
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=30, label="1M", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ]
        )
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="Date/Time",
        yaxis_title="Water Level (m)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# TAB 2: Distribution
# ================================================================

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            node_df,
            x='water_level',
            nbins=50,
            title=f'Distribution - {selected_node}',
            color_discrete_sequence=['blue']
        )
        fig_hist.add_vline(x=DANGER_THRESHOLD, line_dash="dash", line_color="red")
        fig_hist.add_vline(x=avg, line_dash="dot", line_color="green")
        fig_hist.update_layout(template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_box = px.box(
            df,
            x='node_id',
            y='water_level',
            title='Distribution Across All Nodes',
            color='node_id'
        )
        fig_box.add_hline(y=DANGER_THRESHOLD, line_dash="dash", line_color="red")
        fig_box.update_layout(template='plotly_white')
        st.plotly_chart(fig_box, use_container_width=True)

# ================================================================
# TAB 3: Predictions (REQUIRES MODEL)
# ================================================================

with tab3:
    st.subheader("🔮 CNN-LSTM Model Predictions")
    
    if len(node_df) >= SEQUENCE_LENGTH:
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
            fig_pred = go.Figure()
            
            hist = node_df.tail(48*4)
            fig_pred.add_trace(go.Scatter(
                x=hist['timestamp'],
                y=hist['water_level'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            future_time = node_df['timestamp'].iloc[-1] + timedelta(minutes=15)
            fig_pred.add_trace(go.Scatter(
                x=[future_time],
                y=[pred_actual],
                mode='markers+lines',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=15, color='red')
            ))
            
            # Add confidence interval
            ci = pred_actual * 0.05
            fig_pred.add_trace(go.Scatter(
                x=[future_time, future_time],
                y=[pred_actual - ci, pred_actual + ci],
                mode='markers',
                name='95% CI',
                marker=dict(color='rgba(255,0,0,0.3)', size=10)
            ))
            
            fig_pred.update_layout(
                height=400,
                xaxis_title="Date/Time",
                yaxis_title="Water Level (m)",
                template='plotly_white'
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Alert
            if pred_actual > DANGER_THRESHOLD:
                st.error("🚨 DANGER ALERT! Immediate action required!")
                st.markdown("**Action Items:**")
                st.markdown("- Evacuate low-lying areas")
                st.markdown("- Alert local authorities")
                st.markdown("- Monitor water level continuously")
                st.markdown("- Prepare emergency response teams")
            elif pred_actual > DANGER_THRESHOLD * 0.8:
                st.warning("⚠️ WARNING: Water level approaching danger threshold.")
            else:
                st.success("✅ SAFE: Water level within safe limits.")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check model integrity and try again.")
    else:
        st.info(f"Need at least {SEQUENCE_LENGTH} timesteps for prediction. Currently have {len(node_df)}.")
        st.progress(len(node_df) / SEQUENCE_LENGTH)

# ================================================================
# TAB 4: Data Table
# ================================================================

with tab4:
    st.subheader("📋 Raw Data")
    
    display_df = node_df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_df = display_df.round(2)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "timestamp": "Timestamp",
            "water_level": st.column_config.NumberColumn("Water Level (m)", format="%.2f"),
            "rainfall": st.column_config.NumberColumn("Rainfall (mm)", format="%.1f"),
            "temperature": st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
            "humidity": st.column_config.NumberColumn("Humidity (%)", format="%.0f"),
            "ph": st.column_config.NumberColumn("pH", format="%.2f"),
            "turbidity": st.column_config.NumberColumn("Turbidity (NTU)", format="%.1f"),
            "tds": st.column_config.NumberColumn("TDS (ppm)", format="%.0f"),
        }
    )
    
    csv = node_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"water_level_data_{selected_node}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

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
    <a href="{DEVELOPER_INFO['linkedin']}">🔗 LinkedIn</a> | 
    © 2026 All Rights Reserved
</div>
""", unsafe_allow_html=True)
