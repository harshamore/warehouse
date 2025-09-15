import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Warehouse User Productivity Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .shift-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        color: white;
        font-weight: bold;
        margin: 0.25rem;
        display: inline-block;
    }
    .morning { background-color: #ff9500; }
    .afternoon { background-color: #007bff; }
    .night { background-color: #6f42c1; }
    .overtime { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

def load_and_process_data(uploaded_file):
    """Load and process the Excel file"""
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file, sheet_name=0)
        
        # Convert timestamp columns to datetime
        timestamp_columns = ['Requested', 'Confirmed', 'Completed']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Filter out rows with missing User or critical timestamp data
        df = df.dropna(subset=['User', 'Confirmed', 'Completed'])
        
        # Remove system users and invalid entries
        df = df[~df['User'].isin(['', 'SYSTEM', 'AUTO', 'LINPROD'])]
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def determine_shift(confirmed_time):
    """Determine shift based on confirmed timestamp"""
    if pd.isna(confirmed_time):
        return "Unknown"
    
    hour = confirmed_time.hour
    
    if 6 <= hour < 14:
        return "Morning"
    elif 14 <= hour < 22:
        return "Afternoon"
    else:  # 22:00 - 06:00
        return "Night"

def calculate_user_metrics(df, selected_user):
    """Calculate comprehensive metrics for selected user"""
    user_data = df[df['User'] == selected_user].copy()
    
    if user_data.empty:
        return None
    
    # Calculate active time (completion - confirmed) in hours
    user_data['active_time_hours'] = (
        user_data['Completed'] - user_data['Confirmed']
    ).dt.total_seconds() / 3600
    
    # Remove negative or unrealistic times (more than 24 hours)
    user_data = user_data[
        (user_data['active_time_hours'] > 0) & 
        (user_data['active_time_hours'] <= 24)
    ]
    
    if user_data.empty:
        return None
    
    # Add shift information
    user_data['shift'] = user_data['Confirmed'].apply(determine_shift)
    
    # Calculate metrics
    total_active_hours = user_data['active_time_hours'].sum()
    total_tasks = len(user_data)
    tasks_per_hour = total_tasks / total_active_hours if total_active_hours > 0 else 0
    unique_orders = user_data['Order No'].nunique()
    
    # Shift analysis
    shift_counts = user_data['shift'].value_counts()
    
    # Check for overtime (working in multiple shifts on same day)
    user_data['date'] = user_data['Confirmed'].dt.date
    daily_shifts = user_data.groupby('date')['shift'].nunique()
    overtime_days = (daily_shifts > 1).sum()
    
    return {
        'user_data': user_data,
        'total_active_hours': total_active_hours,
        'total_tasks': total_tasks,
        'tasks_per_hour': tasks_per_hour,
        'unique_orders': unique_orders,
        'shift_counts': shift_counts,
        'overtime_days': overtime_days
    }

def create_productivity_charts(user_metrics):
    """Create productivity visualization charts"""
    user_data = user_metrics['user_data']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Tasks Completed by Hour of Day',
            'Shift Distribution',
            'Daily Task Count',
            'Task Completion Time Distribution'
        ),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Chart 1: Tasks by hour of day
    user_data['hour'] = user_data['Confirmed'].dt.hour
    hourly_tasks = user_data['hour'].value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(x=hourly_tasks.index, y=hourly_tasks.values, name="Tasks"),
        row=1, col=1
    )
    
    # Chart 2: Shift distribution
    shift_counts = user_metrics['shift_counts']
    fig.add_trace(
        go.Pie(labels=shift_counts.index, values=shift_counts.values, name="Shifts"),
        row=1, col=2
    )
    
    # Chart 3: Daily task count
    user_data['date'] = user_data['Confirmed'].dt.date
    daily_tasks = user_data['date'].value_counts().sort_index()
    
    fig.add_trace(
        go.Scatter(x=daily_tasks.index, y=daily_tasks.values, mode='lines+markers', name="Daily Tasks"),
        row=2, col=1
    )
    
    # Chart 4: Task completion time distribution
    fig.add_trace(
        go.Histogram(x=user_data['active_time_hours'], name="Completion Time", nbinsx=20),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text=f"Productivity Analytics")
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Warehouse User Productivity Analytics</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload your warehouse operational data Excel file"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("🔄 Processing data..."):
            df = load_and_process_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Data loaded successfully! Total records: {len(df):,}")
            
            # Data overview
            with st.expander("📋 Data Overview", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", f"{len(df):,}")
                with col2:
                    st.metric("Unique Users", f"{df['User'].nunique():,}")
                with col3:
                    st.metric("Date Range", f"{df['Confirmed'].dt.date.min()} to {df['Confirmed'].dt.date.max()}")
                with col4:
                    st.metric("Unique Tasks", f"{df['Task'].nunique():,}")
            
            # User selection
            st.header("👤 User Selection")
            
            # Get unique users
            users = sorted(df['User'].unique())
            
            # User selection dropdown
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_user = st.selectbox(
                    "Select a user to analyze:",
                    options=users,
                    index=0
                )
            
            with col2:
                st.metric("Total Users", len(users))
            
            # User analytics
            if selected_user:
                st.header(f"📈 Analytics for User: {selected_user}")
                
                # Calculate metrics
                with st.spinner("🔄 Calculating user metrics..."):
                    user_metrics = calculate_user_metrics(df, selected_user)
                
                if user_metrics:
                    # Key metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<h3>⏰ Total Active Hours</h3>'
                            f'<h2>{user_metrics["total_active_hours"]:.1f}</h2>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<h3>✅ Tasks Completed</h3>'
                            f'<h2>{user_metrics["total_tasks"]:,}</h2>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<h3>⚡ Tasks/Hour</h3>'
                            f'<h2>{user_metrics["tasks_per_hour"]:.1f}</h2>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col4:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<h3>📦 Unique Orders</h3>'
                            f'<h2>{user_metrics["unique_orders"]:,}</h2>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Shift analysis
                    st.subheader("🕐 Shift Analysis")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        shift_counts = user_metrics['shift_counts']
                        shift_badges = ""
                        for shift, count in shift_counts.items():
                            badge_class = shift.lower()
                            shift_badges += f'<span class="shift-badge {badge_class}">{shift}: {count} tasks</span>'
                        
                        st.markdown(f"**Primary Shifts:** {shift_badges}", unsafe_allow_html=True)
                        
                        if user_metrics['overtime_days'] > 0:
                            st.markdown(
                                f'<span class="shift-badge overtime">Overtime: {user_metrics["overtime_days"]} days</span>',
                                unsafe_allow_html=True
                            )
                    
                    with col2:
                        # Shift distribution pie chart
                        fig_pie = px.pie(
                            values=shift_counts.values,
                            names=shift_counts.index,
                            title="Shift Distribution"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Productivity charts
                    st.subheader("📊 Productivity Charts")
                    productivity_fig = create_productivity_charts(user_metrics)
                    st.plotly_chart(productivity_fig, use_container_width=True)
                    
                    # Orders managed
                    st.subheader("📋 Orders Managed")
                    user_data = user_metrics['user_data']
                    orders_df = user_data[['Order No', 'Task', 'Confirmed', 'Completed', 'shift']].copy()
                    orders_df['Duration (hours)'] = user_data['active_time_hours'].round(2)
                    orders_df = orders_df.sort_values('Confirmed', ascending=False)
                    
                    # Filter options for orders
                    col1, col2 = st.columns(2)
                    with col1:
                        shift_filter = st.multiselect(
                            "Filter by Shift:",
                            options=orders_df['shift'].unique(),
                            default=orders_df['shift'].unique()
                        )
                    
                    with col2:
                        task_filter = st.multiselect(
                            "Filter by Task Type:",
                            options=orders_df['Task'].unique(),
                            default=orders_df['Task'].unique()
                        )
                    
                    # Apply filters
                    filtered_orders = orders_df[
                        (orders_df['shift'].isin(shift_filter)) &
                        (orders_df['Task'].isin(task_filter))
                    ]
                    
                    st.dataframe(
                        filtered_orders,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download option
                    st.subheader("💾 Export Data")
                    csv = filtered_orders.to_csv(index=False)
                    st.download_button(
                        label="📥 Download User Data as CSV",
                        data=csv,
                        file_name=f"{selected_user}_productivity_data.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("⚠️ No valid data found for the selected user.")
        
        else:
            st.error("❌ Failed to load the Excel file. Please check the file format and try again.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👈 Please upload an Excel file using the sidebar to begin analysis.")
        
        st.markdown("""
        ### 📖 How to Use:
        1. **Upload File**: Use the sidebar to upload your warehouse operational Excel file
        2. **Select User**: Choose a user from the dropdown to analyze their productivity
        3. **View Analytics**: Explore comprehensive metrics including:
           - Total active hours and tasks completed
           - Tasks per hour productivity rate
           - Shift patterns and overtime analysis
           - Orders managed with filtering options
        
        ### 📊 Features:
        - **Real-time Calculations**: Automatic computation of productivity metrics
        - **Shift Detection**: Automatic categorization into Morning/Afternoon/Night shifts
        - **Overtime Tracking**: Identifies when users work across multiple shifts
        - **Interactive Charts**: Visual representation of productivity patterns
        - **Data Export**: Download filtered results as CSV
        
        ### 🕐 Shift Definitions:
        - **Morning Shift**: 06:00 - 14:00
        - **Afternoon Shift**: 14:00 - 22:00  
        - **Night Shift**: 22:00 - 06:00
        - **Overtime**: Working in multiple shifts on the same day
        """)

if __name__ == "__main__":
    main()
