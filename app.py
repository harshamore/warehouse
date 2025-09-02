import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, time
import seaborn as sns
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Warehouse Queue Optimizer with Shift Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .optimization-card {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .shift-morning {
        background-color: #fff8dc;
        border-left: 5px solid #ffd700;
    }
    .shift-afternoon {
        background-color: #f0f8ff;
        border-left: 5px solid #4169e1;
    }
    .shift-night {
        background-color: #2f2f4f;
        color: white;
        border-left: 5px solid #9370db;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏭 Warehouse Queue Optimizer with Shift Analysis")
st.markdown("**Analyze operator productivity across shifts and optimize queue performance for maximum efficiency**")

# Sidebar for file upload and controls
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File", 
    type=['xlsx', 'xls'],
    help="Upload your warehouse operations Excel file with timestamp data"
)

class ShiftAwareWarehouseAnalyzer:
    def __init__(self, df):
        self.df = df
        self.queue_analysis = {}
        self.shift_analysis = {}
        self.operator_activity = {}
        self.optimization_results = {}
        
        # Define shift boundaries
        self.shift_definitions = {
            'Morning': {'start': time(6, 0), 'end': time(14, 0), 'label': '06:00-14:00'},
            'Afternoon': {'start': time(14, 0), 'end': time(22, 0), 'label': '14:00-22:00'}, 
            'Night': {'start': time(22, 0), 'end': time(6, 0), 'label': '22:00-06:00'}
        }
        
    def clean_data(self):
        """Clean and prepare data for analysis"""
        # Remove rows with missing essential data
        self.df = self.df.dropna(subset=['User', 'Queue'])
        
        # Convert numeric columns
        numeric_columns = ['Quantity']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        return self.df
    
    def detect_timestamp_columns(self):
        """Detect potential timestamp columns in the dataset"""
        timestamp_cols = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'requested', 'confirmed', 'completed']):
                # Try to convert to datetime
                try:
                    pd.to_datetime(self.df[col].dropna().head(10))
                    timestamp_cols.append(col)
                except:
                    continue
        return timestamp_cols
    
    def determine_shift(self, timestamp):
        """Determine which shift a timestamp belongs to"""
        if pd.isna(timestamp):
            return 'Unknown'
        
        try:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            time_only = timestamp.time()
            
            # Morning: 06:00 - 14:00
            if time(6, 0) <= time_only < time(14, 0):
                return 'Morning'
            # Afternoon: 14:00 - 22:00
            elif time(14, 0) <= time_only < time(22, 0):
                return 'Afternoon'
            # Night: 22:00 - 06:00 (crosses midnight)
            else:
                return 'Night'
        except:
            return 'Unknown'
    
    def add_shift_information(self, timestamp_column):
        """Add shift information to the dataset based on task request time"""
        if timestamp_column in self.df.columns:
            # Convert to datetime
            self.df['timestamp'] = pd.to_datetime(self.df[timestamp_column])
            # Determine shift based on when task was requested
            self.df['shift'] = self.df['timestamp'].apply(self.determine_shift)
            return True
        return False
    
    def filter_data_by_shift(self, shift_name, timestamp_column):
        """Filter dataset to show only tasks requested during specific shift"""
        if 'shift' not in self.df.columns:
            return pd.DataFrame()
        
        return self.df[self.df['shift'] == shift_name].copy()
    
    def analyze_operators_per_shift(self):
        """Analyze operators across different shifts"""
        if 'shift' not in self.df.columns:
            return None
        
        shift_analysis = {}
        for shift in ['Morning', 'Afternoon', 'Night']:
            shift_data = self.df[self.df['shift'] == shift]
            
            shift_analysis[shift] = {
                'total_operators': shift_data['User'].nunique(),
                'total_tasks': len(shift_data),
                'queues': {},
                'operator_list': list(shift_data['User'].unique())
            }
            
            # Analyze queues per shift
            for queue in shift_data['Queue'].unique():
                queue_shift_data = shift_data[shift_data['Queue'] == queue]
                shift_analysis[shift]['queues'][queue] = {
                    'operators': queue_shift_data['User'].nunique(),
                    'tasks': len(queue_shift_data),
                    'operator_list': list(queue_shift_data['User'].unique())
                }
        
        return shift_analysis
    
    def calculate_shift_productivity_metrics(self):
        """Calculate productivity metrics for each shift and queue combination"""
        if 'shift' not in self.df.columns:
            return None
        
        shift_productivity = {}
        
        for shift in ['Morning', 'Afternoon', 'Night']:
            shift_data = self.df[self.df['shift'] == shift]
            shift_productivity[shift] = {}
            
            for queue in shift_data['Queue'].unique():
                queue_shift_data = shift_data[shift_data['Queue'] == queue]
                
                if len(queue_shift_data) == 0:
                    continue
                
                # Count tasks per operator in this shift-queue combination
                operator_tasks = queue_shift_data.groupby('User').size().reset_index(name='task_count')
                
                if len(operator_tasks) == 0:
                    continue
                
                task_counts = operator_tasks['task_count'].values
                
                shift_productivity[shift][queue] = {
                    'total_operators': len(operator_tasks),
                    'total_tasks': len(queue_shift_data),
                    'task_counts': task_counts,
                    'median': np.median(task_counts),
                    'mean': np.mean(task_counts),
                    'std': np.std(task_counts),
                    'min': np.min(task_counts),
                    'max': np.max(task_counts),
                    'operator_details': operator_tasks.sort_values('task_count', ascending=False)
                }
                
                # Identify operators below and above median
                median_val = shift_productivity[shift][queue]['median']
                shift_productivity[shift][queue]['above_median'] = operator_tasks[
                    operator_tasks['task_count'] > median_val
                ].copy()
                shift_productivity[shift][queue]['below_median'] = operator_tasks[
                    operator_tasks['task_count'] < median_val
                ].copy()
                shift_productivity[shift][queue]['at_median'] = operator_tasks[
                    operator_tasks['task_count'] == median_val
                ].copy()
                
                # Calculate performance ratios
                if shift_productivity[shift][queue]['min'] > 0:
                    shift_productivity[shift][queue]['performance_ratio'] = (
                        shift_productivity[shift][queue]['max'] / shift_productivity[shift][queue]['min']
                    )
                else:
                    shift_productivity[shift][queue]['performance_ratio'] = float('inf')
        
        return shift_productivity
    
    def analyze_operator_activity(self, timestamp_column):
        """Analyze operator activity duration and productivity"""
        if 'shift' not in self.df.columns or timestamp_column not in self.df.columns:
            return None
        
        activity_analysis = {}
        
        for operator in self.df['User'].unique():
            operator_data = self.df[self.df['User'] == operator].copy()
            operator_data = operator_data.sort_values('timestamp')
            
            if len(operator_data) == 0:
                continue
            
            # Calculate activity duration
            first_task = operator_data['timestamp'].min()
            last_task = operator_data['timestamp'].max()
            active_duration = last_task - first_task
            active_hours = active_duration.total_seconds() / 3600
            
            # Get shift information
            shifts_worked = list(operator_data['shift'].unique())
            
            # Calculate productivity metrics
            total_tasks = len(operator_data)
            tasks_per_hour = total_tasks / max(active_hours, 0.1)  # Avoid division by zero
            
            # Queue distribution
            queue_distribution = operator_data['Queue'].value_counts().to_dict()
            
            activity_analysis[operator] = {
                'total_tasks': total_tasks,
                'active_duration_hours': active_hours,
                'tasks_per_hour': tasks_per_hour,
                'shifts_worked': shifts_worked,
                'first_task_time': first_task,
                'last_task_time': last_task,
                'queue_distribution': queue_distribution,
                'primary_queue': operator_data['Queue'].mode().iloc[0] if len(operator_data) > 0 else None
            }
        
        return activity_analysis
    
    def generate_shift_optimization_recommendations(self, shift_productivity):
        """Generate shift-specific optimization recommendations"""
        if not shift_productivity:
            return None
        
        recommendations = {}
        
        for shift in shift_productivity.keys():
            shift_recommendations = {
                'shift_summary': {},
                'queue_issues': {},
                'cross_shift_opportunities': [],
                'resource_reallocation': []
            }
            
            # Analyze each queue in the shift
            total_shift_operators = 0
            underperforming_queues = []
            high_performing_queues = []
            
            for queue, analysis in shift_productivity[shift].items():
                total_shift_operators += analysis['total_operators']
                
                below_median_pct = len(analysis['below_median']) / analysis['total_operators'] * 100
                performance_ratio = analysis['performance_ratio']
                
                queue_health = 'Good'
                if below_median_pct > 50:
                    queue_health = 'Poor'
                    underperforming_queues.append(queue)
                elif performance_ratio < 3:
                    queue_health = 'Excellent'
                    high_performing_queues.append(queue)
                
                shift_recommendations['queue_issues'][queue] = {
                    'health': queue_health,
                    'below_median_operators': len(analysis['below_median']),
                    'performance_ratio': performance_ratio,
                    'potential_improvement': (analysis['median'] * len(analysis['below_median']) - 
                                            analysis['below_median']['task_count'].sum()) if len(analysis['below_median']) > 0 else 0
                }
            
            shift_recommendations['shift_summary'] = {
                'total_operators': total_shift_operators,
                'underperforming_queues': len(underperforming_queues),
                'high_performing_queues': len(high_performing_queues),
                'priority_queues': underperforming_queues[:3]
            }
            
            recommendations[shift] = shift_recommendations
        
        return recommendations
    
    def create_shift_visualizations(self, shift_productivity, operator_activity):
        """Create comprehensive shift-based visualizations"""
        if not shift_productivity:
            return None
        
        # Create a comprehensive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Operators per Shift', 'Productivity by Shift-Queue', 
                          'Shift Performance Comparison', 'Operator Activity Heatmap',
                          'Tasks Distribution by Shift', 'Cross-Shift Operator Analysis'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # 1. Operators per shift
        shifts = list(shift_productivity.keys())
        shift_operators = [sum(len(queue_data['operator_details']) for queue_data in shift_productivity[shift].values()) 
                          for shift in shifts]
        
        fig.add_trace(
            go.Bar(x=shifts, y=shift_operators, name='Operators per Shift',
                   marker_color=['#FFD700', '#4169E1', '#9370DB']),
            row=1, col=1
        )
        
        # 2. Productivity distribution by shift-queue
        for shift in shifts:
            for queue, analysis in shift_productivity[shift].items():
                fig.add_trace(
                    go.Box(y=analysis['task_counts'], name=f'{shift}-{queue}',
                           boxpoints='all', jitter=0.3),
                    row=1, col=2
                )
        
        # 3. Shift performance comparison
        shift_medians = []
        shift_names = []
        for shift in shifts:
            for queue, analysis in shift_productivity[shift].items():
                shift_medians.append(analysis['median'])
                shift_names.append(f'{shift}-{queue}')
        
        if shift_medians:
            fig.add_trace(
                go.Scatter(x=shift_names, y=shift_medians, mode='markers+lines',
                          name='Median Performance', marker_size=8),
                row=2, col=1
            )
        
        # 4. Activity heatmap (if operator activity data available)
        if operator_activity:
            operators = list(operator_activity.keys())[:20]  # Limit to top 20
            activity_matrix = []
            for op in operators:
                activity_data = operator_activity[op]
                activity_matrix.append([
                    activity_data['tasks_per_hour'],
                    activity_data['active_duration_hours'],
                    activity_data['total_tasks']
                ])
            
            if activity_matrix:
                fig.add_trace(
                    go.Heatmap(z=activity_matrix, x=['Tasks/Hour', 'Active Hours', 'Total Tasks'],
                              y=operators, colorscale='Viridis'),
                    row=2, col=2
                )
        
        # 5. Task distribution pie chart
        shift_tasks = [sum(queue_data['total_tasks'] for queue_data in shift_productivity[shift].values()) 
                      for shift in shifts]
        
        fig.add_trace(
            go.Pie(labels=shifts, values=shift_tasks, name="Task Distribution"),
            row=3, col=1
        )
        
        # 6. Cross-shift operator analysis
        if operator_activity:
            cross_shift_data = []
            for op, data in operator_activity.items():
                if len(data['shifts_worked']) > 1:
                    cross_shift_data.append({
                        'operator': op,
                        'shifts': len(data['shifts_worked']),
                        'productivity': data['tasks_per_hour']
                    })
            
            if cross_shift_data:
                cross_df = pd.DataFrame(cross_shift_data)
                fig.add_trace(
                    go.Bar(x=cross_df['operator'], y=cross_df['productivity'],
                          name='Cross-shift Productivity'),
                    row=3, col=2
                )
        
        fig.update_layout(height=1200, title_text="Shift-Based Warehouse Analysis Dashboard")
        return fig

def main():
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner("Loading and processing data..."):
                df = pd.read_excel(uploaded_file)
                analyzer = ShiftAwareWarehouseAnalyzer(df)
                df_clean = analyzer.clean_data()
            
            # Display basic info
            st.success(f"✅ Data loaded successfully! {len(df_clean)} records processed.")
            
            # Detect timestamp columns
            timestamp_cols = analyzer.detect_timestamp_columns()
            
            # Sidebar controls
            st.sidebar.header("🕐 Shift Analysis Setup")
            
            if timestamp_cols:
                st.sidebar.markdown("""
                **📋 Timestamp Column Guide:**
                - **Requested**: When task was scheduled ✅ **(Recommended for shift analysis)**
                - **Confirmed**: When task was accepted by system
                - **Completed**: When task was actually finished
                - **Rotadate**: Product rotation/batch dates
                """)
                
                selected_timestamp = st.sidebar.selectbox(
                    "Select Timestamp Column for Shift Analysis",
                    options=timestamp_cols,
                    index=timestamp_cols.index('Requested') if 'Requested' in timestamp_cols else 0,
                    help="Choose 'Requested' to analyze tasks by when they were assigned to shifts"
                )
                
                st.sidebar.info(f"💡 Using **{selected_timestamp}** time means:\n\n"
                               f"• Morning shift analysis = Tasks {selected_timestamp.lower()} 06:00-14:00\n"
                               f"• Afternoon shift analysis = Tasks {selected_timestamp.lower()} 14:00-22:00\n" 
                               f"• Night shift analysis = Tasks {selected_timestamp.lower()} 22:00-06:00")
                
                if st.sidebar.button("🔄 Process Shift Data", type="primary"):
                    with st.spinner(f"Processing shift data using {selected_timestamp} timestamps..."):
                        success = analyzer.add_shift_information(selected_timestamp)
                        if success:
                            st.sidebar.success(f"✅ Shift data processed using {selected_timestamp} time!")
                            
                            # Show quick shift breakdown
                            shift_counts = df_clean.groupby('shift').size()
                            st.sidebar.markdown("**📊 Tasks by Shift:**")
                            for shift, count in shift_counts.items():
                                st.sidebar.markdown(f"• {shift}: {count} tasks")
                        else:
                            st.sidebar.error("❌ Failed to process shift data")
            else:
                st.sidebar.warning("⚠️ No timestamp columns detected. Shift analysis not available.")
                st.sidebar.info("Ensure your data contains columns like 'Requested', 'Confirmed', or 'Completed'")
            
            # Display shift definitions  
            st.sidebar.header("📋 Shift Definitions")
            st.sidebar.markdown("""
            **☀️ Morning Shift**: 06:00 - 14:00  
            **🌅 Afternoon Shift**: 14:00 - 22:00  
            **🌙 Night Shift**: 22:00 - 06:00
            
            *Tasks are assigned to shifts based on their **request time**, not completion time.*
            """)
            
            # Analysis controls
            st.sidebar.header("🔍 Analysis Controls")
            
            queues = [q for q in df_clean['Queue'].unique() if pd.notna(q)]
            selected_queues = st.sidebar.multiselect(
                "Select Queues to Analyze",
                options=queues,
                default=queues
            )
            
            # Main analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Overall Analysis", 
                "🕐 Shift Analysis", 
                "👥 Operator Activity", 
                "🎯 Optimization"
            ])
            
            with tab1:
                st.header("📊 Overall Analysis")
                
                # Overall metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Queues", len(queues))
                with col2:
                    st.metric("Total Operators", df_clean['User'].nunique())
                with col3:
                    st.metric("Total Tasks", len(df_clean))
                with col4:
                    avg_productivity = df_clean.groupby('User').size().mean()
                    st.metric("Avg Tasks/Operator", f"{avg_productivity:.1f}")
                
                # Basic queue analysis (without shift breakdown)
                overall_analysis = {}
                for queue in selected_queues:
                    queue_data = df_clean[df_clean['Queue'] == queue]
                    operator_tasks = queue_data.groupby('User').size()
                    
                    overall_analysis[queue] = {
                        'operators': len(operator_tasks),
                        'tasks': len(queue_data),
                        'median': operator_tasks.median(),
                        'min': operator_tasks.min(),
                        'max': operator_tasks.max()
                    }
                
                # Display overall queue metrics
                st.subheader("Queue Overview")
                queue_df = pd.DataFrame(overall_analysis).T
                queue_df.columns = ['Operators', 'Total Tasks', 'Median Tasks/Op', 'Min Tasks/Op', 'Max Tasks/Op']
                st.dataframe(queue_df, use_container_width=True)
            
            with tab2:
                st.header("🕐 Shift Analysis")
                
                if 'shift' in df_clean.columns:
                    # Explain the analysis approach
                    st.info(f"📊 **Analysis Approach**: Tasks are grouped by their **{selected_timestamp}** time, meaning we're analyzing "
                           f"how operators perform on workload assigned to their shift, regardless of when tasks were actually completed.")
                    
                    # Shift overview
                    shift_summary = analyzer.analyze_operators_per_shift()
                    shift_productivity = analyzer.calculate_shift_productivity_metrics()
                    
                    if shift_summary:
                        st.subheader("👥 Operators per Shift")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            morning_ops = shift_summary['Morning']['total_operators']
                            morning_tasks = shift_summary['Morning']['total_tasks']
                            st.markdown('<div class="shift-morning">', unsafe_allow_html=True)
                            st.metric("☀️ Morning Shift", f"{morning_ops} operators")
                            st.metric("Morning Tasks", f"{morning_tasks}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            afternoon_ops = shift_summary['Afternoon']['total_operators']
                            afternoon_tasks = shift_summary['Afternoon']['total_tasks']
                            st.markdown('<div class="shift-afternoon">', unsafe_allow_html=True)
                            st.metric("🌅 Afternoon Shift", f"{afternoon_ops} operators")
                            st.metric("Afternoon Tasks", f"{afternoon_tasks}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            night_ops = shift_summary['Night']['total_operators']
                            night_tasks = shift_summary['Night']['total_tasks']
                            st.markdown('<div class="shift-night">', unsafe_allow_html=True)
                            st.metric("🌙 Night Shift", f"{night_ops} operators")
                            st.metric("Night Tasks", f"{night_tasks}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Detailed shift-queue analysis
                        st.subheader("🎯 Queue Performance by Shift")
                        
                        for shift in ['Morning', 'Afternoon', 'Night']:
                            if shift in shift_productivity:
                                st.markdown(f"### {shift} Shift Analysis")
                                
                                shift_queues = list(shift_productivity[shift].keys())
                                if not shift_queues:
                                    st.info(f"No queue data available for {shift} shift")
                                    continue
                                
                                # Create columns for each queue in this shift
                                cols = st.columns(min(len(shift_queues), 4))
                                
                                for i, queue in enumerate(shift_queues):
                                    if i < len(cols):
                                        with cols[i]:
                                            analysis = shift_productivity[shift][queue]
                                            st.markdown(f"**{queue}**")
                                            st.metric("Operators", analysis['total_operators'])
                                            st.metric("Median Tasks", f"{analysis['median']:.1f}")
                                            st.metric("Below Median", len(analysis['below_median']))
                                
                                # Detailed analysis for each queue
                                for queue in shift_queues:
                                    if queue in selected_queues:
                                        analysis = shift_productivity[shift][queue]
                                        
                                        with st.expander(f"📈 {shift} - {queue} Detailed Analysis"):
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.markdown("**🔻 Below Median Operators**")
                                                if len(analysis['below_median']) > 0:
                                                    st.dataframe(analysis['below_median'])
                                                else:
                                                    st.success("No operators below median!")
                                            
                                            with col2:
                                                st.markdown("**🔺 Above Median Operators**")
                                                if len(analysis['above_median']) > 0:
                                                    st.dataframe(analysis['above_median'])
                                                else:
                                                    st.info("All operators at or below median")
                else:
                    st.warning("⚠️ Shift analysis not available. Please select a timestamp column and process shift data.")
            
            with tab3:
                st.header("👥 Operator Activity Analysis")
                
                if 'shift' in df_clean.columns and timestamp_cols:
                    operator_activity = analyzer.analyze_operator_activity(selected_timestamp)
                    
                    if operator_activity:
                        st.subheader("🕐 Operator Activity Duration & Productivity")
                        
                        # Convert to DataFrame for easier display
                        activity_df = pd.DataFrame(operator_activity).T
                        activity_df = activity_df.sort_values('tasks_per_hour', ascending=False)
                        
                        # Top performers by productivity
                        st.markdown("**🏆 Top 10 Most Productive Operators (Tasks per Hour)**")
                        top_performers = activity_df.head(10)[['total_tasks', 'active_duration_hours', 'tasks_per_hour', 'shifts_worked']]
                        top_performers.columns = ['Total Tasks', 'Active Hours', 'Tasks/Hour', 'Shifts Worked']
                        top_performers['Active Hours'] = top_performers['Active Hours'].round(2)
                        top_performers['Tasks/Hour'] = top_performers['Tasks/Hour'].round(2)
                        st.dataframe(top_performers, use_container_width=True)
                        
                        # Activity insights
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_active_hours = activity_df['active_duration_hours'].mean()
                            st.metric("Avg Active Hours", f"{avg_active_hours:.1f}h")
                        with col2:
                            avg_productivity = activity_df['tasks_per_hour'].mean()
                            st.metric("Avg Tasks/Hour", f"{avg_productivity:.1f}")
                        with col3:
                            cross_shift_ops = sum(1 for _, data in operator_activity.items() if len(data['shifts_worked']) > 1)
                            st.metric("Cross-Shift Operators", cross_shift_ops)
                        with col4:
                            max_productivity = activity_df['tasks_per_hour'].max()
                            st.metric("Max Tasks/Hour", f"{max_productivity:.1f}")
                        
                        # Detailed operator analysis
                        st.subheader("🔍 Detailed Operator Analysis")
                        
                        # Filter options
                        col1, col2 = st.columns(2)
                        with col1:
                            min_tasks = st.slider("Minimum Tasks", 
                                                min_value=int(activity_df['total_tasks'].min()),
                                                max_value=int(activity_df['total_tasks'].max()),
                                                value=int(activity_df['total_tasks'].min()))
                        with col2:
                            min_hours = st.slider("Minimum Active Hours",
                                                min_value=0.0,
                                                max_value=float(activity_df['active_duration_hours'].max()),
                                                value=0.0)
                        
                        # Filtered data
                        filtered_activity = activity_df[
                            (activity_df['total_tasks'] >= min_tasks) & 
                            (activity_df['active_duration_hours'] >= min_hours)
                        ]
                        
                        st.markdown(f"**Showing {len(filtered_activity)} operators matching criteria:**")
                        display_df = filtered_activity[['total_tasks', 'active_duration_hours', 'tasks_per_hour', 'shifts_worked', 'primary_queue']]
                        display_df.columns = ['Total Tasks', 'Active Hours', 'Tasks/Hour', 'Shifts Worked', 'Primary Queue']
                        display_df['Active Hours'] = display_df['Active Hours'].round(2)
                        display_df['Tasks/Hour'] = display_df['Tasks/Hour'].round(2)
                        st.dataframe(display_df, use_container_width=True)
                        
                else:
                    st.warning("⚠️ Operator activity analysis requires timestamp data. Please process shift data first.")
            
            with tab4:
                st.header("🎯 Optimization Recommendations")
                
                if 'shift' in df_clean.columns:
                    shift_productivity = analyzer.calculate_shift_productivity_metrics()
                    optimization_recs = analyzer.generate_shift_optimization_recommendations(shift_productivity)
                    
                    if optimization_recs:
                        st.subheader("📋 Shift-Specific Optimization Opportunities")
                        
                        for shift, recommendations in optimization_recs.items():
                            st.markdown(f"### {shift} Shift Optimization")
                            
                            summary = recommendations['shift_summary']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Operators", summary['total_operators'])
                            with col2:
                                st.metric("Underperforming Queues", summary['underperforming_queues'])
                            with col3:
                                st.metric("High Performing Queues", summary['high_performing_queues'])
                            
                            # Queue-specific recommendations
                            if recommendations['queue_issues']:
                                st.markdown("**Queue Performance Issues:**")
                                for queue, issue in recommendations['queue_issues'].items():
                                    if issue['health'] == 'Poor':
                                        st.markdown(f'<div class="danger-card">❌ <strong>{queue}</strong>: {issue["below_median_operators"]} operators below median, potential +{issue["potential_improvement"]:.0f} tasks/day</div>', unsafe_allow_html=True)
                                    elif issue['health'] == 'Excellent':
                                        st.markdown(f'<div class="optimization-card">✅ <strong>{queue}</strong>: Excellent performance (ratio {issue["performance_ratio"]:.1f}:1)</div>', unsafe_allow_html=True)
                            
                            st.divider()
                        
                        # Overall recommendations
                        st.subheader("🚀 Implementation Strategy")
                        st.markdown("""
                        **Phase 1: Immediate Actions (Week 1-2)**
                        - Focus on underperforming operators in priority queues
                        - Implement peer mentoring from top performers
                        - Analyze equipment/tool availability issues
                        
                        **Phase 2: Cross-Training (Week 3-4)**
                        - Cross-train operators between shifts for coverage
                        - Develop shift handover protocols
                        - Implement performance tracking systems
                        
                        **Phase 3: Optimization (Month 2+)**
                        - Redistribute operators based on shift demand
                        - Implement dynamic scheduling based on productivity patterns
                        - Monitor and adjust based on performance metrics
                        """)
                
                # Visualizations
                st.subheader("📈 Shift Analysis Dashboard")
                
                if 'shift' in df_clean.columns:
                    shift_productivity = analyzer.calculate_shift_productivity_metrics()
                    operator_activity = analyzer.analyze_operator_activity(selected_timestamp) if timestamp_cols else None
                    
                    viz_fig = analyzer.create_shift_visualizations(shift_productivity, operator_activity)
                    if viz_fig:
                        st.plotly_chart(viz_fig, use_container_width=True)
                else:
                    st.info("📊 Shift visualizations will be available after processing timestamp data.")
                
                # Export functionality
                st.subheader("💾 Export Analysis Results")
                
                if st.button("📥 Generate Export Data"):
                    with st.spinner("Preparing export data..."):
                        # Prepare comprehensive export
                        export_data = []
                        
                        for _, row in df_clean.iterrows():
                            export_row = {
                                'Operator': row['User'],
                                'Queue': row['Queue'],
                                'Shift': row.get('shift', 'Unknown'),
                                'Task': row.get('Task', ''),
                                'Quantity': row.get('Quantity', 0)
                            }
                            
                            if 'timestamp' in df_clean.columns:
                                export_row['Timestamp'] = row['timestamp']
                            
                            export_data.append(export_row)
                        
                        export_df = pd.DataFrame(export_data)
                        
                        # Add productivity metrics
                        if 'shift' in df_clean.columns:
                            productivity_summary = []
                            shift_productivity = analyzer.calculate_shift_productivity_metrics()
                            
                            for shift in shift_productivity:
                                for queue, analysis in shift_productivity[shift].items():
                                    for _, op_row in analysis['operator_details'].iterrows():
                                        productivity_summary.append({
                                            'Shift': shift,
                                            'Queue': queue,
                                            'Operator': op_row['User'],
                                            'Tasks_Completed': op_row['task_count'],
                                            'Queue_Median': analysis['median'],
                                            'Performance_vs_Median': 'Above' if op_row['task_count'] > analysis['median'] else 
                                                                   ('Below' if op_row['task_count'] < analysis['median'] else 'Equal'),
                                            'Improvement_Potential': max(0, analysis['median'] - op_row['task_count'])
                                        })
                            
                            productivity_df = pd.DataFrame(productivity_summary)
                            
                            # Create download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                csv1 = export_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Raw Data (CSV)",
                                    data=csv1,
                                    file_name="warehouse_shift_analysis_raw.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                csv2 = productivity_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Productivity Analysis (CSV)",
                                    data=csv2,
                                    file_name="warehouse_shift_productivity.csv",
                                    mime="text/csv"
                                )
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your Excel file contains 'User' and 'Queue' columns, and at least one timestamp column.")
    
    else:
        # Welcome screen
        st.info("👋 Welcome! Please upload your warehouse operations Excel file to begin shift analysis.")
        
        # Sample data format with timestamp
        st.subheader("📋 Required Data Format")
        sample_data = {
            'User': ['54001', '54002', '54003', '54001', '54002'],
            'Queue': ['FLRP', 'FLRP', 'VC1P', 'FL1P', 'VC1P'],
            'Task': ['RP', 'RP', 'SOCS', 'SOCS', 'SOCS'],
            'Quantity': [36, 42, 63, 45, 52],
            'Requested': ['2024-01-15 08:30:00', '2024-01-15 09:15:00', '2024-01-15 16:45:00', 
                         '2024-01-15 23:20:00', '2024-01-15 17:30:00'],
            'Completed': ['2024-01-15 10:45:00', '2024-01-15 11:30:00', '2024-01-15 18:20:00',
                         '2024-01-16 01:15:00', '2024-01-15 19:45:00']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        
        st.markdown("""
        **Required columns:**
        - **User**: Operator ID or identifier
        - **Queue**: Queue name (FLRP, FL1P, VC1P, etc.)
        - **Timestamp column**: Date/time information for shift analysis
        
        **📋 Timestamp Column Meanings:**
        - **Requested**: When task was scheduled ✅ **(Best for shift analysis)**
        - **Confirmed**: When task was accepted by system  
        - **Completed**: When task was actually finished
        - **Rotadate**: Product rotation/batch dates
        
        **Optional columns:**
        - **Task**: Task type (RP, SOCS, SFCS, etc.)
        - **Quantity**: Task quantity
        - **From Zone**, **To Zone**: Location information
        
        **🕐 Shift Analysis Logic:**
        Using **Requested** time ensures we analyze:
        - **Morning workload**: Tasks requested 06:00-14:00 (handled by morning operators)
        - **Afternoon workload**: Tasks requested 14:00-22:00 (handled by afternoon operators)  
        - **Night workload**: Tasks requested 22:00-06:00 (handled by night operators)
        
        This gives you meaningful insights into shift productivity based on actual shift assignments!
        """)

if __name__ == "__main__":
    main()
