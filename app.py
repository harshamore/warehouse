import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, time, timedelta
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Warehouse Shift Optimizer v3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .shift-morning { background: linear-gradient(90deg, #FFE4B5 0%, #FFF8DC 100%); padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700; margin: 10px 0; }
    .shift-afternoon { background: linear-gradient(90deg, #E6F3FF 0%, #F0F8FF 100%); padding: 15px; border-radius: 10px; border-left: 5px solid #4169E1; margin: 10px 0; }
    .shift-night { background: linear-gradient(90deg, #2F2F4F 0%, #483D8B 100%); color: white; padding: 15px; border-radius: 10px; border-left: 5px solid #9370DB; margin: 10px 0; }
    .optimization-card { background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin: 10px 0; }
    .danger-card { background-color: #f8d7da; padding: 15px; border-radius: 10px; border-left: 5px solid #dc3545; margin: 10px 0; }
    .warning-card { background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; margin: 10px 0; }
    .multi-queue-card { background-color: #e1f5fe; padding: 15px; border-radius: 10px; border-left: 5px solid #00acc1; margin: 10px 0; }
    .utilization-card { background-color: #f3e5f5; padding: 15px; border-radius: 10px; border-left: 5px solid #9c27b0; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏭 Enhanced Warehouse Shift Optimizer - Multi-Queue & Utilization Analysis")
st.markdown("**Advanced analytics for operator productivity, queue switching patterns, and shift utilization**")

class EnhancedShiftOptimizer:
    def __init__(self, df):
        self.df = df
        self.shift_definitions = {
            'Morning': {'start': time(6, 0), 'end': time(14, 0), 'icon': '☀️', 'label': '06:00-14:00', 'duration': 8},
            'Afternoon': {'start': time(14, 0), 'end': time(22, 0), 'icon': '🌅', 'label': '14:00-22:00', 'duration': 8}, 
            'Night': {'start': time(22, 0), 'end': time(6, 0), 'icon': '🌙', 'label': '22:00-06:00', 'duration': 8}
        }
        
    def clean_data(self):
        """Clean and prepare data"""
        self.df = self.df.dropna(subset=['User', 'Queue'])
        
        # Convert numeric columns
        if 'Quantity' in self.df.columns:
            self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce')
        
        return self.df
    
    def determine_shift(self, timestamp):
        """Determine shift based on confirmed timestamp"""
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
            # Night: 22:00 - 06:00
            else:
                return 'Night'
        except:
            return 'Unknown'
    
    def filter_by_shift(self, selected_shift):
        """Filter data by selected shift using Confirmed timestamp"""
        if 'Confirmed' not in self.df.columns:
            return pd.DataFrame()
        
        # Add confirmed timestamp and shift classification
        self.df['confirmed_time'] = pd.to_datetime(self.df['Confirmed'])
        self.df['shift'] = self.df['confirmed_time'].apply(self.determine_shift)
        
        # Return filtered data for selected shift
        return self.df[self.df['shift'] == selected_shift].copy()
    
    def analyze_multi_queue_operators(self, shift_data):
        """Analyze operators who work across different queues"""
        if shift_data.empty:
            return {}
        
        # Create operator-queue mapping
        operator_queues = {}
        operator_tasks_by_queue = {}
        operator_time_by_queue = {}
        
        for operator in shift_data['User'].unique():
            operator_data = shift_data[shift_data['User'] == operator].copy()
            
            # Get unique queues for this operator
            queues = operator_data['Queue'].unique()
            operator_queues[operator] = list(queues)
            
            # Tasks and time per queue
            operator_tasks_by_queue[operator] = {}
            operator_time_by_queue[operator] = {}
            
            for queue in queues:
                queue_tasks = operator_data[operator_data['Queue'] == queue]
                operator_tasks_by_queue[operator][queue] = len(queue_tasks)
                
                # Calculate time spent in each queue
                if len(queue_tasks) > 1:
                    queue_times = pd.to_datetime(queue_tasks['confirmed_time']).sort_values()
                    time_span = queue_times.max() - queue_times.min()
                    operator_time_by_queue[operator][queue] = time_span.total_seconds() / 3600
                else:
                    operator_time_by_queue[operator][queue] = 0.25  # 15 minutes for single task
        
        # Categorize operators
        single_queue_ops = {op: queues for op, queues in operator_queues.items() if len(queues) == 1}
        multi_queue_ops = {op: queues for op, queues in operator_queues.items() if len(queues) > 1}
        
        # Detailed analysis for multi-queue operators
        multi_queue_analysis = {}
        for operator, queues in multi_queue_ops.items():
            total_tasks = sum(operator_tasks_by_queue[operator].values())
            total_time = sum(operator_time_by_queue[operator].values())
            
            # Calculate queue switching frequency
            operator_timeline = shift_data[shift_data['User'] == operator].copy()
            operator_timeline = operator_timeline.sort_values('confirmed_time')
            
            switches = 0
            prev_queue = None
            for _, task in operator_timeline.iterrows():
                if prev_queue and task['Queue'] != prev_queue:
                    switches += 1
                prev_queue = task['Queue']
            
            # Primary vs secondary queues
            task_counts = operator_tasks_by_queue[operator]
            primary_queue = max(task_counts.keys(), key=lambda q: task_counts[q])
            primary_tasks = task_counts[primary_queue]
            secondary_tasks = total_tasks - primary_tasks
            
            multi_queue_analysis[operator] = {
                'queues': queues,
                'queue_count': len(queues),
                'total_tasks': total_tasks,
                'total_time': total_time,
                'tasks_per_queue': operator_tasks_by_queue[operator],
                'time_per_queue': operator_time_by_queue[operator],
                'queue_switches': switches,
                'switches_per_hour': switches / max(total_time, 0.1),
                'primary_queue': primary_queue,
                'primary_tasks': primary_tasks,
                'secondary_tasks': secondary_tasks,
                'queue_flexibility_score': len(queues) * (switches / max(total_tasks, 1))
            }
        
        return {
            'single_queue': single_queue_ops,
            'multi_queue': multi_queue_ops,
            'multi_queue_analysis': multi_queue_analysis,
            'summary': {
                'total_operators': len(operator_queues),
                'single_queue_count': len(single_queue_ops),
                'multi_queue_count': len(multi_queue_ops),
                'multi_queue_percentage': (len(multi_queue_ops) / len(operator_queues)) * 100
            }
        }
    
    def analyze_shift_utilization(self, shift_data, selected_shift):
        """Analyze how operators are utilized throughout the shift"""
        if shift_data.empty:
            return {}
        
        shift_duration = self.shift_definitions[selected_shift]['duration']  # hours
        utilization_analysis = {}
        
        for operator in shift_data['User'].unique():
            operator_data = shift_data[shift_data['User'] == operator].copy()
            operator_data = operator_data.sort_values('confirmed_time')
            
            if len(operator_data) == 0:
                continue
            
            # Get timeline
            confirmed_times = pd.to_datetime(operator_data['confirmed_time'])
            first_task = confirmed_times.min()
            last_task = confirmed_times.max()
            
            # Calculate working span
            working_span = last_task - first_task
            working_hours = working_span.total_seconds() / 3600
            
            # Calculate idle periods between tasks
            idle_periods = []
            if len(confirmed_times) > 1:
                for i in range(1, len(confirmed_times)):
                    time_gap = confirmed_times.iloc[i] - confirmed_times.iloc[i-1]
                    gap_minutes = time_gap.total_seconds() / 60
                    if gap_minutes > 15:  # Consider gaps > 15 minutes as idle time
                        idle_periods.append(gap_minutes)
            
            total_idle_time = sum(idle_periods) / 60  # Convert to hours
            active_time = working_hours - total_idle_time
            
            # Shift utilization metrics
            shift_start_time = self.get_shift_start_datetime(first_task, selected_shift)
            shift_end_time = shift_start_time + timedelta(hours=shift_duration)
            
            # Check if operator worked the full shift or partial
            time_from_shift_start = (first_task - shift_start_time).total_seconds() / 3600
            time_to_shift_end = (shift_end_time - last_task).total_seconds() / 3600
            
            # Calculate utilization percentages
            working_span_utilization = (working_hours / shift_duration) * 100
            active_time_utilization = (active_time / shift_duration) * 100
            
            # Task frequency analysis
            total_tasks = len(operator_data)
            tasks_per_active_hour = total_tasks / max(active_time, 0.1)
            avg_time_between_tasks = (working_hours * 60) / max(total_tasks - 1, 1)  # minutes
            
            # Queue distribution over time
            queue_timeline = []
            for _, task in operator_data.iterrows():
                task_time = pd.to_datetime(task['confirmed_time'])
                hours_into_shift = (task_time - shift_start_time).total_seconds() / 3600
                queue_timeline.append({
                    'hours_into_shift': hours_into_shift,
                    'queue': task['Queue'],
                    'timestamp': task_time
                })
            
            utilization_analysis[operator] = {
                'total_tasks': total_tasks,
                'working_hours': round(working_hours, 2),
                'active_hours': round(active_time, 2),
                'idle_hours': round(total_idle_time, 2),
                'working_span_utilization': round(working_span_utilization, 1),
                'active_time_utilization': round(active_time_utilization, 1),
                'tasks_per_active_hour': round(tasks_per_active_hour, 2),
                'avg_time_between_tasks': round(avg_time_between_tasks, 1),
                'idle_periods_count': len(idle_periods),
                'longest_idle_period': max(idle_periods) if idle_periods else 0,
                'first_task_time': first_task,
                'last_task_time': last_task,
                'late_start_hours': max(0, time_from_shift_start),
                'early_end_hours': max(0, time_to_shift_end),
                'queue_timeline': queue_timeline
            }
        
        return utilization_analysis
    
    def get_shift_start_datetime(self, reference_datetime, shift_name):
        """Get the actual start time of the shift for a given reference datetime"""
        shift_info = self.shift_definitions[shift_name]
        date = reference_datetime.date()
        
        if shift_name == 'Night':
            # Night shift starts on the previous day
            if reference_datetime.time() < time(12, 0):  # If it's early morning, it's part of previous night's shift
                date = date - timedelta(days=1)
            shift_start = datetime.combine(date, shift_info['start'])
        else:
            shift_start = datetime.combine(date, shift_info['start'])
        
        return shift_start
    
    def calculate_operator_active_hours(self, shift_data):
        """Calculate active hours for each operator based on confirmed timestamps"""
        operator_activity = {}
        
        for operator in shift_data['User'].unique():
            operator_data = shift_data[shift_data['User'] == operator].copy()
            
            if len(operator_data) == 0:
                continue
            
            # Get confirmed timestamps for this operator
            confirmed_times = pd.to_datetime(operator_data['confirmed_time'])
            confirmed_times = confirmed_times.sort_values()
            
            # Calculate active duration (first to last confirmed task)
            if len(confirmed_times) > 1:
                first_task = confirmed_times.min()
                last_task = confirmed_times.max()
                active_duration = last_task - first_task
                active_hours = active_duration.total_seconds() / 3600
            else:
                # Single task - assume 15 minutes activity
                active_hours = 0.25
            
            # Calculate productivity metrics
            total_tasks = len(operator_data)
            tasks_per_hour = total_tasks / max(active_hours, 0.1)  # Avoid division by zero
            
            # Queue distribution
            queue_distribution = operator_data['Queue'].value_counts().to_dict()
            primary_queue = operator_data['Queue'].mode().iloc[0] if len(operator_data) > 0 else None
            
            operator_activity[operator] = {
                'total_tasks': total_tasks,
                'active_hours': round(active_hours, 2),
                'tasks_per_hour': round(tasks_per_hour, 2),
                'first_task_time': confirmed_times.min(),
                'last_task_time': confirmed_times.max(),
                'queue_distribution': queue_distribution,
                'primary_queue': primary_queue
            }
        
        return operator_activity

    def calculate_queue_productivity(self, shift_data):
        """Calculate productivity metrics for each queue in the selected shift"""
        if shift_data.empty:
            return {}
        
        # First calculate operator active hours
        operator_activity = self.calculate_operator_active_hours(shift_data)
        
        results = {}
        queues = shift_data['Queue'].unique()
        
        for queue in queues:
            queue_data = shift_data[shift_data['Queue'] == queue]
            
            # Count tasks per operator and add active hours info
            operator_tasks = queue_data.groupby('User').size().reset_index(name='task_count')
            
            # Add active hours and productivity rate for each operator
            operator_tasks['active_hours'] = operator_tasks['User'].map(
                lambda x: operator_activity.get(x, {}).get('active_hours', 0)
            )
            operator_tasks['tasks_per_hour'] = operator_tasks['User'].map(
                lambda x: operator_activity.get(x, {}).get('tasks_per_hour', 0)
            )
            
            if len(operator_tasks) == 0:
                continue
                
            task_counts = operator_tasks['task_count'].values
            
            results[queue] = {
                'total_operators': len(operator_tasks),
                'total_tasks': len(queue_data),
                'median': np.median(task_counts),
                'mean': np.mean(task_counts),
                'min': np.min(task_counts),
                'max': np.max(task_counts),
                'operator_details': operator_tasks.sort_values('task_count', ascending=False),
                'avg_active_hours': operator_tasks['active_hours'].mean(),
                'avg_productivity_rate': operator_tasks['tasks_per_hour'].mean()
            }
            
            # Identify operators above/below median
            median_val = results[queue]['median']
            results[queue]['above_median'] = operator_tasks[
                operator_tasks['task_count'] > median_val
            ].copy()
            results[queue]['below_median'] = operator_tasks[
                operator_tasks['task_count'] < median_val
            ].copy()
            
            # Performance ratio
            if results[queue]['min'] > 0:
                results[queue]['performance_ratio'] = results[queue]['max'] / results[queue]['min']
            else:
                results[queue]['performance_ratio'] = float('inf')
        
        return results
    
    def generate_optimization_recommendations(self, queue_analysis, multi_queue_analysis, utilization_analysis, selected_shift):
        """Generate comprehensive optimization recommendations"""
        recommendations = {
            'shift_summary': {},
            'critical_issues': [],
            'opportunities': [],
            'action_items': [],
            'efficiency_insights': [],
            'multi_queue_insights': [],
            'utilization_insights': []
        }
        
        total_operators = sum(analysis['total_operators'] for analysis in queue_analysis.values()) if queue_analysis else 0
        total_tasks = sum(analysis['total_tasks'] for analysis in queue_analysis.values()) if queue_analysis else 0
        
        recommendations['shift_summary'] = {
            'shift': selected_shift,
            'total_operators': total_operators,
            'total_tasks': total_tasks,
            'queues_analyzed': len(queue_analysis),
            'multi_queue_operators': multi_queue_analysis['summary']['multi_queue_count'],
            'multi_queue_percentage': multi_queue_analysis['summary']['multi_queue_percentage']
        }
        
        # Multi-queue insights
        if multi_queue_analysis['multi_queue_analysis']:
            # Find most flexible operators
            flexible_ops = sorted(multi_queue_analysis['multi_queue_analysis'].items(), 
                                key=lambda x: x[1]['queue_flexibility_score'], reverse=True)[:3]
            
            for op, analysis in flexible_ops:
                recommendations['multi_queue_insights'].append(
                    f"High flexibility: {op} works {analysis['queue_count']} queues with {analysis['queue_switches']} switches ({analysis['switches_per_hour']:.1f} switches/hour)"
                )
            
            # Identify operators who could be better utilized across queues
            single_queue_heavy = [op for op, analysis in multi_queue_analysis['multi_queue_analysis'].items() 
                                if analysis['primary_tasks'] / analysis['total_tasks'] > 0.8]
            
            if single_queue_heavy:
                recommendations['opportunities'].append(
                    f"Cross-training opportunity: {len(single_queue_heavy)} multi-queue operators are 80%+ focused on single queue"
                )
        
        # Utilization insights
        if utilization_analysis:
            low_utilization_ops = [op for op, analysis in utilization_analysis.items() 
                                 if analysis['active_time_utilization'] < 50]
            high_idle_ops = [op for op, analysis in utilization_analysis.items() 
                           if analysis['idle_hours'] > 2]
            
            if low_utilization_ops:
                recommendations['utilization_insights'].append(
                    f"Low utilization: {len(low_utilization_ops)} operators with <50% active time utilization"
                )
                
            if high_idle_ops:
                recommendations['utilization_insights'].append(
                    f"High idle time: {len(high_idle_ops)} operators with >2 hours idle time during shift"
                )
            
            # Calculate average utilization
            avg_utilization = np.mean([analysis['active_time_utilization'] for analysis in utilization_analysis.values()])
            if avg_utilization < 60:
                recommendations['critical_issues'].append(
                    f"Shift utilization only {avg_utilization:.1f}% - significant idle time detected"
                )
        
        # Existing queue analysis insights
        for queue, analysis in queue_analysis.items():
            below_median_count = len(analysis['below_median'])
            below_median_pct = (below_median_count / analysis['total_operators']) * 100
            
            # Critical issues
            if below_median_pct > 50:
                recommendations['critical_issues'].append(
                    f"{queue}: {below_median_count} operators ({below_median_pct:.0f}%) below median performance"
                )
            
            if analysis['performance_ratio'] > 10:
                recommendations['critical_issues'].append(
                    f"{queue}: Extreme performance variation ({analysis['performance_ratio']:.1f}:1 ratio)"
                )
            
            # Opportunities
            if below_median_count > 0:
                potential_gain = (analysis['median'] * below_median_count) - analysis['below_median']['task_count'].sum()
                recommendations['opportunities'].append(
                    f"{queue}: Bring {below_median_count} operators to median → +{potential_gain:.0f} tasks/day (+{potential_gain/analysis['total_tasks']*100:.1f}%)"
                )
        
        return recommendations
    
    def create_multi_queue_visualization(self, multi_queue_analysis):
        """Create visualizations for multi-queue operator analysis"""
        if not multi_queue_analysis['multi_queue_analysis']:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Queue Count Distribution',
                'Queue Switching Frequency', 
                'Primary vs Secondary Tasks',
                'Flexibility Score Distribution'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Prepare data
        ops_data = []
        for op, analysis in multi_queue_analysis['multi_queue_analysis'].items():
            ops_data.append({
                'operator': op,
                'queue_count': analysis['queue_count'],
                'switches_per_hour': analysis['switches_per_hour'],
                'primary_tasks': analysis['primary_tasks'],
                'secondary_tasks': analysis['secondary_tasks'],
                'flexibility_score': analysis['queue_flexibility_score'],
                'total_tasks': analysis['total_tasks']
            })
        
        ops_df = pd.DataFrame(ops_data)
        
        # 1. Queue count distribution
        queue_count_dist = ops_df['queue_count'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=queue_count_dist.index, y=queue_count_dist.values, 
                  name='Queue Count', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Queue switching frequency
        fig.add_trace(
            go.Scatter(x=ops_df['total_tasks'], y=ops_df['switches_per_hour'],
                      mode='markers', name='Switches/Hour',
                      text=ops_df['operator'], textposition='top center',
                      marker=dict(size=10, color='orange')),
            row=1, col=2
        )
        
        # 3. Primary vs Secondary tasks
        fig.add_trace(
            go.Bar(x=ops_df['operator'][:10], y=ops_df['primary_tasks'][:10], 
                  name='Primary Queue', marker_color='green'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=ops_df['operator'][:10], y=ops_df['secondary_tasks'][:10], 
                  name='Secondary Queues', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # 4. Flexibility score distribution
        fig.add_trace(
            go.Histogram(x=ops_df['flexibility_score'], name='Flexibility Score',
                        marker_color='purple', nbinsx=10),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Multi-Queue Operator Analysis", showlegend=True)
        return fig
    
    def create_utilization_visualization(self, utilization_analysis, selected_shift):
        """Create visualizations for shift utilization analysis"""
        if not utilization_analysis:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Active Time Utilization Distribution',
                'Idle Time vs Working Hours',
                'Task Rate vs Utilization',
                'Shift Coverage Timeline'
            ),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Prepare data
        util_data = []
        for op, analysis in utilization_analysis.items():
            util_data.append({
                'operator': op,
                'active_time_utilization': analysis['active_time_utilization'],
                'idle_hours': analysis['idle_hours'],
                'working_hours': analysis['working_hours'],
                'tasks_per_active_hour': analysis['tasks_per_active_hour'],
                'late_start_hours': analysis['late_start_hours'],
                'early_end_hours': analysis['early_end_hours']
            })
        
        util_df = pd.DataFrame(util_data)
        
        # 1. Utilization distribution
        fig.add_trace(
            go.Histogram(x=util_df['active_time_utilization'], 
                        name='Utilization %', marker_color='skyblue', nbinsx=15),
            row=1, col=1
        )
        
        # 2. Idle vs Working hours
        fig.add_trace(
            go.Scatter(x=util_df['working_hours'], y=util_df['idle_hours'],
                      mode='markers', name='Idle vs Working',
                      text=util_df['operator'], textposition='top center',
                      marker=dict(size=10, color='red')),
            row=1, col=2
        )
        
        # 3. Task rate vs utilization
        fig.add_trace(
            go.Scatter(x=util_df['active_time_utilization'], y=util_df['tasks_per_active_hour'],
                      mode='markers', name='Rate vs Utilization',
                      text=util_df['operator'], textposition='top center',
                      marker=dict(size=10, color='green')),
            row=2, col=1
        )
        
        # 4. Shift coverage (late start vs early end)
        fig.add_trace(
            go.Scatter(x=util_df['late_start_hours'], y=util_df['early_end_hours'],
                      mode='markers', name='Shift Coverage',
                      text=util_df['operator'], textposition='top center',
                      marker=dict(size=10, color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text=f"{selected_shift} Shift Utilization Analysis", showlegend=True)
        return fig

def main():
    # Sidebar - File Upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        help="Upload warehouse operations Excel file with 'Confirmed' timestamp column"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process data
            with st.spinner("Loading data..."):
                df = pd.read_excel(uploaded_file)
                optimizer = EnhancedShiftOptimizer(df)
                df_clean = optimizer.clean_data()
            
            st.success(f"✅ Data loaded: {len(df_clean)} records")
            
            # Check for required columns
            if 'Confirmed' not in df_clean.columns:
                st.error("❌ 'Confirmed' column not found. Please ensure your data has a 'Confirmed' timestamp column.")
                return
            
            # Sidebar - Shift Selection
            st.sidebar.header("🕐 Shift Selection")
            
            shift_options = {
                'Morning': '☀️ Morning (06:00-14:00)',
                'Afternoon': '🌅 Afternoon (14:00-22:00)', 
                'Night': '🌙 Night (22:00-06:00)'
            }
            
            selected_shift = st.sidebar.selectbox(
                "Select Shift to Analyze",
                options=list(shift_options.keys()),
                format_func=lambda x: shift_options[x],
                help="Tasks will be filtered based on their 'Confirmed' timestamp"
            )
            
            # Additional filters
            st.sidebar.header("🔍 Additional Filters")
            
            # Get all queues for filtering
            all_queues = [q for q in df_clean['Queue'].unique() if pd.notna(q)]
            selected_queues = st.sidebar.multiselect(
                "Select Queues",
                options=all_queues,
                default=all_queues,
                help="Choose specific queues to analyze"
            )
            
            min_tasks_filter = st.sidebar.slider(
                "Minimum Tasks per Operator",
                min_value=1,
                max_value=50,
                value=1,
                help="Filter out operators with very few tasks"
            )
            
            # Main Analysis
            if st.sidebar.button("🚀 Analyze Selected Shift", type="primary"):
                with st.spinner(f"Analyzing {selected_shift} shift..."):
                    
                    # Filter data by selected shift
                    shift_data = optimizer.filter_by_shift(selected_shift)
                    
                    if shift_data.empty:
                        st.warning(f"⚠️ No data found for {selected_shift} shift. Check your 'Confirmed' timestamp column.")
                        return
                    
                    # Apply queue filters
                    if selected_queues:
                        shift_data = shift_data[shift_data['Queue'].isin(selected_queues)]
                    
                    if shift_data.empty:
                        st.warning("⚠️ No data matches your filters.")
                        return
                    
                    # Calculate all metrics
                    queue_analysis = optimizer.calculate_queue_productivity(shift_data)
                    multi_queue_analysis = optimizer.analyze_multi_queue_operators(shift_data)
                    utilization_analysis = optimizer.analyze_shift_utilization(shift_data, selected_shift)
                    
                    if not queue_analysis:
                        st.warning("⚠️ No queue analysis possible with current filters.")
                        return
                    
                    # Generate recommendations
                    recommendations = optimizer.generate_optimization_recommendations(
                        queue_analysis, multi_queue_analysis, utilization_analysis, selected_shift
                    )
                
                # Display Results
                shift_class = f"shift-{selected_shift.lower()}"
                shift_icon = optimizer.shift_definitions[selected_shift]['icon']
                shift_label = optimizer.shift_definitions[selected_shift]['label']
                
                st.markdown(f'<div class="{shift_class}">', unsafe_allow_html=True)
                st.markdown(f"## {shift_icon} {selected_shift} Shift Analysis ({shift_label})")
                st.markdown(f"**Analysis Period**: Tasks confirmed between {shift_label}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Enhanced Key Metrics
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("Total Operators", recommendations['shift_summary']['total_operators'])
                with col2:
                    st.metric("Multi-Queue Operators", 
                             recommendations['shift_summary']['multi_queue_operators'],
                             f"{recommendations['shift_summary']['multi_queue_percentage']:.1f}%")
                with col3:
                    st.metric("Total Tasks", recommendations['shift_summary']['total_tasks'])
                with col4:
                    avg_tasks = recommendations['shift_summary']['total_tasks'] / max(recommendations['shift_summary']['total_operators'], 1)
                    st.metric("Avg Tasks/Operator", f"{avg_tasks:.1f}")
                with col5:
                    # Calculate average utilization
                    if utilization_analysis:
                        avg_utilization = np.mean([analysis['active_time_utilization'] for analysis in utilization_analysis.values()])
                        st.metric("Avg Utilization", f"{avg_utilization:.1f}%")
                    else:
                        st.metric("Avg Utilization", "N/A")
                with col6:
                    # Calculate average idle time
                    if utilization_analysis:
                        avg_idle = np.mean([analysis['idle_hours'] for analysis in utilization_analysis.values()])
                        st.metric("Avg Idle Time", f"{avg_idle:.1f}h")
                    else:
                        st.metric("Avg Idle Time", "N/A")

                # NEW: Multi-Queue Operator Analysis
                st.header("🔄 Multi-Queue Operator Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="multi-queue-card">', unsafe_allow_html=True)
                    st.markdown("### Queue Assignment Distribution")
                    st.metric("Single-Queue Operators", multi_queue_analysis['summary']['single_queue_count'])
                    st.metric("Multi-Queue Operators", multi_queue_analysis['summary']['multi_queue_count'])
                    st.metric("Multi-Queue Percentage", f"{multi_queue_analysis['summary']['multi_queue_percentage']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    if multi_queue_analysis['multi_queue_analysis']:
                        st.markdown("### Top 5 Most Flexible Operators")
                        flexible_ops = sorted(multi_queue_analysis['multi_queue_analysis'].items(), 
                                            key=lambda x: x[1]['queue_flexibility_score'], reverse=True)[:5]
                        
                        flexibility_data = []
                        for op, analysis in flexible_ops:
                            flexibility_data.append({
                                'Operator': op,
                                'Queues': analysis['queue_count'], 
                                'Switches': analysis['queue_switches'],
                                'Tasks': analysis['total_tasks'],
                                'Flex Score': f"{analysis['queue_flexibility_score']:.2f}"
                            })
                        
                        if flexibility_data:
                            flex_df = pd.DataFrame(flexibility_data)
                            st.dataframe(flex_df, use_container_width=True)
                
                # Detailed Multi-Queue Analysis
                if multi_queue_analysis['multi_queue_analysis']:
                    st.subheader("📊 Detailed Multi-Queue Operator Analysis")
                    
                    # Create detailed table
                    detailed_multi_queue = []
                    for op, analysis in multi_queue_analysis['multi_queue_analysis'].items():
                        queue_str = " + ".join([f"{q}({analysis['tasks_per_queue'][q]})" for q in analysis['queues']])
                        detailed_multi_queue.append({
                            'Operator': op,
                            'Queues (Tasks)': queue_str,
                            'Total Tasks': analysis['total_tasks'],
                            'Queue Switches': analysis['queue_switches'],
                            'Switches/Hour': f"{analysis['switches_per_hour']:.2f}",
                            'Primary Queue': analysis['primary_queue'],
                            'Primary %': f"{(analysis['primary_tasks']/analysis['total_tasks']*100):.1f}%",
                            'Flexibility Score': f"{analysis['queue_flexibility_score']:.3f}"
                        })
                    
                    multi_queue_df = pd.DataFrame(detailed_multi_queue)
                    st.dataframe(multi_queue_df, use_container_width=True)

                # NEW: Shift Utilization Analysis
                st.header("⏰ Shift Utilization Analysis")
                
                if utilization_analysis:
                    # Utilization summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    all_utilizations = [analysis['active_time_utilization'] for analysis in utilization_analysis.values()]
                    all_idle_times = [analysis['idle_hours'] for analysis in utilization_analysis.values()]
                    
                    with col1:
                        st.markdown('<div class="utilization-card">', unsafe_allow_html=True)
                        avg_util = np.mean(all_utilizations)
                        st.metric("Average Utilization", f"{avg_util:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        low_util_count = len([u for u in all_utilizations if u < 50])
                        st.metric("Low Utilization (<50%)", low_util_count)
                    
                    with col3:
                        high_idle_count = len([i for i in all_idle_times if i > 2])
                        st.metric("High Idle Time (>2h)", high_idle_count)
                    
                    with col4:
                        avg_idle = np.mean(all_idle_times)
                        st.metric("Average Idle Time", f"{avg_idle:.1f}h")
                    
                    # Detailed utilization table
                    st.subheader("👥 Individual Operator Utilization")
                    
                    util_table_data = []
                    for op, analysis in utilization_analysis.items():
                        util_table_data.append({
                            'Operator': op,
                            'Working Hours': f"{analysis['working_hours']:.2f}h",
                            'Active Hours': f"{analysis['active_hours']:.2f}h", 
                            'Idle Hours': f"{analysis['idle_hours']:.2f}h",
                            'Utilization %': f"{analysis['active_time_utilization']:.1f}%",
                            'Tasks/Active Hour': f"{analysis['tasks_per_active_hour']:.2f}",
                            'Idle Periods': analysis['idle_periods_count'],
                            'Late Start': f"{analysis['late_start_hours']:.1f}h",
                            'Early End': f"{analysis['early_end_hours']:.1f}h"
                        })
                    
                    util_df = pd.DataFrame(util_table_data)
                    st.dataframe(util_df, use_container_width=True)
                    
                    # Utilization insights
                    st.subheader("💡 Utilization Insights")
                    
                    # Low utilization operators
                    low_util_ops = [op for op, analysis in utilization_analysis.items() 
                                   if analysis['active_time_utilization'] < 60]
                    
                    if low_util_ops:
                        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                        st.markdown(f"**⚠️ {len(low_util_ops)} operators with <60% utilization:**")
                        for op in low_util_ops[:5]:  # Show top 5
                            util_pct = utilization_analysis[op]['active_time_utilization']
                            idle_hrs = utilization_analysis[op]['idle_hours']
                            st.markdown(f"• {op}: {util_pct:.1f}% utilization, {idle_hrs:.1f}h idle")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # High performers
                    high_util_ops = [op for op, analysis in utilization_analysis.items() 
                                   if analysis['active_time_utilization'] > 80]
                    
                    if high_util_ops:
                        st.markdown('<div class="optimization-card">', unsafe_allow_html=True)
                        st.markdown(f"**🏆 {len(high_util_ops)} high-utilization operators (>80%):**")
                        for op in high_util_ops[:5]:  # Show top 5
                            util_pct = utilization_analysis[op]['active_time_utilization']
                            task_rate = utilization_analysis[op]['tasks_per_active_hour']
                            st.markdown(f"• {op}: {util_pct:.1f}% utilization, {task_rate:.2f} tasks/hour")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Original Queue Analysis (enhanced)
                st.header("📊 Queue Performance Analysis")
                
                for queue, analysis in queue_analysis.items():
                    with st.expander(f"📈 {queue} Queue Analysis", expanded=True):
                        
                        # Enhanced queue metrics
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        with col1:
                            st.metric("Operators", analysis['total_operators'])
                        with col2:
                            st.metric("Total Tasks", analysis['total_tasks'])
                        with col3:
                            st.metric("Median Tasks", f"{analysis['median']:.1f}")
                        with col4:
                            st.metric("Range", f"{analysis['min']}-{analysis['max']}")
                        with col5:
                            st.metric("Avg Active Hours", f"{analysis['avg_active_hours']:.1f}h")
                        with col6:
                            st.metric("Avg Tasks/Hour", f"{analysis['avg_productivity_rate']:.1f}")
                        
                        # Multi-queue operators in this queue
                        if multi_queue_analysis['multi_queue_analysis']:
                            queue_multi_ops = [op for op, mq_analysis in multi_queue_analysis['multi_queue_analysis'].items() 
                                             if queue in mq_analysis['queues']]
                            
                            if queue_multi_ops:
                                st.markdown(f"**🔄 Multi-queue operators in {queue}:** {len(queue_multi_ops)} operators")
                                multi_queue_info = []
                                for op in queue_multi_ops[:5]:  # Show top 5
                                    mq_data = multi_queue_analysis['multi_queue_analysis'][op]
                                    other_queues = [q for q in mq_data['queues'] if q != queue]
                                    multi_queue_info.append(f"{op} (also: {', '.join(other_queues)})")
                                st.caption(" • ".join(multi_queue_info))
                        
                        # Rest of the existing queue analysis...
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🔻 Below Median Operators**")
                            if len(analysis['below_median']) > 0:
                                below_df = analysis['below_median'].copy()
                                below_df['Gap_to_Median'] = analysis['median'] - below_df['task_count']
                                display_below = below_df[['User', 'task_count', 'active_hours', 'tasks_per_hour', 'Gap_to_Median']]
                                display_below.columns = ['Operator', 'Tasks', 'Active Hours', 'Tasks/Hour', 'Gap to Median']
                                st.dataframe(display_below, use_container_width=True)
                            else:
                                st.success("✅ All operators at or above median!")
                        
                        with col2:
                            st.markdown("**🔺 Above Median Operators**")
                            if len(analysis['above_median']) > 0:
                                above_df = analysis['above_median'].copy()
                                above_df['Above_Median_By'] = above_df['task_count'] - analysis['median']
                                display_above = above_df[['User', 'task_count', 'active_hours', 'tasks_per_hour', 'Above_Median_By']]
                                display_above.columns = ['Operator', 'Tasks', 'Active Hours', 'Tasks/Hour', 'Above Median By']
                                st.dataframe(display_above, use_container_width=True)
                            else:
                                st.info("No operators above median")

                # Enhanced Optimization Recommendations
                st.header("🎯 Enhanced Optimization Recommendations")
                
                if recommendations['multi_queue_insights']:
                    st.markdown("### 🔄 Multi-Queue Insights")
                    for insight in recommendations['multi_queue_insights']:
                        st.markdown(f'<div class="multi-queue-card">🔄 {insight}</div>', unsafe_allow_html=True)
                
                if recommendations['utilization_insights']:
                    st.markdown("### ⏰ Utilization Insights")
                    for insight in recommendations['utilization_insights']:
                        st.markdown(f'<div class="utilization-card">⏰ {insight}</div>', unsafe_allow_html=True)
                
                if recommendations['critical_issues']:
                    st.markdown("### ⚠️ Critical Issues")
                    for issue in recommendations['critical_issues']:
                        st.markdown(f'<div class="danger-card">❌ {issue}</div>', unsafe_allow_html=True)
                
                if recommendations['opportunities']:
                    st.markdown("### 📈 Improvement Opportunities")
                    for opportunity in recommendations['opportunities']:
                        st.markdown(f'<div class="optimization-card">💡 {opportunity}</div>', unsafe_allow_html=True)
                
                if recommendations['action_items']:
                    st.markdown("### 🎯 Recommended Actions")
                    for i, action in enumerate(recommendations['action_items'], 1):
                        st.markdown(f"**{i}.** {action}")
                
                # Enhanced Visualizations
                st.header("📊 Advanced Analytics Dashboard")
                
                # Multi-queue visualization
                if multi_queue_analysis['multi_queue_analysis']:
                    st.subheader("🔄 Multi-Queue Operator Patterns")
                    multi_queue_viz = optimizer.create_multi_queue_visualization(multi_queue_analysis)
                    if multi_queue_viz:
                        st.plotly_chart(multi_queue_viz, use_container_width=True)
                
                # Utilization visualization
                if utilization_analysis:
                    st.subheader("⏰ Shift Utilization Dashboard")
                    utilization_viz = optimizer.create_utilization_visualization(utilization_analysis, selected_shift)
                    if utilization_viz:
                        st.plotly_chart(utilization_viz, use_container_width=True)
                
                # Enhanced Export
                st.header("💾 Enhanced Export Results")
                
                # Prepare comprehensive export data
                export_data = []
                for operator in shift_data['User'].unique():
                    # Basic operator data
                    operator_queues = shift_data[shift_data['User'] == operator]['Queue'].unique()
                    operator_tasks = len(shift_data[shift_data['User'] == operator])
                    
                    # Multi-queue data
                    is_multi_queue = len(operator_queues) > 1
                    multi_queue_data = multi_queue_analysis['multi_queue_analysis'].get(operator, {})
                    
                    # Utilization data
                    util_data = utilization_analysis.get(operator, {})
                    
                    # Queue performance data
                    primary_queue = multi_queue_data.get('primary_queue') or operator_queues[0]
                    queue_perf = queue_analysis.get(primary_queue, {})
                    
                    export_data.append({
                        'Shift': selected_shift,
                        'Operator': operator,
                        'Total_Tasks': operator_tasks,
                        'Primary_Queue': primary_queue,
                        'All_Queues': ' + '.join(operator_queues),
                        'Is_Multi_Queue': is_multi_queue,
                        'Queue_Count': len(operator_queues),
                        'Queue_Switches': multi_queue_data.get('queue_switches', 0),
                        'Switches_Per_Hour': multi_queue_data.get('switches_per_hour', 0),
                        'Flexibility_Score': multi_queue_data.get('queue_flexibility_score', 0),
                        'Working_Hours': util_data.get('working_hours', 0),
                        'Active_Hours': util_data.get('active_hours', 0), 
                        'Idle_Hours': util_data.get('idle_hours', 0),
                        'Active_Time_Utilization': util_data.get('active_time_utilization', 0),
                        'Tasks_Per_Active_Hour': util_data.get('tasks_per_active_hour', 0),
                        'Idle_Periods_Count': util_data.get('idle_periods_count', 0),
                        'Late_Start_Hours': util_data.get('late_start_hours', 0),
                        'Early_End_Hours': util_data.get('early_end_hours', 0),
                        'Queue_Median_Tasks': queue_perf.get('median', 0),
                        'Performance_vs_Median': 'Above' if operator_tasks > queue_perf.get('median', 0) else 'Below'
                    })
                
                if export_data:
                    enhanced_export_df = pd.DataFrame(export_data)
                    csv = enhanced_export_df.to_csv(index=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label=f"📥 Download Enhanced {selected_shift} Analysis (CSV)",
                            data=csv,
                            file_name=f"enhanced_warehouse_{selected_shift.lower()}_analysis.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        st.markdown("**📊 Enhanced Export Preview:**")
                        preview_cols = ['Operator', 'Total_Tasks', 'Is_Multi_Queue', 'Queue_Count', 
                                      'Active_Time_Utilization', 'Tasks_Per_Active_Hour']
                        preview_df = enhanced_export_df[preview_cols].head()
                        st.dataframe(preview_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your Excel file contains 'User', 'Queue', and 'Confirmed' columns.")
    
    else:
        # Enhanced welcome screen
        st.info("👋 Welcome to Enhanced Warehouse Shift Optimizer! Upload your operations data to analyze multi-queue patterns and shift utilization.")
        
        st.subheader("🆕 New Features Added")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔄 Multi-Queue Analysis:**
            - Identify operators working across multiple queues
            - Track queue switching frequency and patterns
            - Calculate flexibility scores for cross-training insights
            - Analyze primary vs secondary queue focus
            """)
        
        with col2:
            st.markdown("""
            **⏰ Shift Utilization Analysis:**
            - Calculate actual working hours vs shift duration  
            - Identify idle time periods between tasks
            - Track late starts and early departures
            - Measure active time utilization percentage
            """)
        
        st.subheader("📋 Required Data Format")
        sample_data = {
            'User': ['54001', '54002', '54003', '54004', '54005'],
            'Queue': ['FLRP', 'FL1P', 'VC1P', 'FLRP', 'VC1P'],
            'Task': ['RP', 'SOCS', 'SOCS', 'RP', 'SOCS'],
            'Quantity': [36, 63, 45, 42, 58],
            'Confirmed': ['2024-01-15 08:30:00', '2024-01-15 09:15:00', '2024-01-15 16:45:00', 
                         '2024-01-15 23:20:00', '2024-01-15 17:30:00']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        st.markdown("""
        **Enhanced Analytics You'll Get:**
        
        **🔄 Multi-Queue Insights:**
        - **Queue Switching Patterns**: How often operators switch between queues
        - **Flexibility Scoring**: Operators who can handle multiple queue types effectively
        - **Cross-Training Opportunities**: Identify who could be trained on additional queues
        - **Primary vs Secondary Focus**: Understanding workload distribution across queues
        
        **⏰ Utilization Insights:**
        - **Active vs Idle Time**: Actual working time vs gaps between tasks
        - **Shift Coverage**: Late starts, early departures, and coverage gaps  
        - **Efficiency Patterns**: Task completion rate during active periods
        - **Resource Optimization**: Identify underutilized or overworked operators
        
        **📊 Advanced Visualizations:**
        - Multi-queue operator switching patterns
        - Utilization distribution across the shift
        - Idle time vs productivity correlations
        - Shift coverage timeline analysis
        """)


if __name__ == "__main__":
    main()
