import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Warehouse Shift Optimizer v2",
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
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏭 Warehouse Shift Productivity insights - Barath Ashok")
st.markdown("**Select a shift to analyze operator productivity and optimize queue performance**")

class ShiftOptimizer:
    def __init__(self, df):
        self.df = df
        self.shift_definitions = {
            'Morning': {'start': time(6, 0), 'end': time(14, 0), 'icon': '☀️', 'label': '06:00-14:00'},
            'Afternoon': {'start': time(14, 0), 'end': time(22, 0), 'icon': '🌅', 'label': '14:00-22:00'}, 
            'Night': {'start': time(22, 0), 'end': time(6, 0), 'icon': '🌙', 'label': '22:00-06:00'}
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
    
    def generate_optimization_recommendations(self, queue_analysis, selected_shift):
        """Generate optimization recommendations for the selected shift"""
        recommendations = {
            'shift_summary': {},
            'critical_issues': [],
            'opportunities': [],
            'action_items': [],
            'efficiency_insights': []
        }
        
        total_operators = sum(analysis['total_operators'] for analysis in queue_analysis.values())
        total_tasks = sum(analysis['total_tasks'] for analysis in queue_analysis.values())
        
        recommendations['shift_summary'] = {
            'shift': selected_shift,
            'total_operators': total_operators,
            'total_tasks': total_tasks,
            'queues_analyzed': len(queue_analysis)
        }
        
        # Analyze each queue for issues and opportunities
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
            
            # Efficiency insights
            if len(analysis['operator_details']) > 0:
                avg_hours = analysis['avg_active_hours']
                avg_productivity = analysis['avg_productivity_rate']
                
                # Check for efficiency issues
                low_efficiency_ops = analysis['operator_details'][
                    analysis['operator_details']['tasks_per_hour'] < avg_productivity * 0.7
                ]
                
                if len(low_efficiency_ops) > 0:
                    recommendations['efficiency_insights'].append(
                        f"{queue}: {len(low_efficiency_ops)} operators working below 70% of average efficiency ({avg_productivity:.1f} tasks/hour)"
                    )
                
                # Check for operators working long hours but low productivity
                long_low_productivity = analysis['operator_details'][
                    (analysis['operator_details']['active_hours'] > avg_hours * 1.2) & 
                    (analysis['operator_details']['tasks_per_hour'] < avg_productivity * 0.8)
                ]
                
                if len(long_low_productivity) > 0:
                    recommendations['efficiency_insights'].append(
                        f"{queue}: {len(long_low_productivity)} operators working long hours ({avg_hours*1.2:.1f}h+) but low productivity"
                    )
            
            # Opportunities
            if below_median_count > 0:
                potential_gain = (analysis['median'] * below_median_count) - analysis['below_median']['task_count'].sum()
                recommendations['opportunities'].append(
                    f"{queue}: Bring {below_median_count} operators to median → +{potential_gain:.0f} tasks/day (+{potential_gain/analysis['total_tasks']*100:.1f}%)"
                )
                
                # Calculate efficiency improvement potential
                below_median_hours = analysis['below_median']['active_hours'].sum()
                current_below_productivity = analysis['below_median']['tasks_per_hour'].mean()
                target_productivity = analysis['avg_productivity_rate']
                efficiency_gain = (target_productivity - current_below_productivity) * below_median_hours
                
                if efficiency_gain > 0:
                    recommendations['opportunities'].append(
                        f"{queue}: Improve efficiency of {below_median_count} operators to average rate → +{efficiency_gain:.0f} tasks/day from better efficiency"
                    )
            
            # Action items
            if below_median_count > 0:
                worst_performer = analysis['below_median'].iloc[0]['User'] if len(analysis['below_median']) > 0 else 'N/A'
                recommendations['action_items'].append(
                    f"{queue}: Priority training for operator {worst_performer} and {below_median_count-1} others"
                )
            
            if len(analysis['above_median']) > 0:
                # Find the most efficient operator (highest tasks per hour)
                most_efficient = analysis['operator_details'].loc[analysis['operator_details']['tasks_per_hour'].idxmax()]
                recommendations['action_items'].append(
                    f"{queue}: Study efficiency practices from {most_efficient['User']} ({most_efficient['tasks_per_hour']:.1f} tasks/hour)"
                )
        
        return recommendations
    
    def create_shift_visualizations(self, queue_analysis, selected_shift):
        """Create visualizations for the selected shift"""
        if not queue_analysis:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{selected_shift} Shift - Task Distribution by Queue',
                f'{selected_shift} Shift - Performance Distribution',
                f'{selected_shift} Shift - Operator Performance',
                f'{selected_shift} Shift - Queue Comparison'
            ),
            specs=[
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        colors = {'Morning': '#FFD700', 'Afternoon': '#4169E1', 'Night': '#9370DB'}
        color = colors.get(selected_shift, '#1f77b4')
        
        # 1. Box plot for task distribution
        for queue, analysis in queue_analysis.items():
            fig.add_trace(
                go.Box(
                    y=analysis['operator_details']['task_count'],
                    name=queue,
                    marker_color=color,
                    boxpoints='all'
                ),
                row=1, col=1
            )
        
        # 2. Performance distribution (above/below median)
        performance_data = []
        for queue, analysis in queue_analysis.items():
            performance_data.append({
                'Queue': queue,
                'Above Median': len(analysis['above_median']),
                'Below Median': len(analysis['below_median'])
            })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            fig.add_trace(
                go.Bar(x=perf_df['Queue'], y=perf_df['Above Median'], 
                      name='Above Median', marker_color='green'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=perf_df['Queue'], y=perf_df['Below Median'], 
                      name='Below Median', marker_color='red'),
                row=1, col=2
            )
        
        # 3. Operator performance scatter
        scatter_data = []
        for queue, analysis in queue_analysis.items():
            for _, row in analysis['operator_details'].iterrows():
                scatter_data.append({
                    'Queue': queue,
                    'Operator': row['User'],
                    'Tasks': row['task_count'],
                    'vs_Median': 'Above' if row['task_count'] > analysis['median'] else 'Below'
                })
        
        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)
            for performance in ['Above', 'Below']:
                subset = scatter_df[scatter_df['vs_Median'] == performance]
                fig.add_trace(
                    go.Scatter(
                        x=subset['Queue'], 
                        y=subset['Tasks'],
                        mode='markers',
                        name=f'{performance} Median',
                        marker=dict(
                            size=8,
                            color='green' if performance == 'Above' else 'red'
                        ),
                        text=subset['Operator'],
                        textposition='top center'
                    ),
                    row=2, col=1
                )
        
        # 4. Queue comparison metrics
        queue_metrics = []
        for queue, analysis in queue_analysis.items():
            queue_metrics.append({
                'Queue': queue,
                'Median': analysis['median'],
                'Operators': analysis['total_operators']
            })
        
        if queue_metrics:
            metrics_df = pd.DataFrame(queue_metrics)
            fig.add_trace(
                go.Bar(
                    x=metrics_df['Queue'], 
                    y=metrics_df['Median'],
                    name='Median Tasks',
                    marker_color=color,
                    text=metrics_df['Operators'],
                    texttemplate='%{text} ops',
                    textposition='outside'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text=f"{selected_shift} Shift Analysis Dashboard",
            showlegend=True
        )
        
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
                optimizer = ShiftOptimizer(df)
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
                    
                    # Calculate metrics
                    queue_analysis = optimizer.calculate_queue_productivity(shift_data)
                    
                    if not queue_analysis:
                        st.warning("⚠️ No queue analysis possible with current filters.")
                        return
                    
                    # Generate recommendations
                    recommendations = optimizer.generate_optimization_recommendations(queue_analysis, selected_shift)
                
                # Display Results
                shift_class = f"shift-{selected_shift.lower()}"
                shift_icon = optimizer.shift_definitions[selected_shift]['icon']
                shift_label = optimizer.shift_definitions[selected_shift]['label']
                
                st.markdown(f'<div class="{shift_class}">', unsafe_allow_html=True)
                st.markdown(f"## {shift_icon} {selected_shift} Shift Analysis ({shift_label})")
                st.markdown(f"**Analysis Period**: Tasks confirmed between {shift_label}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Key Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Operators", recommendations['shift_summary']['total_operators'])
                with col2:
                    st.metric("Total Tasks", recommendations['shift_summary']['total_tasks'])
                with col3:
                    avg_tasks = recommendations['shift_summary']['total_tasks'] / max(recommendations['shift_summary']['total_operators'], 1)
                    st.metric("Avg Tasks/Operator", f"{avg_tasks:.1f}")
                with col4:
                    # Calculate average active hours across all operators in shift
                    all_active_hours = []
                    for analysis in queue_analysis.values():
                        all_active_hours.extend(analysis['operator_details']['active_hours'].tolist())
                    avg_active_hours = np.mean(all_active_hours) if all_active_hours else 0
                    st.metric("Avg Active Hours", f"{avg_active_hours:.1f}h")
                with col5:
                    # Calculate average productivity rate
                    all_productivity = []
                    for analysis in queue_analysis.values():
                        all_productivity.extend(analysis['operator_details']['tasks_per_hour'].tolist())
                    avg_productivity = np.mean(all_productivity) if all_productivity else 0
                    st.metric("Avg Tasks/Hour", f"{avg_productivity:.1f}")

                # Operator Activity Analysis
                st.header("👥 Operator Activity Analysis")
                
                # Collect all operator activity data
                all_operator_data = []
                for queue, analysis in queue_analysis.items():
                    for _, row in analysis['operator_details'].iterrows():
                        all_operator_data.append({
                            'Operator': row['User'],
                            'Queue': queue,
                            'Total_Tasks': row['task_count'],
                            'Active_Hours': row['active_hours'],
                            'Tasks_Per_Hour': row['tasks_per_hour'],
                            'Performance_vs_Median': 'Above' if row['task_count'] > analysis['median'] else 'Below'
                        })
                
                if all_operator_data:
                    activity_df = pd.DataFrame(all_operator_data)
                    
                    # Top performers by different metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🏆 Top 10 by Tasks per Hour (Efficiency)**")
                        top_efficiency = activity_df.nlargest(10, 'Tasks_Per_Hour')[
                            ['Operator', 'Queue', 'Total_Tasks', 'Active_Hours', 'Tasks_Per_Hour']
                        ]
                        top_efficiency.columns = ['Operator', 'Queue', 'Total Tasks', 'Active Hours', 'Tasks/Hour']
                        st.dataframe(top_efficiency, use_container_width=True)
                    
                    with col2:
                        st.markdown("**⏱️ Top 10 by Total Tasks (Volume)**")
                        top_volume = activity_df.nlargest(10, 'Total_Tasks')[
                            ['Operator', 'Queue', 'Total_Tasks', 'Active_Hours', 'Tasks_Per_Hour']
                        ]
                        top_volume.columns = ['Operator', 'Queue', 'Total Tasks', 'Active Hours', 'Tasks/Hour']
                        st.dataframe(top_volume, use_container_width=True)
                    
                    # Activity insights
                    st.markdown("**📊 Activity Insights**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        max_active_hours = activity_df['Active_Hours'].max()
                        longest_worker = activity_df[activity_df['Active_Hours'] == max_active_hours]['Operator'].iloc[0]
                        st.metric("Longest Active", f"{longest_worker}", f"{max_active_hours:.1f}h")
                    
                    with col2:
                        max_efficiency = activity_df['Tasks_Per_Hour'].max()
                        most_efficient = activity_df[activity_df['Tasks_Per_Hour'] == max_efficiency]['Operator'].iloc[0]
                        st.metric("Most Efficient", f"{most_efficient}", f"{max_efficiency:.1f} t/h")
                    
                    with col3:
                        short_workers = len(activity_df[activity_df['Active_Hours'] < 2])
                        st.metric("Short Sessions", short_workers, "operators < 2h")
                    
                    with col4:
                        long_workers = len(activity_df[activity_df['Active_Hours'] > 6])
                        st.metric("Long Sessions", long_workers, "operators > 6h")
                
                # Queue Analysis
                st.header("📊 Queue Performance Analysis")
                
                for queue, analysis in queue_analysis.items():
                    with st.expander(f"📈 {queue} Queue Analysis", expanded=True):
                        
                        # Queue metrics
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
                        
                        # Detailed operator analysis with active hours
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🔻 Below Median Operators**")
                            if len(analysis['below_median']) > 0:
                                below_df = analysis['below_median'].copy()
                                below_df['Gap_to_Median'] = analysis['median'] - below_df['task_count']
                                display_below = below_df[['User', 'task_count', 'active_hours', 'tasks_per_hour', 'Gap_to_Median']]
                                display_below.columns = ['Operator', 'Tasks', 'Active Hours', 'Tasks/Hour', 'Gap to Median']
                                st.dataframe(display_below, use_container_width=True)
                                
                                # Analysis of below median operators
                                avg_hours_below = below_df['active_hours'].mean()
                                avg_productivity_below = below_df['tasks_per_hour'].mean()
                                st.caption(f"📊 Below median group: {avg_hours_below:.1f}h avg active time, {avg_productivity_below:.1f} tasks/hour avg productivity")
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
                                
                                # Analysis of above median operators
                                avg_hours_above = above_df['active_hours'].mean()
                                avg_productivity_above = above_df['tasks_per_hour'].mean()
                                st.caption(f"📊 Above median group: {avg_hours_above:.1f}h avg active time, {avg_productivity_above:.1f} tasks/hour avg productivity")
                            else:
                                st.info("No operators above median")
                        
                        # Efficiency insights for this queue
                        if len(analysis['operator_details']) > 0:
                            st.markdown("**🎯 Efficiency Insights**")
                            
                            # Find most and least efficient operators
                            most_efficient = analysis['operator_details'].loc[analysis['operator_details']['tasks_per_hour'].idxmax()]
                            least_efficient = analysis['operator_details'].loc[analysis['operator_details']['tasks_per_hour'].idxmin()]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Most Efficient**: {most_efficient['User']}")
                                st.markdown(f"• {most_efficient['tasks_per_hour']:.1f} tasks/hour")
                                st.markdown(f"• {most_efficient['active_hours']:.1f} hours active")
                            
                            with col2:
                                st.markdown(f"**Least Efficient**: {least_efficient['User']}")
                                st.markdown(f"• {least_efficient['tasks_per_hour']:.1f} tasks/hour") 
                                st.markdown(f"• {least_efficient['active_hours']:.1f} hours active")
                            
                            with col3:
                                efficiency_gap = most_efficient['tasks_per_hour'] - least_efficient['tasks_per_hour']
                                st.markdown(f"**Efficiency Gap**: {efficiency_gap:.1f} tasks/hour")
                                potential_improvement = efficiency_gap * least_efficient['active_hours']
                                st.markdown(f"**Improvement Potential**: +{potential_improvement:.0f} tasks if least efficient matched most efficient")
                
                # Optimization Recommendations
                st.header("🎯 Optimization Recommendations")
                
                if recommendations['critical_issues']:
                    st.markdown("### ⚠️ Critical Issues")
                    for issue in recommendations['critical_issues']:
                        st.markdown(f'<div class="danger-card">❌ {issue}</div>', unsafe_allow_html=True)
                
                if recommendations['efficiency_insights']:
                    st.markdown("### ⏱️ Efficiency Insights")
                    for insight in recommendations['efficiency_insights']:
                        st.markdown(f'<div class="warning-card">⚡ {insight}</div>', unsafe_allow_html=True)
                
                if recommendations['opportunities']:
                    st.markdown("### 📈 Improvement Opportunities")
                    for opportunity in recommendations['opportunities']:
                        st.markdown(f'<div class="optimization-card">💡 {opportunity}</div>', unsafe_allow_html=True)
                
                if recommendations['action_items']:
                    st.markdown("### 🎯 Recommended Actions")
                    for i, action in enumerate(recommendations['action_items'], 1):
                        st.markdown(f"**{i}.** {action}")
                
                # Summary insights
                st.markdown("### 💼 Key Takeaways for Shift Management")
                st.markdown("""
                **Focus Areas:**
                - **Volume vs Efficiency**: Some operators work long hours but aren't productive - focus on efficiency training
                - **Best Practice Replication**: Study high-efficiency operators and replicate their methods
                - **Resource Allocation**: Consider redistributing work from low-efficiency to high-efficiency operators
                - **Training Priorities**: Target operators with low tasks/hour rates for immediate improvement
                """)
                
                # Visualizations
                st.header("📊 Performance Dashboard")
                viz_fig = optimizer.create_shift_visualizations(queue_analysis, selected_shift)
                if viz_fig:
                    st.plotly_chart(viz_fig, use_container_width=True)
                
                # Export Data
                st.header("💾 Export Results")
                
                # Prepare export data with active hours
                export_data = []
                for queue, analysis in queue_analysis.items():
                    for _, row in analysis['operator_details'].iterrows():
                        export_data.append({
                            'Shift': selected_shift,
                            'Queue': queue,
                            'Operator': row['User'],
                            'Tasks_Completed': row['task_count'],
                            'Active_Hours': row['active_hours'],
                            'Tasks_Per_Hour': row['tasks_per_hour'],
                            'Queue_Median_Tasks': analysis['median'],
                            'Queue_Avg_Hours': analysis['avg_active_hours'],
                            'Queue_Avg_Productivity': analysis['avg_productivity_rate'],
                            'Performance_Status': 'Above Median' if row['task_count'] > analysis['median'] else 
                                                ('Below Median' if row['task_count'] < analysis['median'] else 'At Median'),
                            'Task_Gap_to_Median': analysis['median'] - row['task_count'],
                            'Productivity_vs_Queue_Avg': row['tasks_per_hour'] - analysis['avg_productivity_rate'],
                            'Efficiency_Category': 'High Efficiency' if row['tasks_per_hour'] > analysis['avg_productivity_rate'] else 'Low Efficiency'
                        })
                
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label=f"📥 Download {selected_shift} Shift Analysis (CSV)",
                            data=csv,
                            file_name=f"warehouse_{selected_shift.lower()}_shift_analysis.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Show preview of export data
                        st.markdown("**📊 Export Data Preview:**")
                        preview_df = export_df[['Operator', 'Queue', 'Tasks_Completed', 'Active_Hours', 'Tasks_Per_Hour', 'Performance_Status']].head()
                        st.dataframe(preview_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your Excel file contains 'User', 'Queue', and 'Confirmed' columns.")
    
    else:
        # Welcome screen
        st.info("👋 Welcome! Upload your warehouse operations Excel file to begin shift analysis.")
        
        # Required format
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
        **Required Columns:**
        - **User**: Operator ID or identifier
        - **Queue**: Queue name (FLRP, FL1P, VC1P, etc.)
        - **Confirmed**: Timestamp when task was confirmed (YYYY-MM-DD HH:MM:SS format)
        
        **How Active Hours are Calculated:**
        - **Active Duration**: Time from first confirmed task to last confirmed task per operator
        - **Productivity Rate**: Total tasks ÷ Active hours = Tasks per hour
        - **Example**: Operator confirms tasks from 08:30 to 12:45 = 4.25 active hours
        
        **Shift Analysis Logic:**
        - **☀️ Morning Shift**: Tasks confirmed 06:00-14:00
        - **🌅 Afternoon Shift**: Tasks confirmed 14:00-22:00
        - **🌙 Night Shift**: Tasks confirmed 22:00-06:00
        
        **What You'll Get:**
        1. **Task volume analysis** per operator in selected shift
        2. **Active hours calculation** based on confirmed task timestamps  
        3. **Productivity rates** (tasks per hour) for efficiency insights
        4. **Above/below median identification** with efficiency analysis
        5. **Optimization recommendations** focusing on efficiency vs volume
        6. **Export data** with active hours and productivity metrics
        
        **Key Insights Provided:**
        - **Volume Leaders**: Operators completing most tasks
        - **Efficiency Leaders**: Operators with highest tasks/hour rates
        - **Long Hours/Low Productivity**: Operators working long but inefficiently
        - **Training Priorities**: Specific operators needing efficiency improvement
        """)


if __name__ == "__main__":
    main()
