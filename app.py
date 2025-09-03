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
st.title("🏭 Warehouse Shift Optimizer v2")
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
    
    def calculate_queue_productivity(self, shift_data):
        """Calculate productivity metrics for each queue in the selected shift"""
        if shift_data.empty:
            return {}
        
        results = {}
        queues = shift_data['Queue'].unique()
        
        for queue in queues:
            queue_data = shift_data[shift_data['Queue'] == queue]
            
            # Count tasks per operator
            operator_tasks = queue_data.groupby('User').size().reset_index(name='task_count')
            
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
                'operator_details': operator_tasks.sort_values('task_count', ascending=False)
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
            'action_items': []
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
            
            # Opportunities
            if below_median_count > 0:
                potential_gain = (analysis['median'] * below_median_count) - analysis['below_median']['task_count'].sum()
                recommendations['opportunities'].append(
                    f"{queue}: Bring {below_median_count} operators to median → +{potential_gain:.0f} tasks/day (+{potential_gain/analysis['total_tasks']*100:.1f}%)"
                )
            
            # Action items
            if below_median_count > 0:
                worst_performer = analysis['below_median'].iloc[0]['User'] if len(analysis['below_median']) > 0 else 'N/A'
                recommendations['action_items'].append(
                    f"{queue}: Priority training for operator {worst_performer} and {below_median_count-1} others"
                )
            
            if len(analysis['above_median']) > 0:
                top_performer = analysis['operator_details'].iloc[0]['User']
                top_tasks = analysis['operator_details'].iloc[0]['task_count']
                recommendations['action_items'].append(
                    f"{queue}: Study best practices from top performer {top_performer} ({top_tasks} tasks)"
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
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Operators", recommendations['shift_summary']['total_operators'])
                with col2:
                    st.metric("Total Tasks", recommendations['shift_summary']['total_tasks'])
                with col3:
                    avg_tasks = recommendations['shift_summary']['total_tasks'] / max(recommendations['shift_summary']['total_operators'], 1)
                    st.metric("Avg Tasks/Operator", f"{avg_tasks:.1f}")
                with col4:
                    st.metric("Queues Analyzed", len(queue_analysis))
                
                # Queue Analysis
                st.header("📊 Queue Performance Analysis")
                
                for queue, analysis in queue_analysis.items():
                    with st.expander(f"📈 {queue} Queue Analysis", expanded=True):
                        
                        # Queue metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Operators", analysis['total_operators'])
                        with col2:
                            st.metric("Total Tasks", analysis['total_tasks'])
                        with col3:
                            st.metric("Median Tasks", f"{analysis['median']:.1f}")
                        with col4:
                            st.metric("Range", f"{analysis['min']}-{analysis['max']}")
                        with col5:
                            st.metric("Below Median", len(analysis['below_median']))
                        
                        # Detailed operator analysis
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🔻 Below Median Operators**")
                            if len(analysis['below_median']) > 0:
                                below_df = analysis['below_median'].copy()
                                below_df['Gap to Median'] = analysis['median'] - below_df['task_count']
                                st.dataframe(below_df[['User', 'task_count', 'Gap to Median']], 
                                           use_container_width=True)
                            else:
                                st.success("✅ All operators at or above median!")
                        
                        with col2:
                            st.markdown("**🔺 Above Median Operators**")
                            if len(analysis['above_median']) > 0:
                                above_df = analysis['above_median'].copy()
                                above_df['Above Median By'] = above_df['task_count'] - analysis['median']
                                st.dataframe(above_df[['User', 'task_count', 'Above Median By']], 
                                           use_container_width=True)
                            else:
                                st.info("No operators above median")
                
                # Optimization Recommendations
                st.header("🎯 Optimization Recommendations")
                
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
                
                # Visualizations
                st.header("📊 Performance Dashboard")
                viz_fig = optimizer.create_shift_visualizations(queue_analysis, selected_shift)
                if viz_fig:
                    st.plotly_chart(viz_fig, use_container_width=True)
                
                # Export Data
                st.header("💾 Export Results")
                
                # Prepare export data
                export_data = []
                for queue, analysis in queue_analysis.items():
                    for _, row in analysis['operator_details'].iterrows():
                        export_data.append({
                            'Shift': selected_shift,
                            'Queue': queue,
                            'Operator': row['User'],
                            'Tasks_Completed': row['task_count'],
                            'Queue_Median': analysis['median'],
                            'Performance_Status': 'Above Median' if row['task_count'] > analysis['median'] else 
                                                ('Below Median' if row['task_count'] < analysis['median'] else 'At Median'),
                            'Gap_to_Median': analysis['median'] - row['task_count'],
                            'Improvement_Potential': max(0, analysis['median'] - row['task_count'])
                        })
                
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label=f"📥 Download {selected_shift} Shift Analysis (CSV)",
                        data=csv,
                        file_name=f"warehouse_{selected_shift.lower()}_shift_analysis.csv",
                        mime="text/csv"
                    )
        
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
        
        **How Shift Analysis Works:**
        - **☀️ Morning Shift**: Shows tasks confirmed 06:00-14:00
        - **🌅 Afternoon Shift**: Shows tasks confirmed 14:00-22:00
        - **🌙 Night Shift**: Shows tasks confirmed 22:00-06:00
        
        **What You'll Get:**
        1. **Queue-by-queue productivity analysis** for selected shift
        2. **Operators above/below median performance** identification
        3. **Specific optimization recommendations** and action items
        4. **Interactive visualizations** and performance dashboards
        5. **Exportable results** for implementation planning
        """)

if __name__ == "__main__":
    main()
