import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Warehouse Queue Optimizer",
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
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏭 Warehouse Queue Optimization Analyzer")
st.markdown("**Analyze operator productivity and optimize queue performance for maximum efficiency**")

# Sidebar for file upload and controls
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File", 
    type=['xlsx', 'xls'],
    help="Upload your warehouse operations Excel file"
)

class WarehouseAnalyzer:
    def __init__(self, df):
        self.df = df
        self.queue_analysis = {}
        self.optimization_results = {}
        
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
    
    def identify_queues(self):
        """Identify all unique queues in the dataset"""
        queues = self.df['Queue'].unique()
        queues = [q for q in queues if pd.notna(q)]
        return sorted(queues)
    
    def analyze_operators_per_queue(self):
        """Calculate number of operators per queue"""
        queue_operators = {}
        for queue in self.identify_queues():
            operators = self.df[self.df['Queue'] == queue]['User'].nunique()
            queue_operators[queue] = operators
        return queue_operators
    
    def calculate_productivity_metrics(self):
        """Calculate comprehensive productivity metrics for each queue"""
        results = {}
        
        for queue in self.identify_queues():
            queue_data = self.df[self.df['Queue'] == queue]
            
            # Count tasks per operator
            operator_tasks = queue_data.groupby('User').size().reset_index(name='task_count')
            
            # Calculate metrics
            task_counts = operator_tasks['task_count'].values
            
            results[queue] = {
                'total_operators': len(operator_tasks),
                'total_tasks': queue_data.shape[0],
                'task_counts': task_counts,
                'median': np.median(task_counts),
                'mean': np.mean(task_counts),
                'std': np.std(task_counts),
                'min': np.min(task_counts),
                'max': np.max(task_counts),
                'q25': np.percentile(task_counts, 25),
                'q75': np.percentile(task_counts, 75),
                'operator_details': operator_tasks.sort_values('task_count', ascending=False)
            }
            
            # Identify operators below and above median
            median_val = results[queue]['median']
            results[queue]['above_median'] = operator_tasks[
                operator_tasks['task_count'] > median_val
            ].copy()
            results[queue]['below_median'] = operator_tasks[
                operator_tasks['task_count'] < median_val
            ].copy()
            results[queue]['at_median'] = operator_tasks[
                operator_tasks['task_count'] == median_val
            ].copy()
            
            # Calculate performance ratios
            if results[queue]['min'] > 0:
                results[queue]['performance_ratio'] = results[queue]['max'] / results[queue]['min']
            else:
                results[queue]['performance_ratio'] = float('inf')
        
        self.queue_analysis = results
        return results
    
    def generate_optimization_recommendations(self):
        """Generate optimization recommendations for each queue"""
        recommendations = {}
        
        for queue, analysis in self.queue_analysis.items():
            queue_recommendations = {
                'current_performance': {},
                'issues': [],
                'opportunities': [],
                'actions': [],
                'expected_impact': {}
            }
            
            # Current performance summary
            queue_recommendations['current_performance'] = {
                'operators': analysis['total_operators'],
                'median_tasks': analysis['median'],
                'performance_variation': f"{analysis['performance_ratio']:.1f}:1",
                'operators_below_median': len(analysis['below_median']),
                'operators_above_median': len(analysis['above_median'])
            }
            
            # Identify issues
            below_median_pct = len(analysis['below_median']) / analysis['total_operators'] * 100
            performance_ratio = analysis['performance_ratio']
            
            if below_median_pct > 50:
                queue_recommendations['issues'].append(
                    f"High underperformance: {below_median_pct:.0f}% operators below median"
                )
            
            if performance_ratio > 10:
                queue_recommendations['issues'].append(
                    f"Extreme performance variation: {performance_ratio:.1f}:1 ratio"
                )
            
            if analysis['total_operators'] < 3:
                queue_recommendations['issues'].append(
                    f"Potential bottleneck: Only {analysis['total_operators']} operators"
                )
            
            # Calculate potential improvements
            below_median_ops = analysis['below_median']
            if len(below_median_ops) > 0:
                current_below_total = below_median_ops['task_count'].sum()
                potential_improvement = (analysis['median'] * len(below_median_ops)) - current_below_total
                
                queue_recommendations['opportunities'].append(
                    f"Bring {len(below_median_ops)} operators to median: +{potential_improvement:.0f} tasks/day"
                )
                
                queue_recommendations['expected_impact']['productivity_gain'] = (
                    potential_improvement / analysis['total_tasks'] * 100
                )
            
            # Generate specific actions
            if len(below_median_ops) > 0:
                worst_performers = below_median_ops.head(3)['User'].tolist()
                queue_recommendations['actions'].append(
                    f"Priority training for operators: {', '.join(map(str, worst_performers))}"
                )
            
            if len(analysis['above_median']) > 0:
                top_performer = analysis['operator_details'].iloc[0]['User']
                top_performance = analysis['operator_details'].iloc[0]['task_count']
                queue_recommendations['actions'].append(
                    f"Study best practices from top performer {top_performer} ({top_performance} tasks)"
                )
            
            # Cross-training opportunities
            if analysis['total_operators'] < 5:
                queue_recommendations['actions'].append(
                    "Consider cross-training operators from other queues to reduce bottleneck risk"
                )
            
            recommendations[queue] = queue_recommendations
        
        self.optimization_results = recommendations
        return recommendations
    
    def create_productivity_visualization(self):
        """Create comprehensive productivity visualizations"""
        # Prepare data for visualization
        viz_data = []
        for queue, analysis in self.queue_analysis.items():
            for _, row in analysis['operator_details'].iterrows():
                viz_data.append({
                    'Queue': queue,
                    'Operator': row['User'],
                    'Tasks': row['task_count'],
                    'Performance': 'Above Median' if row['task_count'] > analysis['median'] else 
                                 ('Below Median' if row['task_count'] < analysis['median'] else 'At Median')
                })
        
        viz_df = pd.DataFrame(viz_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Task Distribution by Queue', 'Performance Distribution', 
                          'Queue Comparison', 'Productivity Heatmap'),
            specs=[[{"type": "box"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Box plot for task distribution
        for queue in viz_df['Queue'].unique():
            queue_data = viz_df[viz_df['Queue'] == queue]
            fig.add_trace(
                go.Box(y=queue_data['Tasks'], name=queue, boxpoints='all'),
                row=1, col=1
            )
        
        # Bar chart for performance distribution
        perf_summary = viz_df.groupby(['Queue', 'Performance']).size().unstack(fill_value=0)
        for performance in perf_summary.columns:
            fig.add_trace(
                go.Bar(x=perf_summary.index, y=perf_summary[performance], name=performance),
                row=1, col=2
            )
        
        # Scatter plot for queue comparison
        queue_summary = []
        for queue, analysis in self.queue_analysis.items():
            queue_summary.append({
                'Queue': queue,
                'Median_Tasks': analysis['median'],
                'Total_Operators': analysis['total_operators'],
                'Performance_Ratio': analysis['performance_ratio']
            })
        
        queue_df = pd.DataFrame(queue_summary)
        fig.add_trace(
            go.Scatter(
                x=queue_df['Total_Operators'], 
                y=queue_df['Median_Tasks'],
                mode='markers+text',
                text=queue_df['Queue'],
                textposition='top center',
                marker=dict(size=queue_df['Performance_Ratio'], sizemode='diameter', sizeref=2)
            ),
            row=2, col=1
        )
        
        # Heatmap for productivity matrix
        pivot_data = viz_df.pivot_table(values='Tasks', index='Queue', columns='Operator', fill_value=0)
        fig.add_trace(
            go.Heatmap(z=pivot_data.values, x=pivot_data.columns, y=pivot_data.index, colorscale='Viridis'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Warehouse Productivity Analysis Dashboard")
        return fig

def main():
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner("Loading and processing data..."):
                df = pd.read_excel(uploaded_file)
                analyzer = WarehouseAnalyzer(df)
                df_clean = analyzer.clean_data()
            
            # Display basic info
            st.success(f"✅ Data loaded successfully! {len(df_clean)} records processed.")
            
            # Sidebar filters
            st.sidebar.header("🔍 Analysis Controls")
            selected_queues = st.sidebar.multiselect(
                "Select Queues to Analyze",
                options=analyzer.identify_queues(),
                default=analyzer.identify_queues()
            )
            
            min_operators = st.sidebar.slider(
                "Minimum Operators per Queue",
                min_value=1,
                max_value=50,
                value=1
            )
            
            # Main analysis
            if st.sidebar.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Analyzing queue performance..."):
                    # Run analysis
                    productivity_metrics = analyzer.calculate_productivity_metrics()
                    optimization_recommendations = analyzer.generate_optimization_recommendations()
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Queues", len(analyzer.identify_queues()))
                with col2:
                    st.metric("Total Operators", df_clean['User'].nunique())
                with col3:
                    st.metric("Total Tasks", len(df_clean))
                with col4:
                    avg_productivity = df_clean.groupby('User').size().mean()
                    st.metric("Avg Tasks/Operator", f"{avg_productivity:.1f}")
                
                # Queue Analysis
                st.header("🎯 Queue-by-Queue Analysis")
                
                for queue in selected_queues:
                    if queue in productivity_metrics:
                        analysis = productivity_metrics[queue]
                        recommendations = optimization_recommendations[queue]
                        
                        st.subheader(f"Queue: {queue}")
                        
                        # Metrics row
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Operators", analysis['total_operators'])
                        with col2:
                            st.metric("Median Tasks", f"{analysis['median']:.1f}")
                        with col3:
                            st.metric("Task Range", f"{analysis['min']}-{analysis['max']}")
                        with col4:
                            st.metric("Below Median", len(analysis['below_median']))
                        with col5:
                            performance_gain = recommendations['expected_impact'].get('productivity_gain', 0)
                            st.metric("Potential Gain", f"{performance_gain:.1f}%")
                        
                        # Detailed analysis
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🔻 Operators Below Median**")
                            if len(analysis['below_median']) > 0:
                                below_df = analysis['below_median'].sort_values('task_count')
                                st.dataframe(below_df, use_container_width=True)
                            else:
                                st.success("No operators below median!")
                        
                        with col2:
                            st.markdown("**🔺 Operators Above Median**")
                            if len(analysis['above_median']) > 0:
                                above_df = analysis['above_median'].sort_values('task_count', ascending=False)
                                st.dataframe(above_df, use_container_width=True)
                            else:
                                st.info("All operators at or below median")
                        
                        # Optimization recommendations
                        st.markdown("**🎯 Optimization Recommendations**")
                        
                        if recommendations['issues']:
                            for issue in recommendations['issues']:
                                st.markdown(f'<div class="danger-card">❌ {issue}</div>', unsafe_allow_html=True)
                        
                        if recommendations['opportunities']:
                            for opportunity in recommendations['opportunities']:
                                st.markdown(f'<div class="optimization-card">📈 {opportunity}</div>', unsafe_allow_html=True)
                        
                        if recommendations['actions']:
                            st.markdown("**Action Items:**")
                            for i, action in enumerate(recommendations['actions'], 1):
                                st.markdown(f"**{i}.** {action}")
                        
                        st.divider()
                
                # Visualization
                st.header("📈 Productivity Visualizations")
                
                try:
                    viz_fig = analyzer.create_productivity_visualization()
                    st.plotly_chart(viz_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating visualizations: {str(e)}")
                
                # Summary optimization report
                st.header("📋 Overall Optimization Summary")
                
                total_potential_gain = 0
                critical_queues = []
                
                for queue, rec in optimization_recommendations.items():
                    if queue in selected_queues:
                        gain = rec['expected_impact'].get('productivity_gain', 0)
                        total_potential_gain += gain
                        
                        if len(rec['issues']) > 2:
                            critical_queues.append(queue)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<div class="optimization-card">')
                    st.markdown(f"**🎯 Total Potential Productivity Gain: {total_potential_gain:.1f}%**")
                    st.markdown(f"**⚠️ Critical Queues Needing Attention: {len(critical_queues)}**")
                    if critical_queues:
                        st.markdown(f"**Priority queues:** {', '.join(critical_queues)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**🚀 Implementation Priorities:**")
                    st.markdown("1. Address underperformers in critical queues")
                    st.markdown("2. Implement best practices from top performers")
                    st.markdown("3. Cross-train operators for bottleneck queues")
                    st.markdown("4. Monitor and adjust based on performance metrics")
                
                # Download results
                st.header("💾 Export Results")
                
                # Prepare export data
                export_data = []
                for queue, analysis in productivity_metrics.items():
                    if queue in selected_queues:
                        for _, row in analysis['operator_details'].iterrows():
                            export_data.append({
                                'Queue': queue,
                                'Operator': row['User'],
                                'Tasks': row['task_count'],
                                'Queue_Median': analysis['median'],
                                'Performance_vs_Median': 'Above' if row['task_count'] > analysis['median'] else 
                                                       ('Below' if row['task_count'] < analysis['median'] else 'Equal'),
                                'Improvement_Potential': max(0, analysis['median'] - row['task_count'])
                            })
                
                export_df = pd.DataFrame(export_data)
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis Results (CSV)",
                    data=csv,
                    file_name="warehouse_queue_analysis.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your Excel file contains 'User' and 'Queue' columns")
    
    else:
        # Welcome screen
        st.info("👋 Welcome! Please upload your warehouse operations Excel file to begin analysis.")
        
        # Sample data format
        st.subheader("📋 Required Data Format")
        sample_data = {
            'User': ['54001', '54002', '54003', '54001', '54002'],
            'Queue': ['FLRP', 'FLRP', 'VC1P', 'FL1P', 'VC1P'],
            'Task': ['RP', 'RP', 'SOCS', 'SOCS', 'SOCS'],
            'Quantity': [36, 42, 63, 45, 52]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        
        st.markdown("""
        **Required columns:**
        - **User**: Operator ID or identifier
        - **Queue**: Queue name (FLRP, FL1P, VC1P, etc.)
        
        **Optional columns:**
        - **Task**: Task type
        - **Quantity**: Task quantity
        - **From Zone**, **To Zone**: Location information
        """)

if __name__ == "__main__":
    main()
