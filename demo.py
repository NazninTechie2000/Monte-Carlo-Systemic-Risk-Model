import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

st.set_page_config(
    page_title="Monte Carlo Systemic Risk Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Monte Carlo Systemic Risk Model</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Traditional vs Blockchain Banking Systems</div>', unsafe_allow_html=True)

@st.cache_data
def load_banking_data_from_file():
    banking_data = pd.read_csv('banks_data.csv')
    if 'Capital Buffer (€B)' not in banking_data.columns:
        banking_data['Capital Buffer (€B)'] = banking_data['CET1 Ratio (%)'] * banking_data['Total Assets (€B)'] * 0.01
    return banking_data

def create_bank_connection_matrix(bank_assets_list, bank_liabilities_list):
    number_of_banks = len(bank_assets_list)
    connection_matrix = np.zeros((number_of_banks, number_of_banks))
    total_system_assets = np.sum(bank_assets_list)
    for bank_i in range(number_of_banks):
        for bank_j in range(number_of_banks):
            if bank_i != bank_j:
                connection_matrix[bank_i, bank_j] = bank_liabilities_list[bank_i] * bank_assets_list[bank_j] / (total_system_assets - bank_assets_list[bank_i])
    return connection_matrix

def run_monte_carlo_banking_simulation(banking_data, bank_connection_matrix, loss_when_bank_fails, initial_crisis_probability, number_of_simulations=1000, banking_system_type="Traditional"):
    total_number_of_banks = len(banking_data)
    simulation_results_list = []
    
    if banking_system_type == "Traditional":
        np.random.seed(42)
    else:
        np.random.seed(44)
    
    adjusted_crisis_probability = initial_crisis_probability
    if banking_system_type == "Blockchain":
        adjusted_crisis_probability = initial_crisis_probability * 0.8
    
    progress_indicator = st.progress(0)
    status_display = st.empty()
    
    for current_simulation in range(number_of_simulations):
        if current_simulation % max(1, number_of_simulations // 20) == 0:
            progress_indicator.progress(current_simulation / number_of_simulations)
            status_display.text(f'Running {banking_system_type} simulation: {current_simulation}/{number_of_simulations}')
        
        banks_failed_initially = np.random.rand(total_number_of_banks) < adjusted_crisis_probability
        bank_capital_remaining = banking_data['Capital Buffer (€B)'].copy()
        
        if banking_system_type == "Blockchain":
            bank_capital_remaining = bank_capital_remaining * 1.1
        
        contagion_round_number = 1
        crisis_still_spreading = True
        
        while crisis_still_spreading and contagion_round_number <= 10:
            losses_this_round = np.zeros(total_number_of_banks)
            
            for bank_index, has_this_bank_failed in enumerate(banks_failed_initially):
                if has_this_bank_failed:
                    losses_caused_by_this_bank = bank_connection_matrix[bank_index] * loss_when_bank_fails
                    if banking_system_type == "Blockchain":
                        losses_caused_by_this_bank *= 0.6
                    losses_this_round += losses_caused_by_this_bank
            
            if banking_system_type == "Traditional" and contagion_round_number > 1:
                market_panic_multiplier = 1.0 + (contagion_round_number * 0.1)
                losses_this_round = losses_this_round * market_panic_multiplier
            
            newly_failed_banks = (losses_this_round > bank_capital_remaining.values) & (~banks_failed_initially)
            crisis_still_spreading = np.any(newly_failed_banks)
            banks_failed_initially = banks_failed_initially | newly_failed_banks
            bank_capital_remaining = bank_capital_remaining - losses_this_round
            contagion_round_number += 1
        
        total_banks_failed = np.sum(banks_failed_initially)
        systemic_crisis_threshold = 3
        is_systemic_crisis = total_banks_failed >= systemic_crisis_threshold
        list_of_failed_banks = np.where(banks_failed_initially)[0].tolist()
        simulation_results_list.append((total_banks_failed, is_systemic_crisis, list_of_failed_banks))
    
    progress_indicator.progress(1.0)
    status_display.text(f'{banking_system_type} simulation completed!')
    time.sleep(0.5)
    progress_indicator.empty()
    status_display.empty()
    return simulation_results_list

def calculate_summary_statistics(simulation_results_list):
    failures_per_simulation = [result[0] for result in simulation_results_list]
    systemic_crisis_occurred = [result[1] for result in simulation_results_list]
    return {
        'Average Failures': np.mean(failures_per_simulation),
        'Max Failures': np.max(failures_per_simulation),
        'Std Dev Failures': np.std(failures_per_simulation),
        'Probability Systemic Event': np.mean(systemic_crisis_occurred),
        'Raw Failures': failures_per_simulation
    }

st.sidebar.header("Bank Simulation Parameters")
banking_data = load_banking_data_from_file()
st.sidebar.success(f"✅ Loaded {len(banking_data)} banks from banks_data.csv")
st.sidebar.subheader("Simulation Settings")

initial_shock_percentage = st.sidebar.slider(
    "Initial Shock Probability (%)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
    help="Probability that a bank experiences an initial shock"
) / 100

number_of_simulations_to_run = st.sidebar.selectbox(
    "Number of Simulations",
    options=[500, 1000, 2000, 5000, 10000],
    index=1,
    help="More simulations = more accurate results but longer runtime"
)

st.sidebar.subheader("Model Parameters")
column1, column2 = st.sidebar.columns(2)

with column1:
    traditional_loss_rate = st.number_input("Traditional LGD", value=0.6, min_value=0.1, max_value=1.0, step=0.05)

with column2:
    blockchain_loss_rate = st.number_input("Blockchain LGD", value=0.3, min_value=0.1, max_value=1.0, step=0.05)

results_tab, data_overview_tab = st.tabs(["📈 Results", "📊 Data Overview"])

with data_overview_tab:
    st.subheader("Bank Data Overview")
    st.dataframe(banking_data, use_container_width=True)
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Number of Banks", len(banking_data))
    with metric_col2:
        st.metric("Avg Total Assets (€B)", f"{banking_data['Total Assets (€B)'].mean():.1f}")
    with metric_col3:
        st.metric("Avg CET1 Ratio (%)", f"{banking_data['CET1 Ratio (%)'].mean():.1f}")

with results_tab:
    st.subheader("Simulation Results")
    
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Building exposure matrices..."):
            traditional_connection_matrix = create_bank_connection_matrix(
                banking_data['Interbank Assets (€B)'], 
                banking_data['Interbank Liabilities (€B)']
            )
            blockchain_connection_matrix = traditional_connection_matrix * 0.5
        
        simulation_col1, simulation_col2 = st.columns(2)
        
        with simulation_col1:
            st.info("🏛️ Running Traditional Banking Simulation...")
            traditional_simulation_results = run_monte_carlo_banking_simulation(
                banking_data, traditional_connection_matrix, traditional_loss_rate, 
                initial_shock_percentage, number_of_simulations_to_run, "Traditional"
            )
            
        with simulation_col2:
            st.info("⛓️ Running Blockchain Banking Simulation...")
            blockchain_simulation_results = run_monte_carlo_banking_simulation(
                banking_data, blockchain_connection_matrix, blockchain_loss_rate, 
                initial_shock_percentage, number_of_simulations_to_run, "Blockchain"
            )
        
        traditional_summary_stats = calculate_summary_statistics(traditional_simulation_results)
        blockchain_summary_stats = calculate_summary_statistics(blockchain_simulation_results)
        
        st.subheader("Key Results")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            if traditional_summary_stats['Average Failures'] > 0:
                failure_improvement_percentage = (1 - blockchain_summary_stats['Average Failures'] / traditional_summary_stats['Average Failures']) * 100
            else:
                failure_improvement_percentage = 0
            st.metric(
                "Average Failures",
                f"{blockchain_summary_stats['Average Failures']:.2f}",
                delta=f"-{failure_improvement_percentage:.1f}%",
                delta_color="inverse"
            )
        
        with metric_col2:
            if traditional_summary_stats['Probability Systemic Event'] > 0:
                systemic_improvement_percentage = (1 - blockchain_summary_stats['Probability Systemic Event'] / traditional_summary_stats['Probability Systemic Event']) * 100
            else:
                systemic_improvement_percentage = 0
            st.metric(
                "Systemic Event Probability",
                f"{blockchain_summary_stats['Probability Systemic Event']*100:.2f}%",
                delta=f"-{systemic_improvement_percentage:.1f}%",
                delta_color="inverse"
            )
        
        with metric_col3:
            if traditional_summary_stats['Std Dev Failures'] > 0:
                volatility_improvement_percentage = (1 - blockchain_summary_stats['Std Dev Failures'] / traditional_summary_stats['Std Dev Failures']) * 100
            else:
                volatility_improvement_percentage = 0
            st.metric(
                "Volatility (Std Dev)",
                f"{blockchain_summary_stats['Std Dev Failures']:.2f}",
                delta=f"-{volatility_improvement_percentage:.1f}%",
                delta_color="inverse"
            )
        
        st.subheader("📋 Detailed Comparison")
        
        traditional_failure_counts = [result[0] for result in traditional_simulation_results]
        blockchain_failure_counts = [result[0] for result in blockchain_simulation_results]
        total_traditional_failures = sum(traditional_failure_counts)
        total_blockchain_failures = sum(blockchain_failure_counts)
        
        detailed_comparison_table = pd.DataFrame({
            'Metric': ['Average Failures', 'Absolute Failure Count', 
                      'Standard Deviation', 'Systemic Event Probability (%)'],
            'Traditional': [
                f"{traditional_summary_stats['Average Failures']:.4f}",
                f"{total_traditional_failures:,}",
                f"{traditional_summary_stats['Std Dev Failures']:.4f}",
                f"{traditional_summary_stats['Probability Systemic Event']*100:.2f}%"
            ],
            'Blockchain': [
                f"{blockchain_summary_stats['Average Failures']:.4f}",
                f"{total_blockchain_failures:,}",
                f"{blockchain_summary_stats['Std Dev Failures']:.4f}",
                f"{blockchain_summary_stats['Probability Systemic Event']*100:.2f}%"
            ]
        })
        
        st.dataframe(detailed_comparison_table, use_container_width=True)
        st.subheader("📊 Visualizations")
        
        traditional_failure_counts = traditional_summary_stats['Raw Failures']
        blockchain_failure_counts = blockchain_summary_stats['Raw Failures']
        
        st.subheader("Distribution of Bank Failures")
        
        histogram_figure = go.Figure()
        maximum_failures_observed = int(max(max(traditional_failure_counts), max(blockchain_failure_counts)))
        number_of_histogram_bins = min(maximum_failures_observed + 1, 50)
        
        histogram_figure.add_trace(
            go.Histogram(x=traditional_failure_counts, name='Traditional', opacity=0.7, 
                        nbinsx=number_of_histogram_bins, histnorm='probability')
        )
        histogram_figure.add_trace(
            go.Histogram(x=blockchain_failure_counts, name='Blockchain', opacity=0.7,
                        nbinsx=number_of_histogram_bins, histnorm='probability')
        )
        histogram_figure.update_layout(
            xaxis_title="Number of Failures",
            yaxis_title="Probability",
            height=400,
            showlegend=True
        )
        st.plotly_chart(histogram_figure, use_container_width=True)
        
        st.subheader("Systemic Event Probability")
        
        bar_figure = go.Figure()
        bar_figure.add_trace(
            go.Bar(
                x=['Traditional', 'Blockchain'], 
                y=[traditional_summary_stats['Probability Systemic Event']*100, 
                   blockchain_summary_stats['Probability Systemic Event']*100],
                name='Systemic Event %',
                marker_color=['#ef553b', '#00cc96']
            )
        )
        bar_figure.update_layout(
            xaxis_title="System Type",
            yaxis_title="Probability (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(bar_figure, use_container_width=True)
        
        st.subheader("💾 Download Results")
        results_csv_data = detailed_comparison_table.to_csv(index=False)
        st.download_button(
            label="📄 Download Results as CSV",
            data=results_csv_data,
            file_name=f"simulation_results_{int(time.time())}.csv",
            mime="text/csv"
        )
        st.success("✅ Simulation completed successfully!")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Monte Carlo Systemic Risk Model | Developed for Academic Research"
    "</div>", 
    unsafe_allow_html=True
)