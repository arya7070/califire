import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Initialize session state variables for persistent data storage across user interactions
# This ensures that user settings and calculations are maintained during the session

# Initial risk scores by ZIP code - higher values indicate higher fire risk
if "zip_risk_scores" not in st.session_state:
    st.session_state.zip_risk_scores = {
        "90001": 1.5,  # high risk area
        "90210": 1.2,  # moderate risk area
        "94105": 1.0,  # low risk area (baseline)
        "95630": 1.7,  # very high risk area
        "94536": 1.3,  # moderate-high risk area
    }

# Risk multipliers based on building structure type - different materials have different fire susceptibility
if "structure_risk_factors" not in st.session_state:
    st.session_state.structure_risk_factors = {
        "Wood": 1.6,     # highest risk - most flammable
        "Stucco": 1.2,   # moderate risk
        "Brick": 1.1,    # lower risk - fire resistant
        "Concrete": 1.0, # baseline risk - very fire resistant
        "Steel": 0.9,    # lowest risk - non-combustible
    }

# Base premium rate as percentage of house value - can be adjusted by user
if "base_rate" not in st.session_state:
    st.session_state.base_rate = 0.005  # 0.5% base premium rate

def calculate_premium(house_value, zip_code, structure_type, base_rate):
    """
    Calculate insurance premium based on house value, location risk, and structure type.
    
    Formula: Premium = House Value × Base Rate × ZIP Risk Factor × Structure Risk Factor
    
    Args:
        house_value (float): Total value of the house in dollars
        zip_code (str): ZIP code for location-based risk assessment
        structure_type (str): Type of building structure (Wood, Stucco, etc.)
        base_rate (float): Base premium rate as decimal (e.g., 0.005 = 0.5%)
    
    Returns:
        float: Calculated annual premium in dollars
    """
    # Get risk factors from session state, with fallback defaults if not found
    zip_risk = st.session_state.zip_risk_scores.get(zip_code, 1.4)
    structure_risk = st.session_state.structure_risk_factors.get(structure_type, 1.3)
    
    # Calculate final premium using multiplicative risk model
    premium = house_value * base_rate * zip_risk * structure_risk
    return premium

def run_monte_carlo_simulation(premium, house_value, num_houses, fire_probability, num_simulations=1000):
    """
    Run Monte Carlo simulation to model financial outcomes under uncertainty.
    
    This simulation models the randomness in the number of fires that occur in a given year
    and calculates the resulting financial impact on the insurance company.
    
    Args:
        premium (float): Annual premium per house
        house_value (float): Value of each house (assuming all houses have same value)
        num_houses (int): Total number of houses insured
        fire_probability (float): Annual probability of any single house burning down
        num_simulations (int): Number of simulation runs to perform
    
    Returns:
        pandas.DataFrame: Results containing premium income, losses, and net profit for each simulation
    """
    results = []
    
    # Run multiple simulations to capture range of possible outcomes
    for _ in range(num_simulations):
        # Simulate number of fires using binomial distribution
        # Each house has independent probability of burning down
        num_fires = np.random.binomial(num_houses, fire_probability)
        
        # Calculate financial outcome for this simulation run
        total_premium_income = premium * num_houses  # Total premiums collected
        actual_losses = house_value * num_fires      # Total claims paid out
        net_profit = total_premium_income - actual_losses  # Net result
        
        # Store results for this simulation
        results.append({
            'premium_income': total_premium_income,
            'actual_losses': actual_losses,
            'net_profit': net_profit,
            'num_fires': num_fires
        })
    
    return pd.DataFrame(results)

# ===== STREAMLIT USER INTERFACE =====

st.title("🏠 California Fire Insurance Premium Estimator")
st.write("Estimate annual insurance premium based on fire risk and house details with advanced analytics.")

# Advanced Settings - Collapsible section for editing risk parameters
with st.expander("⚙️ Advanced Settings: Edit Risk Factors"):
    st.subheader("Base Premium Rate")
    # Allow users to adjust the base premium rate with high precision
    new_base_rate = st.number_input("Base premium rate (%)", 
                                   value=st.session_state.base_rate * 100, 
                                   min_value=0.001, max_value=5.0, step=0.001, 
                                   format="%.3f",
                                   key="base_rate_input")
    st.session_state.base_rate = new_base_rate / 100  # Convert percentage back to decimal
    
    st.subheader("ZIP Code Risk Factors")
    # Dynamic input fields for each ZIP code risk factor
    for zip_code in list(st.session_state.zip_risk_scores.keys()):
        new_val = st.number_input(f"Risk for {zip_code}", 
                                 value=st.session_state.zip_risk_scores[zip_code], 
                                 step=0.001, format="%.3f", key=f"zip_{zip_code}")
        st.session_state.zip_risk_scores[zip_code] = new_val

    st.subheader("Structure Susceptibility Factors")
    # Dynamic input fields for each structure type risk factor
    for structure in list(st.session_state.structure_risk_factors.keys()):
        new_val = st.number_input(f"Risk for {structure}", 
                                 value=st.session_state.structure_risk_factors[structure], 
                                 step=0.001, format="%.3f", key=f"struct_{structure}")
        st.session_state.structure_risk_factors[structure] = new_val

# Initialize session state variables for storing calculation results
if "premium" not in st.session_state:
    st.session_state.premium = None
    st.session_state.house_value = None
    st.session_state.zip_code = None
    st.session_state.structure_type = None
    st.session_state.sim_result = None
    st.session_state.monte_carlo_results = None

# Main premium calculation form
with st.form("premium_form"):
    # Input fields for basic house information
    house_value = st.number_input("Enter house value ($):", 
                                 min_value=50000, max_value=10000000, step=5000)
    zip_code = st.selectbox("Select ZIP code:", 
                           options=list(st.session_state.zip_risk_scores.keys()))
    structure_type = st.selectbox("Select house structure type:", 
                                 options=list(st.session_state.structure_risk_factors.keys()))
    submitted = st.form_submit_button("Calculate Premium")

# Process premium calculation when form is submitted
if submitted:
    # Calculate premium using current settings
    premium = calculate_premium(house_value, zip_code, structure_type, st.session_state.base_rate)
    
    # Store results in session state for persistence
    st.session_state.premium = premium
    st.session_state.house_value = house_value
    st.session_state.zip_code = zip_code
    st.session_state.structure_type = structure_type

# Display premium calculation results if available
if st.session_state.premium:
    # Show calculated premium prominently
    st.success(f"Estimated Annual Premium: ${st.session_state.premium:,.2f}")

    # Display detailed breakdown of premium calculation
    st.markdown("### Breakdown")
    st.write(f"**Base rate**: {st.session_state.base_rate*100:.3f}% of home value")
    st.write(f"**ZIP Risk Factor ({st.session_state.zip_code})**: {st.session_state.zip_risk_scores.get(st.session_state.zip_code)}")
    st.write(f"**Structure Risk Factor ({st.session_state.structure_type})**: {st.session_state.structure_risk_factors.get(st.session_state.structure_type)}")

    # Advanced simulation section
    st.markdown("### 📊 Revenue & Risk Simulation")
    with st.form("simulation_form"):
        # Simulation parameters with detailed controls
        num_houses = st.slider("Number of houses insured:", 
                              10, 10000, 100, key="houses_slider")
        fire_probability = st.slider("Estimated annual probability of house burning down:", 
                                    0.000, 0.100, 0.010, step=0.001, format="%.3f", key="prob_slider")
        num_simulations = st.slider("Number of Monte Carlo simulations:", 
                                   100, 5000, 1000, step=100, key="sim_slider")
        simulate = st.form_submit_button("Run Advanced Simulation")

    # Execute simulation when requested
    if simulate:
        # Calculate expected values (deterministic baseline)
        total_premium_income = st.session_state.premium * num_houses
        expected_loss = st.session_state.house_value * fire_probability * num_houses
        net_profit = total_premium_income - expected_loss
        
        # Run Monte Carlo simulation to model uncertainty
        with st.spinner("Running Monte Carlo simulation..."):
            mc_results = run_monte_carlo_simulation(
                st.session_state.premium, 
                st.session_state.house_value, 
                num_houses, 
                fire_probability, 
                num_simulations
            )
        
        # Store simulation results in session state
        st.session_state.sim_result = {
            "total_premium_income": total_premium_income,
            "expected_loss": expected_loss,
            "net_profit": net_profit,
            "num_houses": num_houses,
            "fire_probability": fire_probability
        }
        st.session_state.monte_carlo_results = mc_results

# Display simulation results and analysis if available
if st.session_state.sim_result and st.session_state.monte_carlo_results is not None:
    res = st.session_state.sim_result
    mc_df = st.session_state.monte_carlo_results
    
    # Calculate key statistical measures from Monte Carlo results
    profit_mean = mc_df['net_profit'].mean()
    profit_std = mc_df['net_profit'].std()
    profit_percentiles = mc_df['net_profit'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    
    # Display financial summary with key metrics
    st.markdown("### 💼 Financial Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expected Premium Income", f"${res['total_premium_income']:,.0f}")
        st.metric("Expected Losses", f"${res['expected_loss']:,.0f}")
    
    with col2:
        st.metric("Expected Net Profit", f"${profit_mean:,.0f}")
        st.metric("Profit Standard Deviation", f"${profit_std:,.0f}")
    
    with col3:
        profit_at_risk_5 = profit_percentiles[0.05]
        profit_at_risk_95 = profit_percentiles[0.95]
        st.metric("5th Percentile (Worst Case)", f"${profit_at_risk_5:,.0f}")
        st.metric("95th Percentile (Best Case)", f"${profit_at_risk_95:,.0f}")

    # Interactive chart selection for data visualization
    st.markdown("### 📈 Financial Analysis Charts")
    chart_type = st.selectbox(
        "Select chart type:",
        ["Bar Chart - Basic Comparison", "Histogram - Profit Distribution", 
         "Line Chart - Probability Analysis", "Pie Chart - Income Breakdown"]
    )

    # Generate selected chart type
    if chart_type == "Bar Chart - Basic Comparison":
        # Create bar chart comparing premium income, expected losses, and net profit
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Total Premiums', 'Expected Losses', 'Net Profit']
        values = [res['total_premium_income'], res['expected_loss'], profit_mean]
        std_devs = [0, mc_df['actual_losses'].std(), profit_std]
        
        # Create bars with error bars showing standard deviation
        bars = ax.bar(categories, [v/1_000_000 for v in values], 
                     color=['#4CAF50', '#F44336', '#2196F3'], alpha=0.7)
        ax.errorbar(categories, [v/1_000_000 for v in values], 
                   yerr=[s/1_000_000 for s in std_devs], 
                   fmt='none', color='black', capsize=5)
        
        ax.set_ylabel("Amount ($ Millions)")
        ax.set_title("Annual Insurance Financial Projection with Standard Deviation")
        ax.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${value/1_000_000:.1f}M', ha='center', va='bottom')
        
        st.pyplot(fig)

    elif chart_type == "Histogram - Profit Distribution":
        # Create histogram showing distribution of profit outcomes from Monte Carlo simulation
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(mc_df['net_profit']/1_000_000, bins=50, alpha=0.7, color='#2196F3', edgecolor='black')
        
        # Add vertical lines for key statistics
        ax.axvline(profit_mean/1_000_000, color='red', linestyle='--', 
                  label=f'Mean: ${profit_mean/1_000_000:.1f}M')
        ax.axvline(profit_mean/1_000_000 - profit_std/1_000_000, color='orange', linestyle='--', 
                  label=f'-1 Std Dev: ${(profit_mean-profit_std)/1_000_000:.1f}M')
        ax.axvline(profit_mean/1_000_000 + profit_std/1_000_000, color='orange', linestyle='--', 
                  label=f'+1 Std Dev: ${(profit_mean+profit_std)/1_000_000:.1f}M')
        
        ax.set_xlabel("Net Profit ($ Millions)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Net Profit from Monte Carlo Simulation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    elif chart_type == "Line Chart - Probability Analysis":
        # Create sensitivity analysis showing how profit changes with fire probability
        prob_range = np.linspace(0.001, 0.05, 50)
        expected_profits = []
        profit_stds = []
        
        # Calculate expected profit and standard deviation for different fire probabilities
        for prob in prob_range:
            expected_loss = st.session_state.house_value * prob * res['num_houses']
            exp_profit = res['total_premium_income'] - expected_loss
            # Standard deviation calculation based on binomial distribution
            profit_std_calc = np.sqrt(res['num_houses'] * prob * (1-prob)) * st.session_state.house_value
            expected_profits.append(exp_profit)
            profit_stds.append(profit_std_calc)
        
        # Create two-panel plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Top panel: Expected profit vs fire probability
        ax1.plot(prob_range * 100, [p/1_000_000 for p in expected_profits], 
                color='#2196F3', linewidth=2, label='Expected Profit')
        ax1.fill_between(prob_range * 100, 
                        [(p-s)/1_000_000 for p, s in zip(expected_profits, profit_stds)],
                        [(p+s)/1_000_000 for p, s in zip(expected_profits, profit_stds)],
                        alpha=0.3, color='#2196F3', label='±1 Std Dev')
        ax1.set_xlabel("Fire Probability (%)")
        ax1.set_ylabel("Expected Profit ($ Millions)")
        ax1.set_title("Expected Profit vs Fire Probability")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Bottom panel: Risk (standard deviation) vs fire probability
        ax2.plot(prob_range * 100, [s/1_000_000 for s in profit_stds], 
                color='#FF9800', linewidth=2, label='Profit Standard Deviation')
        ax2.set_xlabel("Fire Probability (%)")
        ax2.set_ylabel("Profit Std Dev ($ Millions)")
        ax2.set_title("Risk (Standard Deviation) vs Fire Probability")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

    elif chart_type == "Pie Chart - Income Breakdown":
        # Create dual pie charts showing income breakdown and risk factor contribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left pie chart: Premium income vs expected losses
        income_data = [res['total_premium_income'], res['expected_loss']]
        income_labels = ['Premium Income', 'Expected Losses']
        colors1 = ['#4CAF50', '#F44336']
        
        wedges1, texts1, autotexts1 = ax1.pie(income_data, labels=income_labels, autopct='%1.1f%%',
                                             colors=colors1, startangle=90)
        ax1.set_title("Premium Income vs Expected Losses")
        
        # Right pie chart: Risk factor contribution breakdown
        zip_risk = st.session_state.zip_risk_scores.get(st.session_state.zip_code)
        struct_risk = st.session_state.structure_risk_factors.get(st.session_state.structure_type)
        base_factor = 1.0
        
        # Calculate relative contribution of each risk factor
        risk_data = [base_factor, zip_risk - base_factor, struct_risk - base_factor]
        risk_labels = ['Base Risk', f'ZIP Risk (+{zip_risk-base_factor:.1f})', f'Structure Risk (+{struct_risk-base_factor:.1f})']
        colors2 = ['#9E9E9E', '#FF5722', '#795548']
        
        wedges2, texts2, autotexts2 = ax2.pie([abs(x) for x in risk_data], labels=risk_labels, autopct='%1.1f%%',
                                             colors=colors2, startangle=90)
        ax2.set_title("Risk Factor Contribution")
        
        plt.tight_layout()
        st.pyplot(fig)

    # Comprehensive risk metrics table
    st.markdown("### 📋 Risk Metrics Summary")
    risk_metrics = pd.DataFrame({
        'Metric': ['Mean Profit', 'Standard Deviation', 'Coefficient of Variation', 
                  'Value at Risk (5%)', 'Expected Shortfall (5%)', 'Probability of Loss'],
        'Value': [
            f"${profit_mean:,.0f}",
            f"${profit_std:,.0f}",
            f"{(profit_std/abs(profit_mean)*100):.1f}%",  # Measure of relative risk
            f"${profit_percentiles[0.05]:,.0f}",  # 5th percentile (worst 5% of outcomes)
            f"${mc_df[mc_df['net_profit'] <= profit_percentiles[0.05]]['net_profit'].mean():,.0f}",  # Average of worst 5%
            f"{(mc_df['net_profit'] < 0).mean()*100:.1f}%"  # Probability of losing money
        ]
    })
    st.table(risk_metrics)
