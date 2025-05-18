import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import os

def ensure_output_dir():
    """Create static directory for output files"""
    output_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def run_allocation():
    """Run the complete allocation analysis"""
    output_dir = ensure_output_dir()
    results = {}

    # 1.0 Initial Data Input
    supply_cum = {
        'Jan Wk2': 230,
        'Jan Wk3': 270,
        'Jan Wk4': 320,
        'Jan Wk5': 380
    }

    built = {
        'Superman': 70,
        'Superman_Plus': 70,
        'Superman_Mini': 60
    }

    demand_cum = {
        'Superman': [85, 100, 110, 120],
        'Superman_Plus': [85, 120, 150, 175],
        'Superman_Mini': [40, 60, 70, 75]
    }

    Superman_Plus_demand = {
        ('Online Store'): [20, 30, 40, 50],
        ('Retail Store'): [15, 25, 30, 35],
        ('Reseller', 'PAC'): [25, 30, 35, 40],
        ('Reseller', 'AMR'): [20, 25, 30, 35],
        ('Reseller', 'Europe'): [5, 10, 15, 15]
    }

    # Calculate weekly from cumulative
    def cum_to_weekly(cum_list):
        return [cum_list[0]] + [cum_list[i]-cum_list[i-1] for i in range(1,len(cum_list))]

    Superman_Plus_demand_weekly = {k: cum_to_weekly(v) for k,v in Superman_Plus_demand.items()}
    demand_weekly = {k: cum_to_weekly(v) for k,v in demand_cum.items()}

    # Calculate weekly supply
    supply_weekly = {
        'Jan Wk2': supply_cum['Jan Wk2'] - sum(built.values()),
        'Jan Wk3': supply_cum['Jan Wk3'] - supply_cum['Jan Wk2'],
        'Jan Wk4': supply_cum['Jan Wk4'] - supply_cum['Jan Wk3'],
        'Jan Wk5': supply_cum['Jan Wk5'] - supply_cum['Jan Wk4']
    }

    # 1.1 Business Rule: Priority
    alloc_priority = {'Superman': [], 'Superman_Mini': []}
    remaining_capacity = {}

    for week_idx in range(4):
        wk = f'Jan Wk{week_idx+2}'
        superman_needed = max(0, demand_cum['Superman'][week_idx] - built['Superman'])
        superman_alloc = min(superman_needed, supply_weekly[wk])
        alloc_priority['Superman'].append(superman_alloc)
        built['Superman'] += superman_alloc

        mini_needed = max(0, demand_cum['Superman_Mini'][week_idx] - built['Superman_Mini'])
        mini_alloc = min(mini_needed, supply_weekly[wk] - superman_alloc)
        alloc_priority['Superman_Mini'].append(mini_alloc)
        built['Superman_Mini'] += mini_alloc

        remaining_capacity[wk] = supply_weekly[wk] - superman_alloc - mini_alloc

    # 1.2 Optimization Model
    model = LpProblem("Superman_Plus_Allocation", LpMaximize)
    weeks = ['Jan Wk2', 'Jan Wk3', 'Jan Wk4', 'Jan Wk5']
    channels = list(Superman_Plus_demand_weekly.keys())
    alloc_vars = LpVariable.dicts("Alloc", (channels, weeks), lowBound=0, cat='Integer')

    time_decay = [1.2**i for i in range(4)]
    model += lpSum([alloc_vars[c][w] * time_decay[weeks.index(w)] 
                   for c in channels for w in weeks])

    for w in weeks:
        model += lpSum(alloc_vars[c][w] for c in channels) <= remaining_capacity[w]
        if w == 'Jan Wk4':
            pac_reseller = [c for c in channels if isinstance(c, tuple) and c[1]=='PAC'][0]
            model += alloc_vars[pac_reseller][w] >= Superman_Plus_demand_weekly[pac_reseller][weeks.index(w)]

    for c in channels:
        for week_idx, w in enumerate(weeks):
            model += alloc_vars[c][w] <= Superman_Plus_demand_weekly[c][week_idx]

    model.solve()

    # Process Results
    alloc_results = []
    for c in channels:
        for w in weeks:
            val = value(alloc_vars[c][w])
            if val > 0:
                if isinstance(c, tuple):
                    channel = c[0]
                    region = c[1]
                else:
                    channel = c
                    region = ''
                alloc_results.append({
                    'Channel': channel,
                    'Region': region,
                    'Week': w,
                    'Allocated': val,
                    'Demand': Superman_Plus_demand_weekly[c][weeks.index(w)],
                    'Fulfill rate': f"{val/Superman_Plus_demand_weekly[c][weeks.index(w)]*100:.1f}%"
                })

    # Create DataFrames and Visualizations
    allo_df = pd.DataFrame(alloc_results)
    allo_df.to_excel(os.path.join(output_dir, 'allocation_results.xlsx'))
    results['allo_df'] = allo_df

    # Superman Plus Weekly Allocation Plot
    Superman_Plus_allocation = allo_df.groupby('Week')['Allocated'].sum()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(Superman_Plus_allocation.index, Superman_Plus_allocation.values, 
                   color='skyblue', width=0.6)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')
    plt.title('Superman Plus Weekly Allocation', pad=20, fontsize=12)
    plt.xlabel('Week')
    plt.ylabel('Quantity')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'superman_plus_weekly.png'))
    plt.close()

    # Channel Allocation Plot
    weekly_channel_alloc = allo_df.pivot_table(
        index='Week',
        columns=['Channel', 'Region'],
        values='Allocated',
        aggfunc='sum',
        fill_value=0
    )
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(weekly_channel_alloc.columns)))
    
    # Plot stacked bars using pandas plot method
    weekly_channel_alloc.plot(
        kind='bar',
        stacked=True,
        color=colors,
        width=0.8
    )
    
    # Add total labels on top of bars
    totals = weekly_channel_alloc.sum(axis=1)
    for i, total in enumerate(totals):
        plt.text(i, total, f'{int(total)}', 
                 ha='center', va='bottom')
    
    # Customize plot
    plt.title('Superman Plus Weekly Allocation by Channel', pad=20, fontsize=12)
    plt.xlabel('Week')
    plt.ylabel('Quantity')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Channel-Region', 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with high quality
    plt.savefig(os.path.join(output_dir, 'channel_allocation.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    # Save the numerical summary to Excel
    weekly_channel_alloc.to_excel(os.path.join(output_dir, 'channel_allocation.xlsx'))

    # Supply vs Demand Summary
    # Get the actual weeks from the data
    actual_weeks = list(supply_weekly.keys())
    
    # Calculate total demand for each week
    total_demand = []
    for week in actual_weeks:
        week_demand = sum(demand_cum[product][actual_weeks.index(week)] 
                         for product in demand_cum.keys())
        total_demand.append(week_demand)
    
    summary_df = pd.DataFrame({
        'Weekly Supply': [supply_weekly[week] for week in actual_weeks],
        'Cum Supply': [supply_cum[week] for week in actual_weeks],
        'Total Cum Demand': total_demand
    }, index=actual_weeks)
    
    summary_df['Fulfillment Rate %'] = (summary_df['Cum Supply'] / 
                                      summary_df['Total Cum Demand'] * 100).round(1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # Convert to numpy arrays before plotting
    cum_supply = summary_df['Cum Supply'].to_numpy()
    total_demand = summary_df['Total Cum Demand'].to_numpy()
    fulfillment_rate = summary_df['Fulfillment Rate %'].to_numpy()
    
    ax1.plot(actual_weeks, cum_supply, 'b-', marker='o', 
             label='Cumulative Supply', linewidth=2)
    ax1.plot(actual_weeks, total_demand, 'r--', marker='s', 
             label='Cumulative Demand', linewidth=2)
    for i, week in enumerate(actual_weeks):
        rate = summary_df.loc[week, 'Fulfillment Rate %']
        supply = summary_df.loc[week, 'Cum Supply']
        ax1.text(i, supply, f'{rate}%\n({supply})', ha='center', va='bottom')
    ax1.set_title('Supply vs Demand Summary')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Units')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.bar(actual_weeks, fulfillment_rate, color='skyblue')
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5)
    for i, v in enumerate(fulfillment_rate):
        ax2.text(i, v, f'{v}%', ha='center', va='bottom')
    ax2.set_title('Weekly Fulfillment Rates')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Fulfillment Rate (%)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'supply_demand_summary.png'))
    plt.close()

    # Supply vs Demand by Product
    # Calculate cumulative built + allocated by product
    cum_alloc = {
        'Superman': np.array([85, 100, 110, 120]),      # Wk2-Wk5
        'Superman_Plus': np.array([85, 110, 140, 165]), 
        'Superman_Mini': np.array([60, 60, 70, 75])     
    }

    # Create detailed summary DataFrame
    detailed_df = pd.DataFrame(index=actual_weeks)

    for product in ['Superman', 'Superman_Plus', 'Superman_Mini']:
        detailed_df[f'{product} Supply'] = cum_alloc[product]
        detailed_df[f'{product} Demand'] = demand_cum[product]
        detailed_df[f'{product} Fulfill %'] = np.round(
            (detailed_df[f'{product} Supply'].values / 
             detailed_df[f'{product} Demand'].values * 100), 1)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Define colors
    colors = {
        'Superman': '#4E79A7',        # Blue
        'Superman_Plus': '#F28E2B',   # Orange
        'Superman_Mini': '#59A14F'    # Green
    }

    # Plot 1: Fulfillment Rates by Product
    for product in ['Superman', 'Superman_Plus', 'Superman_Mini']:
        rates = detailed_df[f'{product} Fulfill %'].values
        ax1.plot(actual_weeks, rates, marker='o', linewidth=2, 
                 label=product, color=colors[product])
        
        # Add percentage labels
        for i, rate in enumerate(rates):
            ax1.text(i, rate, f'{rate}%', ha='center', va='bottom')

    ax1.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Target')
    ax1.set_title('Weekly Fulfillment Rates by Product', pad=20, fontsize=14)
    ax1.set_ylabel('Fulfillment Rate (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1))

    # Plot 2: Supply vs Demand by Product
    for product in ['Superman', 'Superman_Plus', 'Superman_Mini']:
        supply = detailed_df[f'{product} Supply'].values
        demand = detailed_df[f'{product} Demand'].values
        
        ax2.plot(actual_weeks, supply, '-', marker='o', 
                 label=f'{product} Supply', color=colors[product])
        ax2.plot(actual_weeks, demand, '--', marker='s', 
                 label=f'{product} Demand', color=colors[product], alpha=0.5)
        
        # Add supply numbers
        for i, sup in enumerate(supply):
            ax2.text(i, sup, f'{int(sup)}', ha='center', va='bottom')

    ax2.set_title('Supply vs Demand by Product', pad=20, fontsize=14)
    ax2.set_ylabel('Units')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Supply vs Demand by Product.png'), 
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    # Save the numerical summary to Excel
    detailed_df.to_excel(os.path.join(output_dir, 'product_supply_demand.xlsx'))

    # Save summary data
    summary_df.to_excel(os.path.join(output_dir, 'summary.xlsx'))
    weekly_channel_alloc.to_excel(os.path.join(output_dir, 'channel_allocation.xlsx'))

    # Add results to dictionary
    results['channel_allocation'] = os.path.join(output_dir, 'channel_allocation.png')
    results['supply_demand_summary'] = os.path.join(output_dir, 'supply_demand_summary.png')
    results['supply_demand_product'] = os.path.join(output_dir, 'Supply vs Demand by Product.png')
    results['superman_plus_weekly'] = os.path.join(output_dir, 'superman_plus_weekly.png')

    return results

if __name__ == "__main__":
    run_allocation()