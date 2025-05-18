import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Time Series Weighted Prediction
def time_series_weighted_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price):
    """
    Time series weighted prediction method, considering product lifecycle stages
    Lifecycle characteristics:
    - Launch period (1-3 weeks): Highest sales
    - Fluctuation period (4-10 weeks): Sales fluctuation
    - Stable period (11-15 weeks): Sales stabilization
    """
    # Convert pandas Series to numpy arrays
    princess_sales = np.array(princess_sales)
    dwarf_sales = np.array(dwarf_sales)
    
    # Calculate time weights (considering product lifecycle)
    time_weights = np.array([
        1.0, 1.0, 1.0,  # Launch period (1-3 weeks): Highest weight
        0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5,  # Fluctuation period (4-10 weeks): Gradually decreasing
        0.4, 0.4, 0.4, 0.4, 0.4   # Stable period (11-15 weeks): Stable weight
    ])
    
    # Calculate weighted average sales for each stage
    # Launch period
    launch_princess = np.average(princess_sales[:3], weights=time_weights[:3])
    launch_dwarf = np.average(dwarf_sales[:3], weights=time_weights[:3])
    
    # Fluctuation period
    fluctuation_princess = np.average(princess_sales[3:10], weights=time_weights[3:10])
    fluctuation_dwarf = np.average(dwarf_sales[3:10], weights=time_weights[3:10])
    
    # Stable period
    stable_princess = np.average(princess_sales[10:], weights=time_weights[10:])
    stable_dwarf = np.average(dwarf_sales[10:], weights=time_weights[10:])
    
    # Calculate price ratios
    price_ratio_princess = superman_price / princess_price
    price_ratio_dwarf = superman_price / dwarf_price
    
    # Prediction (giving higher weight to Princess as Superman is an optimized version)
    # Consider different stage weights
    launch_prediction = (launch_princess * 0.8 + launch_dwarf * 0.2) * (price_ratio_princess * 0.8 + price_ratio_dwarf * 0.2)
    fluctuation_prediction = (fluctuation_princess * 0.8 + fluctuation_dwarf * 0.2) * (price_ratio_princess * 0.8 + price_ratio_dwarf * 0.2)
    stable_prediction = (stable_princess * 0.8 + stable_dwarf * 0.2) * (price_ratio_princess * 0.8 + price_ratio_dwarf * 0.2)
    
    # Generate 15-week predictions
    predictions = np.zeros(15)
    predictions[:3] = launch_prediction  # Launch period
    predictions[3:10] = fluctuation_prediction  # Fluctuation period
    predictions[10:] = stable_prediction  # Stable period
    
    # Add some random fluctuation to make predictions more natural
    np.random.seed(42)  # Set random seed for reproducibility
    noise = np.random.normal(0, 0.05, 15)  # Add 5% random fluctuation
    predictions = predictions * (1 + noise)
    
    return predictions

# 2. Product Lifecycle Stage Prediction
def lifecycle_stage_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price):
    """
    Prediction method based on product lifecycle, dividing 15 weeks into three stages:
    - Launch period (1-3 weeks): Highest sales
    - Fluctuation period (4-10 weeks): Sales fluctuation
    - Stable period (11-15 weeks): Sales stabilization
    """
    # Convert pandas Series to numpy arrays
    princess_sales = np.array(princess_sales)
    dwarf_sales = np.array(dwarf_sales)
    
    # Calculate average sales for each stage
    stage_predictions = {}
    stage_weights = {
        'launch': 1.0,      # Launch period weight
        'fluctuation': 0.6, # Fluctuation period weight
        'stable': 0.4      # Stable period weight
    }
    
    # Calculate average sales for each stage
    launch_princess = np.mean(princess_sales[0:3])
    launch_dwarf = np.mean(dwarf_sales[0:3])
    
    fluctuation_princess = np.mean(princess_sales[3:10])
    fluctuation_dwarf = np.mean(dwarf_sales[3:10])
    
    stable_princess = np.mean(princess_sales[10:])
    stable_dwarf = np.mean(dwarf_sales[10:])
    
    # Calculate price impact
    price_ratio_princess = superman_price / princess_price
    price_ratio_dwarf = superman_price / dwarf_price
    
    # Predict sales for each stage
    stage_predictions['launch'] = (
        (launch_princess * 0.8 + launch_dwarf * 0.2) * 
        (price_ratio_princess * 0.8 + price_ratio_dwarf * 0.2) * 
        stage_weights['launch']
    )
    
    stage_predictions['fluctuation'] = (
        (fluctuation_princess * 0.8 + fluctuation_dwarf * 0.2) * 
        (price_ratio_princess * 0.8 + price_ratio_dwarf * 0.2) * 
        stage_weights['fluctuation']
    )
    
    stage_predictions['stable'] = (
        (stable_princess * 0.8 + stable_dwarf * 0.2) * 
        (price_ratio_princess * 0.8 + price_ratio_dwarf * 0.2) * 
        stage_weights['stable']
    )
    
    # Generate 15-week predictions
    predictions = np.zeros(15)
    predictions[0:3] = stage_predictions['launch']  # Launch period
    predictions[3:10] = stage_predictions['fluctuation']  # Fluctuation period
    predictions[10:] = stage_predictions['stable']  # Stable period
    
    # Add some random fluctuation to make predictions more natural
    np.random.seed(42)  # Set random seed for reproducibility
    noise = np.random.normal(0, 0.05, 15)  # Add 5% random fluctuation
    predictions = predictions * (1 + noise)
    
    return predictions

# 3. Trend Analysis Prediction
def trend_analysis_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price):
    """
    Prediction method based on sales trend, considering price impact
    """
    # Convert pandas Series to numpy arrays
    princess_sales = np.array(princess_sales)
    dwarf_sales = np.array(dwarf_sales)
    
    def calculate_trend(sales_data):
        x = np.arange(len(sales_data))
        slope, _ = np.polyfit(x, sales_data, 1)
        return slope
    
    # Calculate sales trends for both products
    princess_trend = calculate_trend(princess_sales)
    dwarf_trend = calculate_trend(dwarf_sales)
    
    # Calculate price impact
    price_ratio_princess = superman_price / princess_price
    price_ratio_dwarf = superman_price / dwarf_price
    
    # Predict initial sales (using weighted average of launch period)
    launch_princess = np.mean(princess_sales[0:3])  # First 3 weeks average
    launch_dwarf = np.mean(dwarf_sales[0:3])       # First 3 weeks average
    
    # Consider price impact and product weights
    initial_sales = (
        (launch_princess * 0.8 + launch_dwarf * 0.2) *  # Product weights
        (price_ratio_princess * 0.8 + price_ratio_dwarf * 0.2)  # Price impact
    )
    
    # Predict trend (using weighted average trend)
    predicted_trend = princess_trend * 0.8 + dwarf_trend * 0.2
    
    # Generate 15-week predictions
    weeks = np.arange(15)
    predictions = initial_sales + predicted_trend * weeks
    
    # Add some random fluctuation to make predictions more natural
    np.random.seed(42)  # Set random seed for reproducibility
    noise = np.random.normal(0, 0.05, 15)  # Add 5% random fluctuation
    predictions = predictions * (1 + noise)
    
    return predictions

# 4. Weighted Average Prediction
def weighted_average_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price):
    """
    Weighted average prediction method based on price differences
    """
    # Convert pandas Series to numpy arrays
    princess_sales = np.array(princess_sales)
    dwarf_sales = np.array(dwarf_sales)
    
    # Calculate price difference weights
    price_diff_princess = abs(superman_price - princess_price)
    price_diff_dwarf = abs(superman_price - dwarf_price)
    total_diff = price_diff_princess + price_diff_dwarf
    
    weight_princess = price_diff_dwarf / total_diff
    weight_dwarf = price_diff_princess / total_diff
    
    # Calculate weighted average prediction
    predictions = (princess_sales * weight_princess + dwarf_sales * weight_dwarf) * 1.05 # assumpttation: 5% growth
    
    # Add new product launch boost
    np.random.seed(42)
    growth_rates = np.random.uniform(0.03, 0.05, 15)
    
    for week in range(15):
        if week < 3:
            growth_multiplier = 1 + growth_rates[week] * 1.5
        elif week < 10:
            growth_multiplier = 1 + growth_rates[week]
        else:
            growth_multiplier = 1 + growth_rates[week] * 0.5
        predictions[week] *= growth_multiplier
    
    return predictions

# 5. Combined Prediction Method
def combined_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price, ts_weight=0.3, lc_weight=0.2, trend_weight=0.2, weighted_weight=0.3):
    """
    Combined prediction method integrating multiple approaches, considering new product launch boost
    Args:
        princess_sales (np.array): Historical sales data for Princess Plus.
        dwarf_sales (np.array): Historical sales data for Dwarf Plus.
        princess_price (float): Current price of Princess Plus.
        dwarf_price (float): Current price of Dwarf Plus.
        superman_price (float): Current price of Superman Plus.
        ts_weight (float): Weight for Time Series Weighted Prediction.
        lc_weight (float): Weight for Lifecycle Stage Prediction.
        trend_weight (float): Weight for Trend Analysis Prediction.
        weighted_weight (float): Weight for Weighted Average Prediction.
    """
    # Get predictions from different methods
    time_series_pred = time_series_weighted_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
    lifecycle_pred = lifecycle_stage_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
    trend_pred = trend_analysis_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
    weighted_pred = weighted_average_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
    
    # Combine predictions
    final_predictions = np.zeros(15)
    
    # Calculate new product launch growth rates (3%-5% random growth per week)
    np.random.seed(42)
    growth_rates = np.random.uniform(0.03, 0.05, 15)
    
    for week in range(15):
        # Determine current stage
        if week < 3:  # Launch period
            stage_weight = 1.0
        elif week < 10:  # Fluctuation period
            stage_weight = 0.6
        else:  # Stable period
            stage_weight = 0.4
        
        # Combine predictions using provided weights
        week_pred = (
            time_series_pred[week] * ts_weight +
            lifecycle_pred[week] * lc_weight +
            trend_pred[week] * trend_weight +
            weighted_pred[week] * weighted_weight
        ) * stage_weight
        
        # Apply new product launch growth rate
        if week < 3:
            growth_multiplier = 1 + growth_rates[week] * 1.5
        elif week < 10:
            growth_multiplier = 1 + growth_rates[week]
        else:
            growth_multiplier = 1 + growth_rates[week] * 0.5
        
        final_predictions[week] = week_pred * growth_multiplier
    
    # Add some random fluctuation to make predictions more natural
    noise = np.random.normal(0, 0.03, 15)
    final_predictions = final_predictions * (1 + noise)
    
    return final_predictions, {
        'time_series': time_series_pred,
        'lifecycle': lifecycle_pred,
        'trend': trend_pred,
        'weighted': weighted_pred
    }

# 6. Visualization Functions
def plot_predictions(predictions, princess_sales, dwarf_sales, title="Sales Prediction Comparison"):
    """
    Visualize prediction results and actual sales data
    """
    # Convert pandas Series to numpy arrays
    princess_sales = np.array(princess_sales)
    dwarf_sales = np.array(dwarf_sales)
    predictions = np.array(predictions)
    
    # Create x-axis data
    weeks = np.arange(1, 16)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot sales trends
    plt.plot(weeks, predictions, marker='o', label='Superman Plus (Predicted)', linewidth=2, color='#2ecc71', linestyle='-')
    plt.plot(weeks, princess_sales, marker='s', label='Princess Plus', linewidth=2, color='#34495e', linestyle='--')
    plt.plot(weeks, dwarf_sales, marker='^', label='Dwarf Plus', linewidth=2, color='#7f8c8d', linestyle='--')
    
    # Set plot properties
    plt.title(title, fontsize=12, pad=15)
    plt.xlabel('Week', fontsize=10, labelpad=8)
    plt.ylabel('Sales', fontsize=10, labelpad=8)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='gray', loc='upper right')
    
    # Add data labels (only show key points)
    for i in [0, 4, 9, 14]:  # Only show weeks 1, 5, 10, 15
        plt.text(i+1, predictions[i], f'{int(predictions[i])}', ha='center', va='bottom', fontsize=8)
        plt.text(i+1, princess_sales[i], f'{int(princess_sales[i])}', ha='center', va='bottom', fontsize=8)
        plt.text(i+1, dwarf_sales[i], f'{int(dwarf_sales[i])}', ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n=== Prediction Statistics ===")
    print(f"\nPredicted Superman Plus:")
    print(f"Total Sales: {np.sum(predictions):,.0f}")
    print(f"Average Weekly Sales: {np.mean(predictions):,.1f}")
    print(f"Starting Point: {predictions[0]:,.0f}")
    print(f"Ending Point: {predictions[-1]:,.0f}")
    
    print(f"\nPrincess Plus:")
    print(f"Total Sales: {np.sum(princess_sales):,.0f}")
    print(f"Average Weekly Sales: {np.mean(princess_sales):,.1f}")
    
    print(f"\nDwarf Plus:")
    print(f"Total Sales: {np.sum(dwarf_sales):,.0f}")
    print(f"Average Weekly Sales: {np.mean(dwarf_sales):,.1f}")
    
    # Calculate correlation coefficients
    correlation_princess = np.corrcoef(predictions, princess_sales)[0,1]
    correlation_dwarf = np.corrcoef(predictions, dwarf_sales)[0,1]
    print(f"\nCorrelation with Princess Plus: {correlation_princess:.3f}")
    print(f"Correlation with Dwarf Plus: {correlation_dwarf:.3f}")

def plot_overall_comparison(df_princess, df_dwarf, predictions_dict):
    """
    Plot overall comparison of three products
    """
    # Calculate total weekly sales
    princess_total = df_princess.sum(axis=1).values
    dwarf_total = df_dwarf.sum(axis=1).values
    superman_total = np.zeros(15)
    
    # Combine predictions from all regions
    for region in ['AMR', 'Europe', 'PAC']:
        superman_total += predictions_dict[region]
    
    # Create x-axis data
    weeks = np.arange(1, 16)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot sales trends
    plt.plot(weeks, superman_total, marker='o', label='Superman Plus (Predicted)', linewidth=2, color='#2ecc71', linestyle='-')
    plt.plot(weeks, princess_total, marker='s', label='Princess Plus', linewidth=2, color='#34495e', linestyle='--')
    plt.plot(weeks, dwarf_total, marker='^', label='Dwarf Plus', linewidth=2, color='#7f8c8d', linestyle='--')
    
    # Set plot properties
    plt.title('Overall Sales Comparison (All Regions)', fontsize=12, pad=15)
    plt.xlabel('Week', fontsize=10, labelpad=8)
    plt.ylabel('Total Sales', fontsize=10, labelpad=8)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='gray', loc='upper right')
    
    # Add data labels (only show key points)
    for i in [0, 4, 9, 14]:  # Only show weeks 1, 5, 10, 15
        plt.text(i+1, superman_total[i], f'{int(superman_total[i])}', ha='center', va='bottom', fontsize=8)
        plt.text(i+1, princess_total[i], f'{int(princess_total[i])}', ha='center', va='bottom', fontsize=8)
        plt.text(i+1, dwarf_total[i], f'{int(dwarf_total[i])}', ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    # plt.show()
    return plt.gcf()
    
    # Print statistics
    print("\n=== Overall Sales Statistics ===")
    print(f"\nSuperman Plus (Predicted):")
    print(f"Total Sales: {np.sum(superman_total):,.0f}")
    print(f"Average Weekly Sales: {np.mean(superman_total):,.1f}")
    print(f"Starting Point: {superman_total[0]:,.0f}")
    print(f"Ending Point: {superman_total[-1]:,.0f}")
    
    print(f"\nPrincess Plus:")
    print(f"Total Sales: {np.sum(princess_total):,.0f}")
    print(f"Average Weekly Sales: {np.mean(princess_total):,.1f}")
    
    print(f"\nDwarf Plus:")
    print(f"Total Sales: {np.sum(dwarf_total):,.0f}")
    print(f"Average Weekly Sales: {np.mean(dwarf_total):,.1f}")

# 7. Overall Sales Trend Analysis
def analyze_overall_trend(princess_plus, dwarf_plus):
    """
    Analyze overall sales trends for Princess Plus and Dwarf Plus
    """
    # Convert to DataFrame
    df_princess = pd.DataFrame(princess_plus)
    df_dwarf = pd.DataFrame(dwarf_plus)
    
    # Calculate total weekly sales (sum of all regions) and convert to numpy array
    princess_total = df_princess.sum(axis=1).values
    dwarf_total = df_dwarf.sum(axis=1).values
    
    # Create x-axis data
    weeks = np.arange(1, 16)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot sales trends
    plt.plot(weeks, princess_total, marker='o', label='Princess Plus', linewidth=2, color='#34495e')
    plt.plot(weeks, dwarf_total, marker='s', label='Dwarf Plus', linewidth=2, color='#7f8c8d')
    
    # Add trend lines
    z_princess = np.polyfit(weeks, princess_total, 1)
    z_dwarf = np.polyfit(weeks, dwarf_total, 1)
    p_princess = np.poly1d(z_princess)
    p_dwarf = np.poly1d(z_dwarf)
    
    plt.plot(weeks, p_princess(weeks), '--', color='#34495e', alpha=0.5, label='Princess Trend')
    plt.plot(weeks, p_dwarf(weeks), '--', color='#7f8c8d', alpha=0.5, label='Dwarf Trend')
    
    # Set plot properties
    plt.title('Overall Sales Trend (All Regions)', fontsize=14, pad=20)
    plt.xlabel('Week', fontsize=12, labelpad=10)
    plt.ylabel('Total Sales', fontsize=12, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    
    # Add data labels
    for i, (p, d) in enumerate(zip(princess_total, dwarf_total)):
        plt.text(i+1, p, f'{int(p)}', ha='center', va='bottom', fontsize=8)
        plt.text(i+1, d, f'{int(d)}', ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    # plt.show()
    return plt.gcf()
    
    # Print statistics
    print("\n=== Overall Sales Statistics ===")
    print("\nPrincess Plus:")
    print(f"Total Sales: {np.sum(princess_total):,.0f}")
    print(f"Average Weekly Sales: {np.mean(princess_total):,.1f}")
    print(f"Sales Trend (slope): {z_princess[0]:,.1f} units/week")
    print(f"Starting Point: {princess_total[0]:,.0f}")
    print(f"Ending Point: {princess_total[-1]:,.0f}")
    
    print("\nDwarf Plus:")
    print(f"Total Sales: {np.sum(dwarf_total):,.0f}")
    print(f"Average Weekly Sales: {np.mean(dwarf_total):,.1f}")
    print(f"Sales Trend (slope): {z_dwarf[0]:,.1f} units/week")
    print(f"Starting Point: {dwarf_total[0]:,.0f}")
    print(f"Ending Point: {dwarf_total[-1]:,.0f}")
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(princess_total, dwarf_total)[0,1]
    print(f"\nCorrelation between Princess and Dwarf sales: {correlation:.3f}")

# 8. Prediction Methods Comparison Visualization
def plot_prediction_methods_comparison(predictions_dict, princess_total, dwarf_total, title="Prediction Methods Comparison"):
    """
    Visualize comparison of different prediction methods
    """
    weeks = np.arange(1, 16)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Define green color scheme (uniform line thickness)
    styles = {
        'time_series': {'color': '#2ecc71', 'linewidth': 2.0},    # Bright green
        'lifecycle': {'color': '#27ae60', 'linewidth': 2.0},      # Dark green
        'trend': {'color': '#16a085', 'linewidth': 2.0},          # Teal
        'weighted': {'color': '#1abc9c', 'linewidth': 2.0}        # Light green
    }
    
    # Plot results from each prediction method (all using solid lines)
    for method, pred in predictions_dict.items():
        plt.plot(weeks, pred, 
                label=f'{method.title()} Prediction',
                linewidth=styles[method]['linewidth'],
                color=styles[method]['color'],
                linestyle='-',
                alpha=0.8)
    
    # Plot actual sales data (using gray-black color scheme)
    plt.plot(weeks, princess_total, 
            marker='s', 
            label='Princess Plus',
            linewidth=2,
            color='#34495e',  # Dark gray-blue
            linestyle='--',
            alpha=0.6)
    plt.plot(weeks, dwarf_total,
            marker='^',
            label='Dwarf Plus',
            linewidth=2,
            color='#7f8c8d',  # Light gray
            linestyle='--',
            alpha=0.6)
    
    # Set plot properties
    plt.title(title, fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Week', fontsize=12, labelpad=8)
    plt.ylabel('Total Sales', fontsize=12, labelpad=8)
    plt.grid(True, linestyle='--', alpha=0.2)
    
    # Optimize legend
    legend = plt.legend(
        fontsize=10,
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        loc='upper right',
        bbox_to_anchor=(1.15, 1)
    )
    
    # Add data labels (only show key points)
    for i in [0, 4, 9, 14]:
        # Add labels for each prediction method
        for method, pred in predictions_dict.items():
            plt.text(i+1, pred[i], 
                    f'{int(pred[i])}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=styles[method]['color'],
                    fontweight='bold')
        
        # Add labels for actual sales data
        plt.text(i+1, princess_total[i],
                f'{int(princess_total[i])}',
                ha='center',
                va='bottom',
                fontsize=8,
                color='#34495e',  # Dark gray-blue
                fontweight='bold')
        plt.text(i+1, dwarf_total[i],
                f'{int(dwarf_total[i])}',
                ha='center',
                va='top',
                fontsize=8,
                color='#7f8c8d',  # Light gray
                fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Add explanatory text
    plt.figtext(0.02, 0.02,
                'Note: Different shades of green represent different prediction methods\n'
                'Time Series: Bright Green\n'
                'Lifecycle: Dark Green\n'
                'Trend: Teal\n'
                'Weighted Average: Light Green',
                fontsize=8,
                style='italic')
    
    plt.show()
    
    # Print statistics for each method
    print("\n=== Prediction Methods Statistics ===")
    for method, pred in predictions_dict.items():
        print(f"\n{method.title()} Prediction:")
        print(f"Total Sales: {np.sum(pred):,.0f}")
        print(f"Average Weekly Sales: {np.mean(pred):,.1f}")
        print(f"Starting Point: {pred[0]:,.0f}")
        print(f"Ending Point: {pred[-1]:,.0f}")
        
        # Calculate correlation with actual sales data
        correlation_princess = np.corrcoef(pred, princess_total)[0,1]
        correlation_dwarf = np.corrcoef(pred, dwarf_total)[0,1]
        print(f"Correlation with Princess Plus: {correlation_princess:.3f}")
        print(f"Correlation with Dwarf Plus: {correlation_dwarf:.3f}")

# 9. Main Function
def main():
    # Example data
    princess_plus = {
        'AMR': [240,170,130,90,110,130,110,110,110,130,70,90,100,80,90],
        'Europe': [100,80,90,80,70,60,60,60,50,50,50,80,80,60,50],
        'PAC': [150,220,240,150,130,120,110,100,110,100,120,130,160,120,100]
    }

    dwarf_plus = {
        'AMR': [320,220,170,190,200,170,160,160,140,140,180,160,160,170,190],
        'Europe': [80,100,60,100,100,90,80,80,80,70,90,80,80,80,70],
        'PAC': [230,210,140,140,140,150,140,175,140,90,90,100,110,100,90]
    }
    
    # First perform overall trend analysis
    analyze_overall_trend(princess_plus, dwarf_plus)
    
    # Price data
    princess_price = 200
    dwarf_price = 120
    superman_price = 205
    
    # Convert to DataFrame
    df_princess = pd.DataFrame(princess_plus)
    df_dwarf = pd.DataFrame(dwarf_plus)
    
    # Calculate total sales data
    princess_total = df_princess.sum(axis=1).values
    dwarf_total = df_dwarf.sum(axis=1).values
    
    # Store predictions for all regions
    predictions_dict = {}
    all_methods_predictions = {}
    
    # Make predictions for each region
    regions = ['AMR', 'Europe', 'PAC']
    for region in regions:
        predictions, methods_predictions = combined_prediction(
            princess_sales=df_princess[region].values,
            dwarf_sales=df_dwarf[region].values,
            princess_price=princess_price,
            dwarf_price=dwarf_price,
            superman_price=superman_price
        )
        predictions_dict[region] = predictions
        
        # Accumulate predictions from each method
        if not all_methods_predictions:
            all_methods_predictions = methods_predictions
        else:
            for method in methods_predictions:
                all_methods_predictions[method] += methods_predictions[method]
    
    # Plot prediction methods comparison
    plot_prediction_methods_comparison(
        all_methods_predictions,
        princess_total,
        dwarf_total,
        title="Prediction Methods Comparison (All Regions)"
    )
    
    # Then plot overall comparison
    plot_overall_comparison(df_princess, df_dwarf, predictions_dict)
    
    # Finally plot predictions for each region
    for region in regions:
        plot_predictions(
            predictions_dict[region],
            df_princess[region].values,
            df_dwarf[region].values,
            title=f"Sales Prediction Comparison - {region}"
        )

    # Add debug code
    for name, preds in predictions_dict.items():
        print(f"{name}: {type(preds)}, shape: {np.shape(preds)}")

if __name__ == "__main__":
    main() 