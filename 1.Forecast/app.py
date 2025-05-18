from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from Forecast import (
    time_series_weighted_prediction,
    lifecycle_stage_prediction,
    trend_analysis_prediction,
    weighted_average_prediction,
    combined_prediction,
    analyze_overall_trend
)

app = Flask(__name__)

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default parameters
    princess_price = 200.0
    dwarf_price = 120.0
    superman_price = 205.0
    
    # Using real historical sales data for all regions (assuming similar structure as in Forecast.py main function)
    princess_plus_all_regions = {
        'AMR': np.array([240,170,130,90,110,130,110,110,110,130,70,90,100,80,90]),
        'Europe': np.array([100,80,90,80,70,60,60,60,50,50,50,80,80,60,50]),
        'PAC': np.array([150,220,240,150,130,120,110,100,110,100,120,130,160,120,100])
    }

    dwarf_plus_all_regions = {
        'AMR': np.array([320,220,170,190,200,170,160,160,140,140,180,160,160,170,190]),
        'Europe': np.array([80,100,60,100,100,90,80,80,80,70,90,80,80,80,70]),
        'PAC': np.array([230,210,140,140,140,150,140,175,140,90,90,100,110,100,90])
    }

    # Use AMR data for individual predictions as before (or modify if needed)
    princess_sales = princess_plus_all_regions['AMR']
    dwarf_sales = dwarf_plus_all_regions['AMR']

    if request.method == 'POST':
        princess_price = float(request.form.get('princess_price', princess_price))
        dwarf_price = float(request.form.get('dwarf_price', dwarf_price))
        superman_price = float(request.form.get('superman_price', superman_price))

    # Default weights for the four methods
    ts_weight = 0.3
    lc_weight = 0.3
    trend_weight = 0.2
    weighted_weight = 0.2
    
    # Normalize weights (optional here with default 0.25, but good practice if they were user-adjustable)
    total_weight = ts_weight + lc_weight + trend_weight + weighted_weight
    if total_weight > 0:
        ts_weight /= total_weight
        lc_weight /= total_weight
        trend_weight /= total_weight
        weighted_weight /= total_weight

    # Generate individual predictions
    arima_preds = time_series_weighted_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
    prophet_preds = lifecycle_stage_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
    lstm_preds = trend_analysis_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
    xgboost_preds = weighted_average_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)

    # Calculate combined prediction using the specific weighted average
    # Ensure all individual predictions are 1D arrays and have the same length
    # (Assuming they do based on previous code logic)
    combined_preds = (
        ts_weight * np.array(arima_preds).flatten() +
        lc_weight * np.array(prophet_preds).flatten() +
        trend_weight * np.array(lstm_preds).flatten() +
        weighted_weight * np.array(xgboost_preds).flatten()
    )

    # Predictions dictionary for plotting and tables
    predictions_dict = {
        "Time Series Weighted Prediction": arima_preds,
        "Product Lifecycle Stage Prediction": prophet_preds,
        "Trend Analysis Prediction": lstm_preds,
        "Weighted Average Prediction": xgboost_preds,
        "Combined Prediction": combined_preds
    }

    # Weights dictionary to pass to the template
    display_weights = {
        "Time Series Weighted Prediction": ts_weight,
        "Product Lifecycle Stage Prediction": lc_weight,
        "Trend Analysis Prediction": trend_weight,
        "Weighted Average Prediction": weighted_weight
    }

    # Generate plots and tables
    plots = {}
    tables = {}
    weeks = np.arange(1, 16)

    # Generate and add the overall sales trend plot
    overall_trend_fig = analyze_overall_trend(princess_plus_all_regions, dwarf_plus_all_regions)
    plots["Overall Sales Trend (All Regions)"] = plot_to_base64(overall_trend_fig)

    for name, preds in predictions_dict.items():
        preds = np.array(preds).flatten()  # Ensure 1D array
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(weeks, preds, label='Prediction', marker='o')
        ax.plot(weeks, princess_sales, label='Princess History', linestyle='--', marker='x')
        ax.plot(weeks, dwarf_sales, label='Dwarf History', linestyle='--', marker='x')
        ax.set_title(f'{name}')
        ax.set_xlabel('Week')
        ax.set_ylabel('Sales')
        ax.legend()
        plots[name] = plot_to_base64(fig)

        # Table
        df = pd.DataFrame({
            'Week': weeks,
            'Predicted Sales': np.round(preds, 2)
        })
        tables[name] = df.to_html(classes='table table-striped', index=False, border=0, justify='center')

    # Generate regional predictions after combined prediction
    regions = ['AMR', 'Europe', 'PAC']
    regional_predictions = {}
    
    for region in regions:
        # Get predictions for this region
        predictions, methods_predictions = combined_prediction(
            princess_sales=princess_plus_all_regions[region],
            dwarf_sales=dwarf_plus_all_regions[region],
            princess_price=princess_price,
            dwarf_price=dwarf_price,
            superman_price=superman_price
        )
        regional_predictions[region] = predictions
        
        # Create plot for this region
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(weeks, predictions, label='Prediction', marker='o')
        ax.plot(weeks, princess_plus_all_regions[region], label='Princess History', linestyle='--', marker='x')
        ax.plot(weeks, dwarf_plus_all_regions[region], label='Dwarf History', linestyle='--', marker='x')
        ax.set_title(f'Regional Prediction - {region}')
        ax.set_xlabel('Week')
        ax.set_ylabel('Sales')
        ax.legend()
        plots[f"Regional Prediction - {region}"] = plot_to_base64(fig)
        
        # Create table for this region
        df = pd.DataFrame({
            'Week': weeks,
            'Predicted Sales': np.round(predictions, 2)
        })
        tables[f"Regional Prediction - {region}"] = df.to_html(classes='table table-striped', index=False, border=0, justify='center')

    return render_template('index.html',
        princess_price=princess_price,
        dwarf_price=dwarf_price,
        superman_price=superman_price,
        plots=plots,
        tables=tables,
        weights=display_weights
    )

@app.route('/update_combined_weights', methods=['POST'])
def update_combined_weights():
    try:
        # Get updated weights from the form
        ts_weight = float(request.form.get('ts_weight', 0.25))
        lc_weight = float(request.form.get('lc_weight', 0.25))
        trend_weight = float(request.form.get('trend_weight', 0.25))
        weighted_weight = float(request.form.get('weighted_weight', 0.25))

        # Normalize weights
        total_weight = ts_weight + lc_weight + trend_weight + weighted_weight
        if total_weight > 0:
            ts_weight /= total_weight
            lc_weight /= total_weight
            trend_weight /= total_weight
            weighted_weight /= total_weight
        else:
             # Handle the case where total_weight is 0, maybe set default weights or return an error
             # For now, let's set to default if all are 0
             ts_weight = lc_weight = trend_weight = weighted_weight = 0.25

        # Retrieve necessary data for prediction
        princess_price = 200.0
        dwarf_price = 120.0
        superman_price = 205.0

        # Historical sales data for all regions
        princess_plus_all_regions = {
            'AMR': np.array([240,170,130,90,110,130,110,110,110,130,70,90,100,80,90]),
            'Europe': np.array([100,80,90,80,70,60,60,60,50,50,50,80,80,60,50]),
            'PAC': np.array([150,220,240,150,130,120,110,100,110,100,120,130,160,120,100])
        }

        dwarf_plus_all_regions = {
            'AMR': np.array([320,220,170,190,200,170,160,160,140,140,180,160,160,170,190]),
            'Europe': np.array([80,100,60,100,100,90,80,80,80,70,90,80,80,80,70]),
            'PAC': np.array([230,210,140,140,140,150,140,175,140,90,90,100,110,100,90])
        }

        # Using AMR data for combined prediction
        princess_sales = princess_plus_all_regions['AMR']
        dwarf_sales = dwarf_plus_all_regions['AMR']

        # Generate individual predictions for AMR
        arima_preds = time_series_weighted_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
        prophet_preds = lifecycle_stage_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
        lstm_preds = trend_analysis_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)
        xgboost_preds = weighted_average_prediction(princess_sales, dwarf_sales, princess_price, dwarf_price, superman_price)

        # Calculate combined prediction using the updated weights
        combined_preds = (
            ts_weight * np.array(arima_preds).flatten() +
            lc_weight * np.array(prophet_preds).flatten() +
            trend_weight * np.array(lstm_preds).flatten() +
            weighted_weight * np.array(xgboost_preds).flatten()
        )

        # Generate plot and table for Combined Prediction
        weeks = np.arange(1, 16)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(weeks, combined_preds, label='Prediction', marker='o')
        ax.plot(weeks, princess_sales, label='Princess History', linestyle='--', marker='x')
        ax.plot(weeks, dwarf_sales, label='Dwarf History', linestyle='--', marker='x')
        ax.set_title('Combined Prediction')
        ax.set_xlabel('Week')
        ax.set_ylabel('Sales')
        ax.legend()
        combined_plot_img = plot_to_base64(fig)

        df = pd.DataFrame({
            'Week': weeks,
            'Predicted Sales': np.round(combined_preds, 2)
        })
        combined_table_html = df.to_html(classes='table table-striped', index=False, border=0, justify='center')

        # Generate updated regional predictions
        regional_plots = {}
        regional_tables = {}
        regions = ['AMR', 'Europe', 'PAC']

        for region in regions:
            # Get predictions for this region using the same weights
            predictions, _ = combined_prediction(
                princess_sales=princess_plus_all_regions[region],
                dwarf_sales=dwarf_plus_all_regions[region],
                princess_price=princess_price,
                dwarf_price=dwarf_price,
                superman_price=superman_price,
                ts_weight=ts_weight,
                lc_weight=lc_weight,
                trend_weight=trend_weight,
                weighted_weight=weighted_weight
            )
            
            # Create plot for this region
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(weeks, predictions, label='Prediction', marker='o')
            ax.plot(weeks, princess_plus_all_regions[region], label='Princess History', linestyle='--', marker='x')
            ax.plot(weeks, dwarf_plus_all_regions[region], label='Dwarf History', linestyle='--', marker='x')
            ax.set_title(f'Regional Prediction - {region}')
            ax.set_xlabel('Week')
            ax.set_ylabel('Sales')
            ax.legend()
            regional_plots[region] = plot_to_base64(fig)
            
            # Create table for this region
            df = pd.DataFrame({
                'Week': weeks,
                'Predicted Sales': np.round(predictions, 2)
            })
            regional_tables[region] = df.to_html(classes='table table-striped', index=False, border=0, justify='center')

        return jsonify({
            'status': 'success',
            'plot': combined_plot_img,
            'table': combined_table_html,
            'weights': {
                "Time Series Weighted Prediction": ts_weight,
                "Product Lifecycle Stage Prediction": lc_weight,
                "Trend Analysis Prediction": trend_weight,
                "Weighted Average Prediction": weighted_weight
            },
            'regional_plots': regional_plots,
            'regional_tables': regional_tables
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 