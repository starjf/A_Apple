import os
import sys
from flask import Flask, render_template, send_from_directory, jsonify
from allocation_module import run_allocation

# 
sys.stdout.flush()
print("Starting application...")

app = Flask(__name__, static_folder='static')

@app.route('/get_allocation_data')
def get_allocation_data():
    try:
        results = run_allocation()
        allo_df = results['allo_df']
        
        # Prepare weekly data
        weekly_data = []
        for week in ['Jan Wk2', 'Jan Wk3', 'Jan Wk4', 'Jan Wk5']:
            week_total = allo_df[allo_df['Week'] == week]['Allocated'].sum()
            weekly_data.append({
                'week': week,
                'value': int(week_total)
            })
        
        # Prepare channel data
        channel_data = {}
        for week in ['Jan Wk2', 'Jan Wk3', 'Jan Wk4', 'Jan Wk5']:
            week_data = allo_df[allo_df['Week'] == week]
            channels = []
            for _, row in week_data.iterrows():
                channel_name = f"{row['Channel']} {row['Region']}".strip()
                channels.append({
                    'name': channel_name,
                    'value': int(row['Allocated']),
                    'color': f'rgba({hash(channel_name) % 255}, {hash(channel_name + "1") % 255}, {hash(channel_name + "2") % 255}, 0.7)'
                })
            channel_data[week] = channels
        
        return jsonify({
            'weekly': weekly_data,
            'channel': channel_data
        })
    except Exception as e:
        print(f"Error getting allocation data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    try:
        print("Processing request to /")
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        
        if not os.path.exists(static_dir):
            print(f"Creating static directory: {static_dir}")
            os.makedirs(static_dir)
        
        print("Running allocation analysis...")
        try:
            results = run_allocation()
            print("Allocation analysis complete")
            print(f"Results: {results}")
        except Exception as alloc_error:
            print(f"Allocation error details: {str(alloc_error)}")
            return f"Error in allocation analysis: {str(alloc_error)}", 500
        
        files = os.listdir(static_dir)
        print(f"Files in static directory: {files}")
        
        # Extract just the filenames from the full paths
        supply_demand_summary = os.path.basename(results['supply_demand_summary'])
        supply_demand_product = os.path.basename(results['supply_demand_product'])
        
        return render_template('index.html', 
                           supply_demand_summary=supply_demand_summary,
                           supply_demand_product=supply_demand_product)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    try:
        port = 5000
        print(f"Starting Flask app on port {port}...")
        print(f"Please open http://127.0.0.1:{port} in your browser")
        app.run(debug=True, port=port, host='127.0.0.1')
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        sys.exit(1)
