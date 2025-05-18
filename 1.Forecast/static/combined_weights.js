function updateCombinedPrediction() {
    const tsWeight = document.getElementById('ts_weight').value;
    const lcWeight = document.getElementById('lc_weight').value;
    const trendWeight = document.getElementById('trend_weight').value;
    const weightedWeight = document.getElementById('weighted_weight').value;

    // Validate weights sum to 1 (optional, could also normalize on backend)
    const weights = [parseFloat(tsWeight) || 0, parseFloat(lcWeight) || 0, parseFloat(trendWeight) || 0, parseFloat(weightedWeight) || 0];
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);

    if (totalWeight !== 1) {
        alert('Weights must sum to 1. Please adjust the weights.');
        return; // Stop if weights don't sum to 1
    }

    const formData = new FormData();
    formData.append('ts_weight', tsWeight);
    formData.append('lc_weight', lcWeight);
    formData.append('trend_weight', trendWeight);
    formData.append('weighted_weight', weightedWeight);

    fetch('/update_combined_weights', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Update combined prediction plot and table
            const combinedPlotElement = document.getElementById('plot-Combined-Prediction');
            if (combinedPlotElement) {
                combinedPlotElement.src = `data:image/png;base64,${data.plot}`;
            }

            const combinedTableElement = document.getElementById('table-Combined-Prediction');
            if (combinedTableElement) {
                combinedTableElement.innerHTML = data.table;
            }

            // Optionally update the displayed weights in the input fields with normalized values
             document.getElementById('ts_weight').value = data.weights["Time Series Weighted Prediction"];
             document.getElementById('lc_weight').value = data.weights["Product Lifecycle Stage Prediction"];
             document.getElementById('trend_weight').value = data.weights["Trend Analysis Prediction"];
             document.getElementById('weighted_weight').value = data.weights["Weighted Average Prediction"];

            alert('Combined Prediction updated successfully!');
        } else {
            alert('Error updating Combined Prediction: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error updating Combined Prediction. Please try again.');
    });
} 