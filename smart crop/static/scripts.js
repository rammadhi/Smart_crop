// Handle crop prediction form submission
document.getElementById('predict-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    // Get the input features from the form
    const features = document.getElementById('features').value.split(',').map(item => item.trim());

    // Send the input features to the /predict API endpoint
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features: features })
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result
        if (data.predicted_crop) {
            document.getElementById('prediction-result').innerHTML = `<strong>Predicted Crop:</strong> ${data.predicted_crop}`;
        } else if (data.error) {
            document.getElementById('prediction-result').innerHTML = `<strong>Error:</strong> ${data.error}`;
        }
    })
    .catch(error => {
        document.getElementById('prediction-result').innerHTML = `<strong>Error:</strong> ${error.message}`;
    });
});

// Handle crop trade form submission
document.getElementById('trade-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    // Get the form data
    const crop = document.getElementById('crop').value;
    const quantity = document.getElementById('quantity').value;
    const price = document.getElementById('price').value;

    // Send the trade data to the /trade API endpoint
    fetch('/trade', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ crop: crop, quantity: quantity, price: price })
    })
    .then(response => response.json())
    .then(data => {
        // Display the trade result
        if (data.status) {
            document.getElementById('trade-result').innerHTML = `<strong>Status:</strong> ${data.status}<br><strong>Details:</strong> ${JSON.stringify(data.trade_data)}`;
        } else if (data.error) {
            document.getElementById('trade-result').innerHTML = `<strong>Error:</strong> ${data.error}`;
        }
    })
    .catch(error => {
        document.getElementById('trade-result').innerHTML = `<strong>Error:</strong> ${error.message}`;
    });
});