from flask import Flask, request, render_template, jsonify
import joblib
import pickle
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and mappings at the start
model = joblib.load('prediction_model.joblib')  # Replace with the path to your model
file_path = './'  # Replace with the correct path

# Load the dictionary from the provided file path
with open(file_path + 'categorical_mappings.pkl', 'rb') as file:
    categorical_mappings = pickle.load(file)

with open(file_path + 'scaler_pkl.pkl', 'rb') as file:
    scaler_pkl = pickle.load(file)

# Function to preprocess the user input
def preprocess_user_input(user_input):
    for column, mapping in categorical_mappings.items():
        if column in user_input:
            mapped_value = mapping.get(user_input[column], -1)  # Default to -1 if not found
            user_input[column] = mapped_value

    # Convert to DataFrame
    user_input_df = pd.DataFrame([user_input])

    # Extract and scale numerical columns
    columns_to_scale = ['population', 'GDHI', 'bank_rate', 'CPIH', 'unemployment_rate', 'GDP']
    user_input_scaled = user_input_df[columns_to_scale]
    user_input_scaled_values = scaler_pkl.transform(user_input_scaled)

    # Extract untouched columns (latitude, longitude, year, month, day)
    untouched_columns = ['latitude', 'longitude', 'year', 'month', 'day']
    untouched_data = user_input_df[untouched_columns]

    # Combine scaled data with untouched columns
    scaled_data_df = pd.DataFrame(user_input_scaled_values, columns=columns_to_scale)
    final_input_df = pd.concat([scaled_data_df, untouched_data, 
                                user_input_df[['property_type', 'old_new', 'duration', 'town_city', 'pdp_category_type']]], axis=1)

    final_input_df = final_input_df[['property_type', 'old_new', 'duration', 'town_city', 'pdp_category_type', 'latitude', 'longitude', 'year', 'population', 'GDHI', 'bank_rate', 'CPIH', 'unemployment_rate', 'GDP', 'month', 'day']]

    return final_input_df

# Home route to render HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route for HTML form submissions (POST)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        user_input = {
            'population': int(request.form['population']),
            'GDHI': float(request.form['GDHI']),
            'bank_rate': float(request.form['bank_rate']),
            'CPIH': float(request.form['CPIH']),
            'unemployment_rate': float(request.form['unemployment_rate']),
            'GDP': float(request.form['GDP']),
            'property_type': request.form['property_type'],
            'old_new': request.form['old_new'],
            'duration': request.form['duration'],
            'pdp_category_type': request.form['pdp_category_type'],
            'town_city': request.form['town_city'],
            'latitude': float(request.form['latitude']),
            'longitude': float(request.form['longitude']),
            'year': int(request.form['year']),
            'month': int(request.form['month']),
            'day': int(request.form['day']),
        }

        # Preprocess the data
        final_input_df = preprocess_user_input(user_input)

        # Predict using the pre-trained model
        prediction = model.predict(final_input_df)

        # Round prediction to 3 decimal places
        predicted_value = np.round(prediction[0], 3)

        # Render the form again with the prediction
        return render_template('index.html', prediction=predicted_value, user_input=user_input)

    except Exception as e:
        # Handle any exceptions and return the error message
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)