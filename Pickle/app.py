from flask import Flask, jsonify, request
from flask_cors import CORS 
import pickle
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Allow specific file types
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

# Configure upload folder and max content length (optional)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB

# Load the pickle model
try:
    with open('crop_analysis.pkl', 'rb') as file:
        model_dict = pickle.load(file)
    print(f"Model loaded successfully. Type: {type(model_dict)}")
    print(f"Dictionary keys: {list(model_dict.keys())}")
    print("First 5 rows of dataset:")
    print(model_dict['data'].head())
except Exception as e:
    print(f"Error loading pickle file: {e}")
    model_dict = None

# Function to prepare test data from an image
def prepare_test_data(image_path, data):
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")
    
    # Process image with K-Means (for debugging, not used in prediction yet)
    image_resized = cv2.resize(image, (256, 256))
    pixels = image_resized.reshape(-1, 3) / 255.0
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(pixels)
    print(f"K-Means cluster centers: {kmeans.cluster_centers_}")
    
    # Match to a row in the dataset (using first row for testing)
    row = data.iloc[0]  # Replace with logic to match image to row if needed
    features = ['Rainfall', 'Temperature', 'Humidity', 'Pest_Disease_Flag', 'Crop_Health_Score']
    test_data = np.array([row[features].values])  # Shape: (1, 5)
    print(f"Test data shape: {test_data.shape}")
    print(f"Features from dataset: {row[features].to_dict()}")
    print(f"Actual Crop_Yield from dataset: {row['Crop_Yield']}")
    
    test_data_df = pd.DataFrame(test_data, columns=features)
    return test_data_df

# Check if the uploaded file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if model_dict is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    model = model_dict.get('model')
    if model is None:
        return jsonify({'error': 'No model found in dictionary'}), 500
    
    data = model_dict.get('data')
    if data is None:
        return jsonify({'error': 'No data found in dictionary'}), 500

    # Check if the request contains the image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    # If no file is selected, return an error
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the image to the server
        file.save(image_path)
        
        try:
            test_data = prepare_test_data(image_path, data)
        except Exception as e:
            print(f"Error preparing test data: {e}")
            return jsonify({'error': f'Error preparing test data: {str(e)}'}), 500
        
        try:
            prediction = model.predict(test_data)
            result = {
                'predicted_crop_yield': prediction.tolist(),
                'status': 'success'
            }
            return jsonify(result), 200
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/')
def home():
    return "Flask server is running!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)
