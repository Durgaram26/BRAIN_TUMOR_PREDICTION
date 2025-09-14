from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for
from ultralytics import YOLO
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Load YOLO model
model = YOLO(r'C:\Users\Durga\Desktop\peer\braintumour\model\braitumour1.pt')

# Ensure upload and result directories exist
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
RESULT_FOLDER = os.path.join(os.getcwd(), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check for file and confidence in the form
    if 'file' not in request.files or 'confidence' not in request.form:
        return redirect(request.url)

    file = request.files['file']
    confidence_threshold = float(request.form['confidence'])

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Load image for prediction
        img = Image.open(image_path)

        # Perform prediction with confidence threshold
        results = model.predict(img, conf=confidence_threshold)

        # Extract prediction result and bounding boxes
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        pred_conf = results[0].boxes.conf.cpu().numpy()   # Confidence scores

        # Load image into a NumPy array for drawing
        img_np = np.array(img)
        tumor_detected = False
        tumor_count = 0

        # Create a figure to plot the image with bounding boxes
        fig, ax = plt.subplots(1)
        ax.imshow(img_np)

        # Draw bounding boxes on the image
        for i, box in enumerate(pred_boxes):
            if pred_conf[i] >= confidence_threshold:  # Check confidence threshold
                tumor_detected = True
                tumor_count += 1
                # Draw rectangle (bounding box)
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=2, fill=False)
                ax.add_patch(rect)
                # Add label
                ax.text(x1, y1, f'Tumor {i+1}: {pred_conf[i]:.2f}', color='white', fontsize=12, backgroundcolor='red')

        if tumor_detected:
            prediction = f"Brain Tumor Detected: {tumor_count} Tumors"
        else:
            prediction = "No Brain Tumor Detected"

        # Save the image with bounding boxes
        result_image_name = 'predicted_' + file.filename
        result_image_path = os.path.join(RESULT_FOLDER, result_image_name)
        plt.axis('off')  # Hide axis
        plt.savefig(result_image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Return JSON response for JavaScript
        return jsonify({
            "result_image_url": url_for('result_image', filename=result_image_name),
            "prediction": prediction
        })

# Serve the result image with predictions
@app.route('/results/<filename>')
def result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
