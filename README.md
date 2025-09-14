# Brain Tumour Detection (YOLO)

This is a small Flask app that uses an Ultralytics YOLO model to detect brain tumors in uploaded images and return annotated results.

## Overview

- Web UI served at `/` to upload medical images (e.g., MRI slices) and set a confidence threshold.
- Prediction endpoint `POST /predict` accepts an image file and `confidence` form value, returns a JSON response with a URL to the annotated result image and a human-readable prediction string.
- Annotated images (with bounding boxes and labels) are saved under the `results/` folder and served at `/results/<filename>`.

## Important files

- `app.py` — Flask application and inference logic.
- `model/` — Contains the YOLO model weights (example: `braitumour1.pt`).
- `templates/index.html` — Upload UI used by the app.
- `uploads/` — Temporary uploads are saved here during prediction.
- `results/` — Annotated output images are written here and served back to the client.

## Dependencies

- Python 3.8+ (3.10 recommended)
- Flask
- Ultralytics YOLO (`ultralytics`)
- Pillow
- Matplotlib
- NumPy

Install dependencies with:

```bash
pip install flask ultralytics pillow matplotlib numpy
```

## Setup & configuration

1. Ensure the model weights are placed in `braintumour/model/` (e.g., `braintumour/model/braitumour1.pt`).
2. In `app.py` update the model path if needed. The current code loads a hard-coded absolute path; change this line to use the local repository copy, for example:

```python
# Replace the hard-coded path with a relative path inside the repo
model = YOLO(os.path.join(os.path.dirname(__file__), 'model', 'braitumour1.pt'))
```

3. Create the `uploads/` and `results/` directories (the app already attempts to create them at startup).

## Running the app

1. Activate your virtual environment and install dependencies.
2. Start the Flask app locally:

```bash
python app.py
```

3. Open `http://127.0.0.1:5000/` in your browser, upload an image and set a confidence threshold, then submit to get predictions.

## API

- POST `/predict`
  - Form fields: `file` (image file), `confidence` (float between 0 and 1)
  - Response JSON example:

```json
{
  "result_image_url": "/results/predicted_sample.jpg",
  "prediction": "Brain Tumor Detected: 1 Tumors"
}
```

- GET `/results/<filename>` — returns the annotated image file from `results/`.

## Notes & recommendations

- The app currently reads the uploaded image into PIL and passes it directly to `model.predict(...)`. For large images you may want to resize or normalize before inference.
- The YOLO model used here should be trained for brain tumor detection; verify that classes and confidence thresholds match your expectations.
- For production use:
  - Do not use debug mode in Flask.
  - Secure file uploads and sanitize filenames.
  - Replace hard-coded absolute paths with configurable environment variables.

## Troubleshooting

- If `model.predict()` fails, ensure the `ultralytics` package version is compatible with your model weights.
- If no detections appear, try lowering the `confidence` input or inspect the model's `model.names` to confirm class labels.

## License

Add a `LICENSE` file if you plan to distribute or publish this project. 