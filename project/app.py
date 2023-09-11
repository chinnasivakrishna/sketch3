import os
import cv2
import numpy as np
from flask import Flask, request, send_file, render_template
from io import BytesIO

app = Flask(__name__)
def process_image(file):
    # Read the uploaded image from memory
    image_stream = BytesIO(file.read())
    image_stream.seek(0)
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return None

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filtering to enhance edges while removing noise
    filtered_image = cv2.bilateralFilter(grayscale_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply adaptive thresholding to create a binary image
    threshold_image = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)

    # Apply erosion to eliminate small dots
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(threshold_image, kernel, iterations=1)

    # Find the contours in the eroded image
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank white image with the same size as the original image
    sketch_image = np.ones_like(image) * 255

    # Draw each contour on the blank image with black color (0) and thicker lines (3)
    for contour in contours:
        cv2.drawContours(sketch_image, [contour], -1, (0), 3)

    # Create a mask to isolate the eye globe area
    if len(contours) > 0:
        mask = np.zeros_like(grayscale_image)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        # Change the eye globe area to dim black using the mask
        sketch_image[mask == 255] = 50  # Change the intensity value here

    return sketch_image

@app.route('/')
@app.route('/')
def upload_form():
    # Render the HTML form for image upload (index.html)
    with open('index.html', 'r') as f:
        html_content = f.read()
    return html_content


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    if file:
        try:
            # Process the uploaded image
            processed_image = process_image(file)

            if processed_image is not None:
                # Convert processed image to JPEG format in memory
                ret, img_data = cv2.imencode('.jpg', processed_image)
                img_bytes = img_data.tobytes()

                # Send the processed image as a response
                return send_file(BytesIO(img_bytes), mimetype='image/jpeg')
            else:
                return "Image processing failed"
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)
