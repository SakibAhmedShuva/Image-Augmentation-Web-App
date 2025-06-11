from flask import Flask, request, jsonify, send_file, render_template, after_this_request
from flask_cors import CORS
import cv2
import numpy as np
import os
from PIL import Image
import albumentations as A
from dotenv import load_dotenv
import random
import tempfile
import shutil
import zipfile
from werkzeug.utils import secure_filename
import base64
import io
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
SESSIONS_FOLDER = 'sessions'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSIONS_FOLDER, exist_ok=True)

# Set fixed seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ImageAugmenter:
    """
    Class to create augmented versions of images for damage detection,
    without altering the geometry (no rotation, zoom, etc.)
    """

    def __init__(self, enabled_augmentations=None):
        # Define augmentations that don't change geometry
        self.augmentations = {}
        
        # If specific augmentations are provided, use those; otherwise use env vars
        if enabled_augmentations:
            enabled_augs = set(enabled_augmentations)
        else:
            enabled_augs = set()

        # Check each augmentation's enable flag
        if enabled_augmentations is None:
            # Use environment variables
            if os.getenv("ENABLE_AUG_GRAYSCALE", "0") == "1":
                enabled_augs.add("grayscale")
            if os.getenv("ENABLE_AUG_HUE_NEG", "0") == "1":
                enabled_augs.add("hue_neg")
            # Add all other env checks...
        
        # Define all available augmentations
        aug_definitions = {
            "grayscale": A.Compose([A.ToGray(p=1.0)], p=1.0),
            "hue_neg": A.Compose([A.HueSaturationValue(hue_shift_limit=(-20, -20), sat_shift_limit=0, val_shift_limit=0, p=1.0)], p=1.0),
            "hue_pos": A.Compose([A.HueSaturationValue(hue_shift_limit=(20, 20), sat_shift_limit=0, val_shift_limit=0, p=1.0)], p=1.0),
            "sat_neg": A.Compose([A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-30, -30), val_shift_limit=0, p=1.0)], p=1.0),
            "sat_pos": A.Compose([A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(40, 40), val_shift_limit=0, p=1.0)], p=1.0),
            "brightness_neg": A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.2), contrast_limit=0, p=1.0)], p=1.0),
            "brightness_pos": A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=0, p=1.0)], p=1.0),
            "contrast_neg": A.Compose([A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.2, -0.2), p=1.0)], p=1.0),
            "contrast_pos": A.Compose([A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.3, 0.5), p=1.0)], p=1.0),
            "sharpen": A.Compose([A.Sharpen(alpha=(0.4, 0.6), lightness=(0.5, 1.0), p=1.0)], p=1.0),
            "noise": A.Compose([A.GaussNoise(var_limit=(10.0, 30.0), p=1.0)], p=1.0),
            "edge_enhance": A.Compose([A.ImageCompression(quality_lower=80, quality_upper=100, p=1.0), A.Sharpen(alpha=(0.3, 0.5), lightness=(0.7, 1.0), p=1.0)], p=1.0),
            "local_contrast": A.Compose([A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)], p=1.0),
            "detail": A.Compose([A.Sharpen(alpha=(0.3, 0.3), lightness=(0.7, 0.7), p=1.0), A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.15, 0.15), p=1.0)], p=1.0),
            "equalize": A.Compose([A.Equalize(p=1.0)], p=1.0),
            "enhanced_detail": A.Compose([A.Sharpen(alpha=(0.6, 0.6), lightness=(1.2, 1.2), p=1.0), A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0.3, 0.3), p=1.0), A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)], p=1.0),
            "enhanced_edges": A.Compose([A.Sharpen(alpha=(0.7, 0.7), lightness=(1.3, 1.3), p=1.0), A.CLAHE(clip_limit=6.0, tile_grid_size=(4, 4), p=1.0), A.RandomBrightnessContrast(brightness_limit=(0.15, 0.15), contrast_limit=(0.4, 0.4), p=1.0), A.ImageCompression(quality_lower=95, quality_upper=100, p=1.0)], p=1.0),
            "scratch_detector": A.Compose([A.Sharpen(alpha=(0.8, 0.8), lightness=(1.0, 1.0), p=1.0), A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0.5, 0.5), p=1.0), A.CLAHE(clip_limit=3.0, tile_grid_size=(3, 3), p=1.0), A.GaussNoise(var_limit=(5.0, 10.0), p=0.5)], p=1.0),
        }
        
        # Add enabled augmentations
        for aug_name in enabled_augs:
            if aug_name in aug_definitions:
                self.augmentations[aug_name] = aug_definitions[aug_name]

    def create_augmentations(self, image):
        """
        Create augmented versions of the input image.

        Args:
            image: OpenCV image (BGR format)

        Returns:
            List of (augmentation_name, augmented_image) tuples
        """
        # Convert BGR to RGB for Albumentations
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented_images = []

        # Apply each augmentation
        for aug_name, transform in self.augmentations.items():
            try:
                # Apply augmentation with fixed seed for reproducibility
                augmented = transform(image=image_rgb)
                aug_image_rgb = augmented['image']

                # Convert back to BGR for OpenCV
                aug_image_bgr = cv2.cvtColor(aug_image_rgb, cv2.COLOR_RGB2BGR)

                # Add to results
                augmented_images.append((aug_name, aug_image_bgr))
            except Exception as e:
                print(f"Error applying {aug_name} augmentation: {str(e)}")

        return augmented_images

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/augmentations', methods=['GET'])
def get_available_augmentations():
    """Get list of all available augmentations"""
    augmentations = [
        {"id": "grayscale", "name": "Grayscale", "description": "Convert to grayscale"},
        {"id": "hue_neg", "name": "Hue Negative", "description": "Decrease hue by 20"},
        {"id": "hue_pos", "name": "Hue Positive", "description": "Increase hue by 20"},
        {"id": "sat_neg", "name": "Saturation Negative", "description": "Decrease saturation"},
        {"id": "sat_pos", "name": "Saturation Positive", "description": "Increase saturation"},
        {"id": "brightness_neg", "name": "Brightness Negative", "description": "Decrease brightness"},
        {"id": "brightness_pos", "name": "Brightness Positive", "description": "Increase brightness"},
        {"id": "contrast_neg", "name": "Contrast Negative", "description": "Decrease contrast"},
        {"id": "contrast_pos", "name": "Contrast Positive", "description": "Increase contrast"},
        {"id": "sharpen", "name": "Sharpen", "description": "Apply sharpening filter"},
        {"id": "noise", "name": "Gaussian Noise", "description": "Add gaussian noise"},
        {"id": "edge_enhance", "name": "Edge Enhancement", "description": "Enhance edges"},
        {"id": "local_contrast", "name": "Local Contrast", "description": "CLAHE enhancement"},
        {"id": "detail", "name": "Detail Enhancement", "description": "Enhance fine details"},
        {"id": "equalize", "name": "Histogram Equalization", "description": "Equalize histogram"},
        {"id": "enhanced_detail", "name": "Enhanced Detail", "description": "Advanced detail enhancement"},
        {"id": "enhanced_edges", "name": "Enhanced Edges", "description": "Advanced edge enhancement"},
        {"id": "scratch_detector", "name": "Scratch Detector", "description": "Optimize for scratch detection"},
    ]
    return jsonify(augmentations)

@app.route('/api/augment', methods=['POST'])
def augment_image():
    """Process uploaded image with selected augmentations"""
    session_folder = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file provided'}), 400
        
        augmentations = request.form.getlist('augmentations')
        if not augmentations:
            return jsonify({'error': 'No augmentations selected'}), 400
        
        # Create a session folder for this request
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        session_folder = os.path.join(SESSIONS_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Save and load image
        filename = secure_filename(file.filename)
        filepath = os.path.join(session_folder, filename)
        file.save(filepath)
        image = cv2.imread(filepath)
        if image is None:
            shutil.rmtree(session_folder)
            return jsonify({'error': 'Could not read image file'}), 400
        
        augmenter = ImageAugmenter(augmentations)
        augmented_images = augmenter.create_augmentations(image)
        
        # Create folder to store images for zipping
        aug_folder = os.path.join(session_folder, "augmented")
        os.makedirs(aug_folder, exist_ok=True)
        
        name_without_ext, _ = os.path.splitext(filename)
        cv2.imwrite(os.path.join(aug_folder, f"{name_without_ext}_original.jpg"), image)
        for aug_name, aug_image in augmented_images:
            cv2.imwrite(os.path.join(aug_folder, f"{name_without_ext}_{aug_name}.jpg"), aug_image)
        
        # Prepare data for JSON response (base64 for display)
        results = [{'name': aug_name, 'image': image_to_base64(aug_image)} for aug_name, aug_image in augmented_images]
        original_b64 = image_to_base64(image)
        
        # Clean up the initial uploaded file, as it's now saved in the 'augmented' folder
        os.remove(filepath)
        
        return jsonify({
            'original': original_b64,
            'augmented': results,
            'session_id': session_id  # Crucial for the "Download All" button
        })
        
    except Exception as e:
        if session_folder and os.path.exists(session_folder):
            shutil.rmtree(session_folder)
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-zip/<session_id>', methods=['GET'])
def download_zip(session_id):
    """Creates a zip from a session's augmented images and sends it."""
    try:
        session_folder = os.path.join(SESSIONS_FOLDER, secure_filename(session_id))
        if not os.path.isdir(session_folder):
            return jsonify({'error': 'Session not found or has expired.'}), 404
        
        aug_folder = os.path.join(session_folder, "augmented")
        zip_filename = f"augmented_images_{session_id}.zip"
        zip_path = os.path.join(session_folder, zip_filename)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in os.listdir(aug_folder):
                zipf.write(os.path.join(aug_folder, file), file)

        @after_this_request
        def cleanup(response):
            try:
                shutil.rmtree(session_folder)
            except Exception as e:
                app.logger.error(f"Error cleaning up session folder {session_folder}: {e}")
            return response

        return send_file(zip_path, as_attachment=True, download_name=zip_filename)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-augment', methods=['POST'])
def batch_augment():
    """Process multiple images and return as downloadable zip"""
    session_folder = None
    try:
        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        augmentations = request.form.getlist('augmentations')
        if not augmentations:
            return jsonify({'error': 'No augmentations selected'}), 400
        
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        session_folder = os.path.join(SESSIONS_FOLDER, session_id)
        aug_folder = os.path.join(session_folder, "augmented")
        os.makedirs(aug_folder, exist_ok=True)
        
        augmenter = ImageAugmenter(augmentations)
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Read image directly from file stream
                filestr = file.read()
                npimg = np.frombuffer(filestr, np.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

                if image is None: continue
            
                name_without_ext = os.path.splitext(filename)[0]
                augmented_images = augmenter.create_augmentations(image)
                
                for aug_name, aug_image in augmented_images:
                    aug_img_path = os.path.join(aug_folder, f"{name_without_ext}_{aug_name}.jpg")
                    cv2.imwrite(aug_img_path, aug_image)
        
        zip_path = os.path.join(session_folder, f"augmented_images_{session_id}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
             for file in os.listdir(aug_folder):
                zipf.write(os.path.join(aug_folder, file), file)

        @after_this_request
        def cleanup(response):
            try:
                shutil.rmtree(session_folder)
            except Exception as e:
                app.logger.error(f"Error cleaning up batch session folder {session_folder}: {e}")
            return response

        return send_file(zip_path, as_attachment=True, download_name=f"augmented_images_{session_id}.zip")
        
    except Exception as e:
        if session_folder and os.path.exists(session_folder):
            shutil.rmtree(session_folder)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)