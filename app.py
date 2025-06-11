from flask import Flask, request, jsonify, send_file, render_template
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
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get selected augmentations
        augmentations = request.form.getlist('augmentations')
        if not augmentations:
            return jsonify({'error': 'No augmentations selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not read image file'}), 400
        
        # Create augmenter with selected augmentations
        augmenter = ImageAugmenter(augmentations)
        
        # Generate augmentations
        augmented_images = augmenter.create_augmentations(image)
        
        # Convert images to base64 for web display
        results = []
        original_b64 = image_to_base64(image)
        
        for aug_name, aug_image in augmented_images:
            aug_b64 = image_to_base64(aug_image)
            results.append({
                'name': aug_name,
                'image': aug_b64
            })
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'original': original_b64,
            'augmented': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-augment', methods=['POST'])
def batch_augment():
    """Process multiple images and return as downloadable zip"""
    try:
        # Check if files are present
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Get selected augmentations
        augmentations = request.form.getlist('augmentations')
        if not augmentations:
            return jsonify({'error': 'No augmentations selected'}), 400
        
        # Create session folder
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        session_folder = os.path.join(SESSIONS_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Save uploaded files
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_folder, filename)
                file.save(filepath)
                uploaded_files.append(filepath)
        
        if not uploaded_files:
            return jsonify({'error': 'No valid image files found'}), 400
        
        # Create augmenter
        augmenter = ImageAugmenter(augmentations)
        
        # Create augmented folder
        aug_folder = os.path.join(session_folder, "augmented")
        os.makedirs(aug_folder, exist_ok=True)
        
        # Process each image
        for img_path in uploaded_files:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            base_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Generate augmentations
            augmented_images = augmenter.create_augmentations(image)
            
            for aug_name, aug_image in augmented_images:
                aug_img_path = os.path.join(aug_folder, f"{name_without_ext}_{aug_name}.jpg")
                cv2.imwrite(aug_img_path, aug_image)
        
        # Create zip file
        zip_path = os.path.join(session_folder, f"augmented_images_{session_id}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(aug_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, aug_folder)
                    zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name=f"augmented_images_{session_id}.zip")
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)