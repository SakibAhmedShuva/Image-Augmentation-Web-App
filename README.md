# 🎨 AI Image Augmentation Studio

<div align="center">

**A powerful, all-in-one web application for generating augmented image datasets**

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-f7df1e?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://html.spec.whatwg.org/)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://www.w3.org/Style/CSS/)

*Built with Flask backend and dynamic vanilla JavaScript frontend, specifically tailored for damage detection and analysis tasks.*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-reference) • [Contributing](#-contributing)

</div>

---

## 🌟 Features

### 🔄 **Dual Processing Modes**
- **Single Image Mode**: Quick previews and testing with instant results
- **Batch Processing Mode**: Generate complete datasets with ZIP download

### 🎯 **Rich Augmentation Library**
Powered by **Albumentations**, featuring:
- **Color Space Adjustments**: Grayscale, Hue, Saturation variations
- **Brightness & Contrast**: Dynamic lighting modifications
- **Advanced Filtering**: Sharpen, Gaussian Noise, Histogram Equalization
- **Detail Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Edge Processing**: Specialized techniques for damage detection

### 💻 **Intuitive User Experience**
- **Drag & Drop Interface**: Seamless file upload experience
- **Real-time Preview**: Instant visualization in single mode
- **Responsive Design**: Works perfectly on desktop and mobile
- **Progress Tracking**: Visual feedback during processing

### 🚀 **Production Ready**
- **RESTful API**: Complete programmatic access
- **Scalable Architecture**: Flask-based backend with efficient processing
- **Error Handling**: Robust error management and user feedback

---

## 🎯 Demo

### Single Image Processing

![image](https://github.com/user-attachments/assets/2e24cbe8-c2d5-4841-958f-a8e0db4ddfe8)


![image](https://github.com/user-attachments/assets/bb8176f5-89cf-49d2-80c8-d51c2b1c339e)


![image](https://github.com/user-attachments/assets/c0067f96-89f3-400a-a362-8796aba15b6e)


---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center"><strong>Backend</strong></td>
<td align="center"><strong>Frontend</strong></td>
<td align="center"><strong>Image Processing</strong></td>
</tr>
<tr>
<td>
• Python 3.8+<br>
• Flask 2.x<br>
• Flask-CORS<br>
• python-dotenv
</td>
<td>
• HTML5<br>
• CSS3<br>
• Vanilla JavaScript (ES6+)<br>
• Responsive Design
</td>
<td>
• Albumentations<br>
• OpenCV<br>
• NumPy<br>
• Pillow (PIL)
</td>
</tr>
</table>

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/SakibAhmedShuva/Image-Augmentation-Web-App.git
cd Image-Augmentation-Web-App
```

### Step 2: Create Virtual Environment
**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary>📋 View requirements.txt contents</summary>

```
Flask>=2.0.0
Flask-Cors>=3.0.0
opencv-python-headless>=4.5.0
numpy>=1.21.0
Pillow>=8.3.0
albumentations>=1.3.0
python-dotenv>=0.19.0
```
</details>

### Step 4: Launch Application
```bash
python app.py
```

🎉 **Success!** Open your browser and navigate to `http://127.0.0.1:5000`

---

## 🚀 Usage

### Quick Start Guide

1. **📂 Select Processing Mode**
   - Choose **Single Image** for quick testing
   - Choose **Batch Processing** for dataset generation

2. **📤 Upload Your Images**
   - Drag and drop files into the upload area
   - Or click to browse and select files
   - Supported formats: JPG, JPEG, PNG, BMP

3. **🎛️ Configure Augmentations**
   - Click on augmentation cards to select techniques
   - Selected cards are highlighted for easy identification
   - Mix and match different augmentation types

4. **⚡ Generate Results**
   - Click **Generate** to start processing
   - Watch the progress indicator
   - Results appear automatically when ready

5. **💾 Download & Save**
   - **Single Mode**: Download individual augmented images
   - **Batch Mode**: Automatic ZIP file download with all results

---

## 📁 Project Structure

```
Image-Augmentation-Web-App/
├── 📄 app.py                    # Main Flask application
├── 🌐 index.html               # Frontend SPA
├── 📂 sessions/                # Batch processing workspace
├── 📂 uploads/                 # Single image uploads
├── ⚙️ requirements.txt         # Python dependencies  
├── 🔧 .env.example            # Environment variables template
└── 📖 README.md               # Project documentation
```

---

## 🔌 API Reference

### Base URL
```
http://127.0.0.1:5000
```

### Endpoints

#### `GET /`
**Description**: Serves the main application interface
- **Response**: HTML page

#### `GET /api/augmentations`
**Description**: Retrieve available augmentation techniques
- **Response**: JSON array of augmentation objects
- **Example**:
```json
[
  {
    "id": "grayscale",
    "name": "Grayscale",
    "description": "Convert image to grayscale"
  }
]
```

#### `POST /api/augment`
**Description**: Process single image with selected augmentations
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image`: Image file (required)
  - `augmentations[]`: Array of augmentation IDs (required)
- **Response**: JSON with Base64 encoded images

#### `POST /api/batch-augment`
**Description**: Process multiple images for dataset creation
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `images[]`: Array of image files (required)
  - `augmentations[]`: Array of augmentation IDs (required)
- **Response**: ZIP file download

---

## 💡 Advanced Features

### Environment Configuration
Create a `.env` file for custom settings:
```env
FLASK_ENV=development
UPLOAD_FOLDER=uploads
SESSION_FOLDER=sessions
MAX_CONTENT_LENGTH=16777216  # 16MB
```

### Custom Augmentations
Extend the augmentation library by modifying the augmentation definitions in `app.py`:
```python
# Add new augmentation
custom_augmentations = {
    'custom_blur': A.Blur(blur_limit=7, p=1.0),
    'custom_noise': A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
}
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🐛 Bug Reports
- Use the [Issue Tracker](https://github.com/SakibAhmedShuva/Image-Augmentation-Web-App/issues)
- Include detailed reproduction steps
- Provide system information and error logs

### 💡 Feature Requests
- Open an issue with the `enhancement` label  
- Describe the feature and its benefits
- Include mockups or examples if applicable

### 🔧 Pull Requests
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Submit** a Pull Request

### 📋 Development Guidelines
- Follow PEP 8 for Python code
- Use meaningful commit messages
- Add tests for new features
- Update documentation as needed

---

## 🏆 Roadmap

- [ ] **Multi-format Support**: Add support for TIFF, WebP formats
- [ ] **Cloud Integration**: AWS S3, Google Cloud Storage support
- [ ] **Advanced Filters**: Custom kernel filters and transformations
- [ ] **Batch Configuration**: Save and load augmentation presets
- [ ] **Performance Optimization**: GPU acceleration support
- [ ] **API Authentication**: Secure API access with tokens

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License - Copyright (c) 2024 AI Image Augmentation Studio
```

---

## 🙏 Acknowledgments

- **[Albumentations](https://albumentations.ai/)** - Fast image augmentation library
- **[OpenCV](https://opencv.org/)** - Computer vision and image processing
- **[Flask](https://flask.palletsprojects.com/)** - Lightweight web framework
- **Community Contributors** - Thank you for your valuable contributions!

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

*Made with ❤️ by the AI Image Augmentation Studio team*

</div>
