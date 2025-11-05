# MRI-Brain-Tumor-Detector-Classifier  
### Shuaib Siddiqui

## Overview of Project  
This project starts with a Jupyter Notebook (`tumor-classification-cnn.ipynb`) that preprocesses an MRI brain image dataset and explores why deep learning, especially CNNs, is effective for medical image classification problems. The notebook showcases building, tuning, and evaluating a CNN model optimized for brain tumor classification using MRI scans.

Deep learning was chosen due to its proven ability to manage complex image recognition tasks, with CNNs highly effective at identifying intricate patterns and features in medical imaging.

The project culminates in a Flask-based web application that uses the trained CNN model to deliver real-time brain tumor predictions on pre-loaded MRI images, demonstrating practical deployment and model capabilities in an accessible web interface.

Additionally, this project has been successfully deployed on both Render.com and Railway.app. The Railway deployment is faster and more reliable, with noticeably less downtime. Public URLs for both deployments are provided in the repository's description.

The dataset is included locally in the `brain_tumor_data` folder. However, should any users encounter issues accessing it, they can download the data directly themselves following the outlined steps.

A Dockerfile is also included, allowing you to build a Docker image effortlessly. This image can be run locally or deployed on any compatible cloud service, facilitating easy and consistent deployment.

## Features Created  
- **Jupyter Notebook**: Detailed CNN model development including preprocessing, training, and evaluation.  
- **Web Application**: Flask app serving real-time tumor classification predictions using the trained CNN.

## Technologies Used  
- Backend: Flask  
- Machine Learning: TensorFlow, Keras  
- Image Processing: PIL  
- Data Handling: Pandas, NumPy  
- Visualization: Matplotlib, Seaborn  
- Others: Logging, Exception Handling  

## Data Preprocessing  
- **Data Acquisition**: MRI images and labels sourced from [Kaggle dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).  
- **Augmentation**: Image augmentations such as rotation, zoom, and flip using TensorFlow's `ImageDataGenerator`.  
- **Normalization & Resizing**: Standardizing image sizes and pixel values.  
- **Train-Test Split**: For effective evaluation.

## Model Structure  
- Multiple `Conv2D` layers (128 and 256 filters).  
- `MaxPooling2D` layers to reduce spatial dimensions.  
- Dense output with softmax activation for multi-class classification.  
- Adamax optimizer and categorical cross-entropy loss.  
- 3.76 million parameters (~14.36 MB).

## Model Evaluation  
- Assessed with accuracy, loss, confusion matrix, and classification reports across datasets.

## Model Predictions  
- Display of test image predictions with probabilities.

## Saving & Loading Model  
- Model saved in TensorFlow/Keras format.

## Deployment & Performance  
- Deployed on Render.com and Railway.app, with Railway providing better speed and uptime.  
- Deployment URLs provided in repo description.

## Running the Notebook & Web Application Locally  

### Prerequisites  
- Python 3.11  
- pip  
- Jupyter Notebook  
- Internet for dataset download (if needed)

### Dataset Placement  
Dataset is located within `brain_tumor_data`. If not available, users can download it from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

### Setup and Run  

```bash
# Clone the repository
git clone https://github.com/shuaibsiddiqui786/MRI-Brain-Tumor-Detector-Classifier.git

# Change to project directory
cd MRI-Brain-Tumor-Detection

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
```

Open a browser and navigate to http://127.0.0.1:5000 to access the application.

## Building and Running Docker Container Locally  

```bash
# Build Docker image
docker build -t mri-brain-tumor-app .

# Run container and map port to localhost
docker run -p 5000:5000 mri-brain-tumor-app
```

Visit http://localhost:5000 to use the containerized app.

## Deploying Using Docker  

The Dockerfile provided enables building and running a Docker image suitable for deployment on any Docker-supported platform, ensuring environment consistency and simplified deployment.

---

Feel free to ask for further help or additional instructions!