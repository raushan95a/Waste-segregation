# Waste Detection & Segregation - Project Setup Guide

## 📋 Project Overview

This project implements **automated waste detection and segregation** using deep learning object detection. It uses PyTorch with RetinaNet architecture to identify and classify waste materials (wet and dry waste) in images.

### Key Features:
- **Object Detection Model**: PyTorch RetinaNet
- **Dataset**: 800 annotated waste images
- **Annotation Format**: Pascal VOC format (converted to COCO)
- **Framework**: Monk Object Detection Library
- **Execution Environment**: Google Colab or Local System with GPU/CPU

---

## 🧭 Project Info

- **Repository**: https://github.com/raushan95a/Waste-segregation
- **Dataset Location**: `My Drive/Waste_Dataset/` (Google Drive)
- **Note**: The dataset is not stored in the GitHub repo.

---

## 🗂️ Project Structure

```
Waste-segregation/
├── Waste_Detection(executed).ipynb          # Main workflow notebook (RECOMMENDED)
├── Wet and dry waste detection.ipynb        # Data preprocessing notebook
├── Copy_of_Waste_Detection.ipynb           # Backup of main notebook
├── Important links/                         # Reference links
├── .git/                                   # Git repository
└── SETUP_GUIDE.md                          # This file
```

### Notebook Descriptions:

| Notebook | Purpose | Status |
|----------|---------|--------|
| **Waste_Detection(executed).ipynb** | Complete workflow: data processing, model training, and inference | ✅ MAIN |
| **Wet and dry waste detection.ipynb** | Detailed data annotation conversion pipeline | 📚 Reference |
| **Copy_of_Waste_Detection.ipynb** | Backup copy of main notebook | 🔄 Backup |

---

## 📦 System Requirements

### Hardware Requirements:
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: At least 10GB free space for dataset and models

### Software Requirements:
- Python 3.7+
- CUDA 10.2+ (if using GPU)
- cuDNN 7.6+ (if using GPU)
- pip or conda package manager

---

## 🚀 Installation & Setup

### ✅ Quick Start (Colab)

1. Upload your dataset to Google Drive at:
   - `My Drive/Waste_Dataset/Images_merged/`
   - `My Drive/Waste_Dataset/Annotations_merged/`
2. Open the main notebook:
   - https://colab.research.google.com/github/raushan95a/Waste-segregation/blob/main/Waste_Detection%28executed%29.ipynb
3. Set runtime to GPU:
   - `Runtime → Change runtime type → GPU`
4. Run all cells top to bottom.

---

### Option A: Google Colab (Recommended for Beginners)

Google Colab provides free GPU access. Follow these steps:

#### Step 1: Prepare Your Dataset
1. Create a Google Drive folder named `My Drive/Waste_Dataset/`
2. Create subdirectories:
   - `Images_merged/` - Contains all 800 waste images
   - `Annotations_merged/` - Contains all 800 XML annotation files (Pascal VOC format)

#### Step 2: Access the Notebook
1. Open the notebook on Colab:
   - Navigate to: [Waste_Detection Colab Notebook](https://colab.research.google.com/github/raushan95a/Waste-segregation/blob/main/Waste_Detection%28executed%29.ipynb)
   - Click "Open in Colab" button

#### Step 3: Configure Runtime
```
Runtime → Change Runtime Type → Select GPU
```

#### Step 4: Run the Notebook
- Execute cells sequentially from top to bottom
- The notebook will automatically:
  - Mount your Google Drive
  - Clone Monk Object Detection repository
  - Install required dependencies
  - Process dataset annotations
  - Train the model
  - Perform inference on test images

---

### Option B: Local System Setup

#### Step 1: Install Python & CUDA

**For Windows:**
```powershell
# Install Python 3.8 or higher from python.org
# Install CUDA 11.0+ from NVIDIA website
# Verify installation
python --version
nvcc --version
```

**For Linux/Mac:**
```bash
# Using conda (recommended)
conda create -n waste_detection python=3.8
conda activate waste_detection
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

#### Step 2: Install Dependencies

```bash
# Clone the project
git clone https://github.com/raushan95a/Waste-segregation.git
cd Waste-segregation

# Install required Python packages
pip install torch torchvision torchaudio
pip install numpy pandas opencv-python
pip install xmltodict
pip install pycocotools
pip install tqdm
pip install Pillow
pip install matplotlib
```

**For Local Systems with RetinaNet:**
```bash
# Clone Monk Object Detection
git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git

# Install RetinaNet requirements
cd Monk_Object_Detection/5_pytorch_retinanet/installation

# Choose the appropriate requirements file:
# - requirements.txt (CPU)
# - requirements_cuda.txt (GPU)
# - requirements_colab.txt (Google Colab)

cat requirements_cuda.txt | xargs -n 1 -L 1 pip install
```

#### Step 3: Prepare Dataset

```
Create the following directory structure:
Waste_Dataset/
├── Images_merged/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (800 images total)
└── Annotations_merged/
    ├── image1.xml
    ├── image2.xml
    └── ... (800 XML files total)
```

---

## 📊 Data Processing Pipeline

### Step 1: Data Annotation Format Conversion

**Input**: Pascal VOC XML format annotations
```xml
<annotation>
  <filename>image.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
  </size>
  <object>
    <name>waste_type</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
</annotation>
```

**Conversion Process**:
```
Pascal VOC (XML) → Monk Format → COCO Format (JSON)
```

**Output Files Generated**:
- `train_labels.csv` - Monk format labels
- `Waste_Dataset/annotations/instances_Imagesmerged.json` - COCO format annotations
- `Waste_Dataset/annotations/classes.txt` - Class names

### Step 2: Dataset Statistics

- **Total Images**: 800
- **Total Annotations**: ~1200 (multiple objects per image)
- **Image Format**: JPG
- **Classes**: Wet waste, Dry waste (and subcategories)
- **Train/Test Split**: Typically 80/20

---

## 🧠 Model Training

### Architecture: RetinaNet with ResNet50 Backbone

#### Training Configuration:

```python
# Model Setup
Model: RetinaNet
Backbone: ResNet50
Input Size: Variable (optimized internally)
Output: Bounding boxes + Class predictions

# Hyperparameters
Learning Rate: 0.0001
Optimizer: Adam/SGD
Batch Size: 2-4 (adjust based on GPU memory)
Epochs: 8 (minimum) to 20+ (for better accuracy)
Loss Function: Focal Loss (handles class imbalance)
```

#### Training Steps in Notebook:

1. **Load Dataset**
   ```python
   gtf.Train_Dataset(root_dir, coco_dir, img_dir, set_dir, batch_size=2)
   ```

2. **Initialize Model**
   ```python
   gtf.Model(model_name="resnet50")
   ```

3. **Set Hyperparameters**
   ```python
   gtf.Set_Hyperparams(lr=0.0001, print_interval=20)
   ```

4. **Train Model**
   ```python
   gtf.Train(num_epochs=8, output_model_name="final_model.pt")
   ```

#### Expected Output:
- `final_model.pt` - Trained model weights
- Training logs with loss values
- Validation metrics

---

## 🔍 Model Inference

### Step 1: Load Pre-trained Model

```python
from infer_detector import Infer
gtf = Infer()
gtf.Model(model_path="final_model.pt")
```

### Step 2: Perform Predictions

```python
img_path = "Waste_Dataset/Images_merged/test_image.jpg"
scores, labels, boxes = gtf.Predict(
    img_path, 
    class_list, 
    vis_threshold=0.4
)
```

### Step 3: Interpret Results

- **Boxes**: Bounding box coordinates [x, y, width, height]
- **Scores**: Confidence scores (0.0 - 1.0)
- **Labels**: Predicted class names (waste types)
- **Output**: Visualized image with predictions saved as `output.jpg`

---

## 🔧 Common Issues & Troubleshooting

### Issue 1: Module Not Found Errors
**Solution**: Install missing packages using pip
```bash
pip install <module_name>
# or for conda
conda install <module_name>
```

### Issue 2: CUDA Out of Memory
**Solution**: 
- Reduce batch size in training
- Use CPU mode (slower but uses less memory)
- Optimize image size

### Issue 3: Dataset Path Errors
**Solution**:
- Verify directory structure matches notebook expectations
- Update path variables in notebook cells:
  ```python
  root_dir = "path/to/Waste_Dataset/"
  img_dir = "Images_merged/"
  anno_dir = "Annotations_merged/"
  ```

### Issue 4: Poor Model Accuracy
**Solution**:
- Increase number of training epochs
- Adjust learning rate
- Check dataset quality and annotations
- Add data augmentation

---

## 📈 Expected Results

### Training Metrics:
- Initial Loss: ~2.5-3.0
- Final Loss: ~0.1-0.5
- Training Time: 2-4 hours per epoch (GPU dependent)

### Inference Performance:
- Inference Speed: 10-50 ms per image (GPU)
- Accuracy: 75-90% (depends on model training)
- Supported Class Detection: Wet waste, Dry waste

### Sample Predictions:
- Plastic waste detection with 85%+ confidence
- Metal waste classification
- Organic waste identification
- Multiple objects per image support

---

## 📚 Key Libraries & Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| torch | 1.9+ | Deep learning framework |
| torchvision | 0.10+ | Computer vision utilities |
| numpy | 1.19+ | Numerical computing |
| pandas | 1.1+ | Data manipulation |
| opencv-python | 4.5+ | Image processing |
| xmltodict | Latest | XML parsing |
| pycocotools | Latest | COCO dataset tools |

---

## 🎯 Usage Workflow

### Complete Workflow for First-Time Users:

```
1. Prepare Dataset
   └─ Collect 800 waste images
   └─ Create Pascal VOC annotations
   └─ Organize in Waste_Dataset/ folder

2. Setup Environment (Choose One)
   ├─ Option A: Google Colab (recommended)
   └─ Option B: Local system

3. Run Data Processing
   └─ Convert VOC → COCO format
   └─ Generate training labels

4. Train Model
   └─ Load dataset
   └─ Initialize RetinaNet model
   └─ Train for 8+ epochs

5. Evaluate & Inference
   └─ Test on sample images
   └─ Visualize predictions
   └─ Measure accuracy

6. Deployment (Optional)
   └─ Save trained model
   └─ Create inference API
   └─ Integrate with applications
```

---

## 🔗 External Resources

### Official Repositories:
- [Monk Object Detection](https://github.com/Tessellate-Imaging/Monk_Object_Detection)

### Documentation:
- [PyTorch Documentation](https://pytorch.org/docs/)
- [COCO Dataset Format](https://cocodataset.org/)
- [Pascal VOC Format](http://host.robots.ox.ac.uk/pascal/VOC/)

### Learning Resources:
- [Object Detection Basics](https://towardsdatascience.com/object-detection-in-6-steps-6907b996bd34)
- [RetinaNet Paper](https://arxiv.org/abs/1708.02002)
- [COCO Detection Challenge](https://cocodataset.org/detection-leaderboard/)

---

## 📝 Tips for Best Results

1. **Data Quality**
   - Ensure annotations are accurate and complete
   - Remove corrupted or low-quality images
   - Balance classes in dataset

2. **Model Training**
   - Start with fewer epochs and gradually increase
   - Monitor validation metrics
   - Use learning rate scheduling
   - Save best model checkpoints

3. **Hyperparameter Tuning**
   - Lower learning rate for fine-tuning
   - Adjust batch size based on GPU memory
   - Use data augmentation to prevent overfitting

4. **Inference Optimization**
   - Use appropriate confidence threshold (0.3-0.5)
   - Optimize image preprocessing
   - Consider model quantization for deployment

---

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make improvements
4. Submit a pull request

---

## 📄 License

This project follows the same license as the original repository.

---

## ✉️ Support & Contact

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues first
- Provide detailed error messages and logs

---

**Last Updated**: May 30, 2026  
**Version**: 1.0  
**Status**: Active
