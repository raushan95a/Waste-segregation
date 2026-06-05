# Waste Detection and Segregation

This project detects waste objects in images and provides a simple web interface for testing the model. It uses a PyTorch RetinaNet object detection model trained through the Monk Object Detection library. The app can accept an uploaded image or a camera capture, run detection, draw bounding boxes, and show prediction details in the browser.

## Project Overview

The project has two main parts:

1. Model training and experimentation through Jupyter notebooks.
2. A FastAPI web app for image upload, camera capture, and inference.

The trained model file is `final_model.pt`. The current class file at `Waste_Dataset/annotations/classes.txt` contains the detector class `Garbage`. In the web app, wet or dry waste is estimated separately after object detection using filename keywords and simple image color analysis.

## Key Features

- PyTorch RetinaNet object detection model.
- ResNet50 backbone.
- Monk Object Detection library integration.
- Pascal VOC XML annotations converted to COCO format.
- Local FastAPI backend.
- Browser UI for image upload.
- Browser camera capture support.
- Result image with bounding boxes.
- Confidence threshold control.
- Google Colab workflow for training.
- Local workflow for training and deployment.

## Repository and Dataset

- Repository: `https://github.com/raushan95a/Waste-segregation`
- Dataset Google Drive folder: `https://drive.google.com/drive/folders/1VJE5qj9DjZ9rZy6JT_-AIWVTwgbnRLDl`
- Older setup guide note: the dataset may also be expected in Google Drive at `My Drive/Waste_Dataset/`.

The notebooks can use a local `Waste_Dataset/` folder or download the dataset from Google Drive, depending on the notebook version and environment.

## Project Structure

```text
Waste-segregation/
|-- app.py
|-- web/
|   `-- index.html
|-- Waste_Detection(executed).ipynb
|-- Wet and dry waste detection.ipynb
|-- Copy_of_Waste_Detection.ipynb
|-- SETUP_GUIDE.md
|-- requirements-deploy.txt
|-- final_model.pt
|-- resume.pt
|-- resnet50-19c8e357.pth
|-- Waste_Dataset/
|   |-- Images_merged/
|   |-- Annotations_merged/
|   |-- annotations/
|   |   |-- classes.txt
|   |   `-- instances_Images_merged.json
|   |-- train_labels.csv
|   `-- waste_type_labels.csv
|-- Monk_Object_Detection/
|-- uploads/
|-- Inference/
`-- .venv/
```

## Important Files and Folders

### `app.py`

This is the FastAPI backend. It does the following:

- Defines the API server.
- Loads `final_model.pt` on startup.
- Reads class names from `Waste_Dataset/annotations/classes.txt`.
- Accepts image uploads through the `/predict` endpoint.
- Preprocesses images for RetinaNet.
- Runs model inference.
- Estimates wet or dry waste type.
- Draws bounding boxes on the image.
- Saves result images inside `Inference/`.
- Serves result images through `/result/{filename}`.
- Serves the browser UI at `/`.

### `web/index.html`

This is the frontend. It does the following:

- Shows the upload area.
- Allows image drag-and-drop.
- Allows camera access through the browser.
- Captures a frame from the camera.
- Lets the user adjust confidence threshold.
- Sends the selected image to `/predict`.
- Displays the returned result image.
- Shows detection count, image size, confidence, class, and bounding box details.

### `final_model.pt`

This is the trained model used by the web app. Keep this file if you want to run prediction locally.

### `resume.pt`

This is most likely a training checkpoint. It is useful if you want to resume training. It is not required for running the current web app if `final_model.pt` is available.

### `resnet50-19c8e357.pth`

This is a pretrained ResNet50 weight file. It is useful during model training. It is not required for normal web app inference if the final model already loads correctly.

### `Waste_Dataset/`

This folder contains the training data and generated annotation files.

Expected dataset structure:

```text
Waste_Dataset/
|-- Images_merged/
|   |-- image1.jpg
|   |-- image2.jpg
|   `-- ...
|-- Annotations_merged/
|   |-- image1.xml
|   |-- image2.xml
|   `-- ...
`-- annotations/
    |-- classes.txt
    `-- instances_Images_merged.json
```

Current local dataset summary:

- `Images_merged/`: waste images.
- `Annotations_merged/`: Pascal VOC XML annotation files.
- `annotations/classes.txt`: class names used by the detector.
- `annotations/instances_Images_merged.json`: COCO-style annotation file.
- `train_labels.csv`: Monk-style labels.
- `waste_type_labels.csv`: extra labels used by the project.

### `Monk_Object_Detection/`

This is a large third-party library used for training and inference support. Your app mainly depends on:

```text
Monk_Object_Detection/5_pytorch_retinanet/lib
```

Do not delete this folder unless the app is rewritten to remove the dependency.

### `uploads/`

Uploaded or camera-captured images are saved here when the web app runs. This is runtime output and can be deleted when you no longer need old uploads.

### `Inference/`

Prediction result images are saved here. This is runtime output and can be deleted when you no longer need old results.

### `.venv/`

This is the local Python virtual environment. It contains installed packages. It is not source code and should not be committed.

## Notebook Descriptions

| Notebook | Purpose | Status |
| --- | --- | --- |
| `Waste_Detection(executed).ipynb` | Complete workflow for data processing, training, and inference | Main notebook |
| `Wet and dry waste detection.ipynb` | Data preprocessing and annotation conversion reference | Reference |
| `Copy_of_Waste_Detection.ipynb` | Backup copy of the main workflow | Backup |

## System Requirements

### For Running the Web App

- Python 3.7 or newer.
- At least 4 GB RAM, preferably more.
- CPU is supported.
- GPU is optional for faster inference.
- `final_model.pt` must be present.
- `Monk_Object_Detection/` must be present.
- `Waste_Dataset/annotations/classes.txt` should be present.

### For Training the Model

- Python 3.7 or newer.
- NVIDIA GPU with CUDA support is recommended.
- 8 GB RAM minimum.
- 16 GB RAM or more recommended.
- At least 10 GB free disk space.
- CUDA and cuDNN if training on GPU.
- Google Colab GPU can be used instead of local GPU.

## Web App Quick Start

Use this when `final_model.pt`, `Monk_Object_Detection/`, and `Waste_Dataset/annotations/classes.txt` are already available.

```powershell
cd c:\Users\raush\project\Waste-segregation
.venv\Scripts\Activate.ps1
pip install -r requirements-deploy.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## How to Use the Web App

1. Start the FastAPI server.
2. Open `http://127.0.0.1:8000`.
3. Upload an image or click `Use Camera`.
4. If using the camera, allow camera permission.
5. Capture an image.
6. Adjust the confidence threshold if needed.
7. Click `Analyze Image`.
8. View the marked result image and detection table.

Camera access works on `localhost`, `127.0.0.1`, or HTTPS. Most browsers block camera access when opening `index.html` directly as a local file.

## API Endpoints

### `GET /`

Returns the web UI from `web/index.html`.

### `GET /api`

Returns basic API information:

- API status message.
- Model filename.
- Device being used, such as CPU or CUDA.
- Loaded class list.
- API docs path.

### `POST /predict`

Accepts an image file and returns prediction results.

Request:

- Form field: `file`
- Query parameter: `threshold`
- Example threshold: `0.4`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.4" -F "file=@sample.jpg"
```

Response includes:

- Original filename.
- Confidence threshold.
- Image width and height.
- Estimated waste type.
- Detection list.
- Detection count.
- Class counts.
- Best detection.
- Result image URL.

### `GET /result/{filename}`

Returns the generated result image from the `Inference/` folder.

## Deployment Dependencies

The web app dependencies are listed in `requirements-deploy.txt`:

```text
fastapi
uvicorn
python-multipart
torch
torchvision
opencv-python
numpy
scikit-image
```

Install them with:

```powershell
pip install -r requirements-deploy.txt
```

## Google Colab Quick Start

Use Colab if you want to train or rerun the full notebook workflow without setting up a local GPU.

1. Open the notebook:

```text
https://colab.research.google.com/github/raushan95a/Waste-segregation/blob/main/Waste_Detection%28executed%29.ipynb
```

2. Set runtime to GPU:

```text
Runtime -> Change runtime type -> GPU
```

3. Run all cells from top to bottom.

Depending on the notebook version, it may:

- Mount Google Drive.
- Download the dataset from the Google Drive folder.
- Use `My Drive/Waste_Dataset/`.
- Clone Monk Object Detection.
- Install dependencies.
- Convert annotations.
- Train the model.
- Run inference.

## Local Training Setup

### Windows

```powershell
cd c:\Users\raush\project\Waste-segregation
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio
pip install jupyter notebook pandas numpy opencv-python xmltodict pycocotools tqdm Pillow matplotlib
```

If training with CUDA, install the PyTorch build that matches your CUDA version. Example:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If `Monk_Object_Detection/` is missing:

```powershell
git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git
```

Start Jupyter:

```powershell
jupyter notebook
```

Then open the notebook in the browser and run cells top to bottom.

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install jupyter notebook pandas numpy opencv-python xmltodict pycocotools tqdm Pillow matplotlib
git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git
jupyter notebook
```

## Data Processing Pipeline

The dataset begins with images and Pascal VOC XML annotations.

Example Pascal VOC annotation:

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

The conversion flow is:

```text
Pascal VOC XML -> Monk CSV format -> COCO JSON format
```

Generated files include:

- `Waste_Dataset/train_labels.csv`
- `Waste_Dataset/annotations/classes.txt`
- `Waste_Dataset/annotations/instances_Images_merged.json`

## Model Training

The training workflow uses RetinaNet with a ResNet50 backbone.

Typical configuration:

```python
model = "RetinaNet"
backbone = "ResNet50"
learning_rate = 0.0001
batch_size = 2
epochs = 8
loss = "Focal Loss"
```

Typical notebook flow:

```python
gtf.Train_Dataset(root_dir, coco_dir, img_dir, set_dir, batch_size=2)
gtf.Model(model_name="resnet50")
gtf.Set_Hyperparams(lr=0.0001, print_interval=20)
gtf.Train(num_epochs=8, output_model_name="final_model.pt")
```

Training output:

- `final_model.pt`
- Training logs.
- Optional checkpoint files such as `resume.pt`.
- Inference test outputs.

## Model Inference

Notebook-style inference usually follows this pattern:

```python
from infer_detector import Infer

gtf = Infer()
gtf.Model(model_path="final_model.pt")

scores, labels, boxes = gtf.Predict(
    "Waste_Dataset/Images_merged/test_image.jpg",
    class_list,
    vis_threshold=0.4
)
```

In the web app, `app.py` handles the inference directly. It preprocesses the image, calls the loaded model, filters detections by threshold, draws boxes, and returns a JSON response.

## Wet and Dry Waste Estimation

The current app has two stages:

1. Object detection: the model detects waste objects.
2. Segregation estimate: the app guesses `wet_waste` or `dry_waste`.

The segregation estimate is based on:

- Filename keywords such as `plastic`, `paper`, `food`, `organic`, or `leaf`.
- Basic HSV color analysis for green, brown, vivid, or bright neutral pixels.

This means the current wet/dry result is a heuristic estimate, not necessarily a separately trained wet/dry classifier.

## Expected Results

Training and inference results depend on dataset quality, GPU, epochs, and threshold.

Common expectations from the original setup guides:

- Training can take several hours on GPU.
- More epochs may improve results.
- Lower confidence thresholds show more detections but may include false positives.
- Higher confidence thresholds show fewer detections but are stricter.
- Suggested inference threshold range: `0.3` to `0.5`.

## Common Issues and Fixes

### `ModuleNotFoundError`

Install the missing package:

```bash
pip install package-name
```

For the web app, start with:

```powershell
pip install -r requirements-deploy.txt
```

### Model File Not Found

Make sure this file exists:

```text
final_model.pt
```

It should be in the project root, next to `app.py`.

### Monk Library Import Error

Make sure this folder exists:

```text
Monk_Object_Detection/5_pytorch_retinanet/lib
```

`app.py` adds this path to Python automatically.

### Dataset Path Error

Check that the dataset follows this structure:

```text
Waste_Dataset/
|-- Images_merged/
|-- Annotations_merged/
`-- annotations/
```

If using Colab, check whether the notebook expects the dataset at:

```text
My Drive/Waste_Dataset/
```

### CUDA Out of Memory

Try:

- Reduce batch size.
- Use fewer images while testing.
- Use CPU mode for small tests.
- Resize images.
- Restart the runtime or Python process.

### Poor Accuracy

Try:

- Check annotation quality.
- Increase training epochs.
- Balance classes.
- Add more examples.
- Tune learning rate.
- Use data augmentation.
- Test different confidence thresholds.

### Camera Not Working

Use:

```text
http://127.0.0.1:8000
```

Do not open `web/index.html` directly from the filesystem for camera testing. Browser camera permissions generally require localhost or HTTPS.

## Files That Can Usually Be Ignored

These are generated or local-only:

```text
.ipynb_checkpoints/
__pycache__/
uploads/
Inference/
*.log
.venv/
```

These are useful for training but not always required for running the web app:

```text
resume.pt
resnet50-19c8e357.pth
Copy_of_Waste_Detection.ipynb
Wet and dry waste detection.ipynb
Waste_Dataset/
```

Do not remove `Waste_Dataset/annotations/classes.txt` unless you also update `app.py`, because the app reads class names from that file.

## Recommended Workflow

For only using the app:

```text
1. Keep app.py, web/index.html, final_model.pt, requirements-deploy.txt, Monk_Object_Detection/, and classes.txt.
2. Install requirements.
3. Run uvicorn.
4. Open the browser app.
5. Upload or capture an image.
```

For retraining:

```text
1. Prepare Waste_Dataset/Images_merged/.
2. Prepare Waste_Dataset/Annotations_merged/.
3. Open the main notebook.
4. Convert annotations.
5. Train RetinaNet.
6. Save final_model.pt.
7. Test in the web app.
```

## External Resources

- Monk Object Detection: `https://github.com/Tessellate-Imaging/Monk_Object_Detection`
- PyTorch documentation: `https://pytorch.org/docs/`
- COCO dataset format: `https://cocodataset.org/`
- Pascal VOC format: `http://host.robots.ox.ac.uk/pascal/VOC/`
- RetinaNet paper: `https://arxiv.org/abs/1708.02002`

## Project Status

- Current setup guide version merged: `SETUP_GUIDE.md` version 1.1.
- Checkpoint setup guide merged: `.ipynb_checkpoints/SETUP_GUIDE-checkpoint.md` version 1.0.
- README created on June 5, 2026.
- Web app support is included.
- Google Drive dataset notes from both setup guides are included.
