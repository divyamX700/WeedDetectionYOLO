# Weed Detection with YOLOv8

This repository contains a Jupyter-based workflow for detecting weeds in agricultural images using Ultralytics YOLOv8. The pipeline handles data ingestion from a YOLO-formatted dataset, restructures it for training, and trains a YOLOv8n model to classify objects into "weed" and "non-weed" categories.

## Contents

* **model_46.ipynb**: Main notebook containing the data preparation, training, and evaluation code.
* **model_weights.pt**: Serialized PyTorch weights from the trained YOLOv8 model.
* **report_weed_detection_46.pdf**: Project report detailing the experimental results and analysis.
* **46_semi_supervised_weed_detection_deck.pdf**: Slide deck outlining the project.

## Setup and Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/divyamX700/WeedDetectionYOLO
   cd https://github.com/divyamX700/WeedDetectionYOLO
   ```

2. Install the required dependencies. A GPU-enabled Python environment is highly recommended for reasonable training times:

   ```bash
   pip install ultralytics supervision numpy opencv-python torch torchvision matplotlib pandas scikit-learn
   ```

3. Download the dataset from Kaggle ([weeddetection](https://www.kaggle.com/datasets/divyamkulshrestha/weeddetection)) and extract it locally.

## Usage

1. Open **model_46.ipynb** in Jupyter Notebook or JupyterLab.
2. Update the dataset paths. In the second cell of the notebook, update the `images_path`, `annotations_path`, and `unlabeled_images_path` variables to point to the location of your downloaded dataset.
3. Run the notebook sequentially. The code will parse the annotations, build the YOLO directory structure (`dataset/images/train`, etc.), generate the `dataset.yaml` file, and execute the training loop.

## Hardware Notes

The training logs reflect execution on a CUDA-enabled GPU (Tesla T4). While you can run the inference and dataset prep on a CPU, training the YOLOv8 model will be significantly slower without a dedicated GPU.
