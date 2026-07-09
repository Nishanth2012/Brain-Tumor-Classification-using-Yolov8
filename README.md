# Brain Tumor Detection using YOLOv8

This was our final year project at SRM University-AP. The idea is simple. You give it an MRI scan of the brain and it finds the tumor and draws a box around it. We used YOLOv8 for the detection part and ESRGAN to sharpen the scans before running them.

## Dataset

The data is from Roboflow (braintumor-pzusy, version 2). Each MRI image has the tumor marked with a bounding box. There is only one class, brain tumor. We split it like this:

- 453 training images
- 127 validation images
- 57 test images

The images are not in this repo, that would make it too big. You can pull them from Roboflow using the link in `4_Dataset/data.yaml`, or let the notebook download them (you will need your own Roboflow API key for that).

## Model

We started from the `yolov8s` pretrained weights and trained on our data for 30 epochs at image size 650. It is not a heavy setup, it runs fine on a single Colab T4 GPU. Everything is inside the notebook.

For the image enhancement part we cloned ESRGAN (https://github.com/xinntao/ESRGAN) and ran it on the scans. The notebook does this with a `git clone`, so it is not committed here.

## Results

On the validation set after 30 epochs:

- Precision: 0.95
- Recall: 0.95
- mAP@50: 0.97
- mAP@50-95: 0.80

The curves, confusion matrix and some sample predictions are in `5_Results`.

## How to run

1. Open `2_Code/brain_tumor.ipynb`. It was made for Google Colab.
2. Replace `YOUR_ROBOFLOW_API_KEY` with your own key and run the download cell to get the dataset.
3. Run the training cell. The weights get saved under `runs/detect`.
4. The cells at the bottom run predictions on new images and write the results to a CSV.

If you just want to test with our trained model, use `3_Model/best.pt` directly with the ultralytics `YOLO` class, no training needed.

## What is in the repo

- `2_Code` - the main notebook and a small loss script
- `3_Model` - our trained weights (`best.pt`, `last.pt`)
- `4_Dataset` - the `data.yaml` config, images not included
- `5_Results` - `training_metrics` and `validation_metrics` (curves, confusion matrices, results.csv) and `sample_predictions` (a few example detections)

## Team

G. Nishanth, P. Dileep, B. Tarun Kumar.
Guided by Dr. Pradyut Kumar Sanki and Dr. Pranab Mandal.
