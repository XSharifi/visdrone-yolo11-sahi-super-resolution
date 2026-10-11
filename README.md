# visdrone-yolo11-sahi-super-resolution
Small object detection on drone images from VisDrone using YOLO11n. Compares standard inference, SAHI slicing, and ESPCN x4 super-resolution applied either before or after slicing, evaluated on 20 images with detection counts and visual comparisons. Built with Python, Ultralytics, SAHI, and OpenCV on Kaggle.



# Small Object Detection in Drone Imagery: YOLO11 + SAHI + Super-Resolution

Drone images contain many small, densely packed objects that standard detectors often miss after downscaling. This project uses 20 images from the **VisDrone2019-DET** dataset to study whether **slicing-aided inference (SAHI)** and **super-resolution (SR)** improve detection, and whether the order of SR and slicing matters.

## Experiments

1. **YOLO11 only:** detection on the full image.
2. **YOLO11 + SAHI:** the image is split into overlapping slices, detection runs on each slice, and the results are merged. Slice sizes tested: 256×256, 512×512, 1024×1024, and 512×256, all with 20% overlap.
3. **Slice → SR → YOLO11n:** the image is sliced, each slice is upscaled 4× with SR, and YOLO11n runs on the upscaled slices. Boxes are mapped back to original image coordinates. Results are compared with YOLO + SAHI without SR.
4. **SR → Slice → YOLO11n:** the full image is upscaled 4× first, then sliced with SAHI and passed to YOLO11n.
5. **SR → YOLO11n (no slicing):** the upscaled full image goes directly to the detector, as a baseline.

Additional experiments:
- Resizing images to 640×640 before detection, with 100×100 slices.
- Confidence thresholds of 0.35, 0.50 and 0.70 and their effect on detection counts.

## Methods and Tools

- **Dataset:** VisDrone2019-DET (first 20 training images)
- **Detector:** Ultralytics YOLO11n (`yolo11n.pt`)
- **Slicing:** SAHI (`get_prediction` vs. `get_sliced_prediction`)
- **Super-resolution:** ESPCN ×4 through OpenCV's `dnn_superres` module
- **Default confidence threshold:** 0.35
- **Evaluation:** visual comparison and number of detected objects per image for each pipeline
- **Stack:** Python, PyTorch, Ultralytics, SAHI, OpenCV, Pillow, Matplotlib (run on Kaggle with a GPU)

## Setup

```bash
pip install ultralytics sahi opencv-contrib-python pillow matplotlib
wget https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb
```

1. Download VisDrone2019-DET and set `image_folder` to your local path.
2. Make sure `yolo11n.pt` and `ESPCN_x4.pb` are available at the paths used in the notebook.
3. Run the notebook to reproduce the comparisons.

## Notes and Limitations

- Evaluation is based on detection counts and visual inspection only. No ground-truth metrics (mAP, precision, recall) are computed, and more detections do not necessarily mean better detections.
- In the manual Slice → SR → YOLO pipeline, slices smaller than the slice size (at image borders) are skipped.
- SR increases computation time considerably, especially when applied to every slice.
- Only 20 images were used, so the results are illustrative rather than statistically conclusive.

## Results

*Add your findings here, e.g. average detections per image for each pipeline, which slice size worked best, and whether SR helped before or after slicing, with example figures.*

## Acknowledgments

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [SAHI](https://github.com/obss/sahi)
- [TF-ESPCN](https://github.com/fannymonori/TF-ESPCN)
