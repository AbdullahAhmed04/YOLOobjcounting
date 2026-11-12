# AI Object Counting Using Computer Vision

A web application that detects and counts objects in images with a pretrained YOLOv8 model. Users upload one or more images, choose which classes to count, set a confidence threshold, and get annotated results with total and per-class counts. Results are logged to CSV and can be compared against manual counts in a built-in evaluation table.

## Problem statement
Counting objects (people, vehicles, and similar) by hand is slow and error-prone. This project applies a pretrained object-detection model to count selected object classes automatically, and measures how accurate those counts are.

## Features
- Upload a single image or a batch of images (JPG, PNG, BMP, WEBP, up to 10 MB each)
- Choose one or more of the 80 COCO classes to count
- Adjustable confidence threshold and image size (640, 960, 1280)
- Annotated output with bounding boxes, labels, and confidence scores
- Total count and per-class counts
- Results log with input filename, timestamp, settings, and counts (CSV export)
- Enter actual (hand-counted) values to build an evaluation table with error, mean absolute error (MAE), and accuracy
- Download the annotated image and the evaluation CSV
- Input validation with clear error messages

## Technology
| Part | Tool |
|---|---|
| Detection model | YOLOv8s (Ultralytics), COCO pretrained weights |
| Image handling | OpenCV, NumPy |
| Backend | Python, FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript (no framework) |
| Environment | Google Colab (T4 GPU), ngrok for the public link |
| Storage | CSV files on Google Drive |

## Architecture
See `architecture.mermaid` (diagram of the flow from browser, to FastAPI, to YOLOv8, to the CSV log and evaluation).

## Setup (Google Colab)
1. Open `object_counting_week3_app.ipynb` in Google Colab.
2. Set the runtime to **T4 GPU** (Runtime > Change runtime type).
3. Create a free ngrok account and copy your authtoken.
4. In Colab, open **Secrets** (key icon), add `NGROK_AUTH_TOKEN` with your token, and turn on Notebook access.
5. Run the cells in order. Allow Google Drive access when asked.
6. Open the link printed by the last cell.

Dependencies are listed in `requirements.txt`.

## How to use
1. Upload one or more images.
2. Select the classes to count (for example, person).
3. Set the confidence threshold and image size.
4. Click **Count objects** to see the annotated image and counts.
5. In the History tab, enter the actual count for each image.
6. Open the Evaluation tab to see MAE and accuracy, and download the evaluation CSV.

## Testing
Test cases cover normal, invalid, and edge inputs. See `test_cases.md` and the screenshots folder.

## Evaluation results
(Fill in: number of test images, MAE, mean accuracy, and the predicted vs. actual table.)

## Known limitations
- Crowded scenes are undercounted because of occlusion (hidden or overlapping objects).
- Small or distant objects can be missed, especially at image size 640.
- Dark, blurry, or low-contrast images reduce accuracy.
- The model counts only the 80 COCO classes.
- Counts are estimates; confidence settings trade missed objects against false detections.
- The public link works only while the Colab session is running.

## Third-party components
- **YOLOv8** (Ultralytics), pretrained COCO weights, AGPL-3.0 license
- **COCO** dataset classes (via the pretrained model)
- **FastAPI**, **Uvicorn**, **OpenCV**, **NumPy**, **pyngrok / ngrok**
- Test images: (list each image source, for example Unsplash or Pexels, with links)

## Future improvements
- Video counting and line-crossing counts
- Compare model sizes (YOLOv8n, s, m) on speed and accuracy
- Fine-tune the model on a custom dataset
- Persistent hosting and a database instead of CSV files

## Author
Abdullah Ahmed
