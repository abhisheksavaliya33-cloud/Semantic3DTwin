Semantic 3D Reconstruction for Indoor Digital Twins

Minor Project

Project Title: Semantic 3D Reconstruction for Indoor Digital Twins
Domain: Computer Vision, 3D Reconstruction, Semantic Understanding, Digital Twins
Application Type: Flask Web Application
Recommended Python: Python 3.11.x

1. Project Overview

This project creates an approximate semantic 3D digital twin of an indoor environment from a room image or video.

The system combines:

YOLOv8n for object detection

Depth Anything V2 for monocular depth estimation

Open3D for point-cloud reconstruction

FPFH, RANSAC and ICP for multi-frame registration

Semantic object fusion and deduplication

Semantic scene graphs and spatial relationships

Rule-based room-level intelligence

Three.js for interactive browser-based 3D visualization

Flask for the web application and file-serving backend

The reconstructed depth and 3D coordinates are relative/non-metric. They must not be interpreted as accurate measurements in meters.

2. Complete Processing Pipeline

Room Image / Video
        |
        v
OpenCV Input Processing
        |
        v
YOLOv8n Object Detection
        |
        v
Depth Anything V2
Monocular Depth Estimation
        |
        v
Open3D Point-Cloud Reconstruction
        |
        v
Per-Frame Semantic Objects
        |
        v
FPFH + RANSAC + ICP
Multi-Frame Registration
        |
        v
Merged Room Point Cloud
        |
        v
Semantic Fusion + Deduplication
        |
        v
Semantic Scene Graph
        |
        v
Spatial Relationship Reasoning
        |
        v
Room-Level Intelligence
        |
        v
Final Digital Twin Report
        |
        v
Flask + Three.js Dashboard

3. Major Project Steps

Image Reconstruction Phase

YOLO object detection

Depth Anything V2 depth estimation

RGB-D point-cloud generation

Semantic object positioning

Digital twin JSON generation

Three.js point-cloud visualization

Object inventory and room statistics

Video Reconstruction Phase

Step 28: Video upload and backend processing
Step 29: Per-frame 3D reconstruction
Step 30: Video web interface
Step 31: Multi-frame registration and point-cloud fusion
Step 32: Semantic fusion and object deduplication
Step 33: Semantic scene graph and spatial relationships
Step 34: Interactive spatial-intelligence dashboard
Step 35: Room-level intelligence and room classification
Step 36: Final Digital Twin Report and export
Step 37: Final project testing, validation and cleanup
Step 38: Documentation and submission package

4. Technologies Used

Technology

Purpose

Python

Main programming language

Flask

Web backend

OpenCV

Image/video processing

YOLOv8n

Object detection

Depth Anything V2

Monocular depth estimation

PyTorch

Deep-learning runtime

Transformers

Depth Anything V2 model loading

Open3D

Point-cloud reconstruction and registration

NumPy

Numerical processing

Pillow

Image handling

Three.js

Interactive 3D web visualization

HTML/CSS/JavaScript

Dashboard interface

5. Recommended Environment

Use Python 3.11.x.

The project was developed using an Anaconda environment named:

semantic3d

Activate it with:

conda activate semantic3d

Then move to the project directory:

cd C:\Users\abhis\Desktop\Semantic3DTwin

6. Installation

Install the required packages:

pip install -r requirements.txt

If Open3D causes an installation problem, first verify that the active Python version is Python 3.11:

python --version

Expected format:

Python 3.11.x

Check Open3D:

python -c "import open3d as o3d; print(o3d.__version__)"

7. Main Project Structure

Semantic3DTwin/
|
|-- app.py
|-- spatial_intelligence.py
|-- room_intelligence.py
|-- final_report.py
|-- final_validation.py
|-- requirements.txt
|-- README.md
|-- yolov8n.pt
|
|-- templates/
|   |-- index.html
|   |-- results.html
|   `-- video_results.html
|
|-- static/
|
|-- uploads/
|
`-- outputs/
    |-- detections/
    |-- depth/
    |-- pointclouds/
    |-- reports/
    |-- digital_twin/
    |-- video_frames/
    |-- video_detections/
    |-- video_depth/
    |-- video_pointclouds/
    |-- video_fusion/
    |-- video_transforms/
    |-- video_semantic_twin/
    |-- scene_graphs/
    |-- room_intelligence/
    |-- final_reports/
    `-- validation/

Some output folders are created automatically when the corresponding pipeline stage runs.

8. Running the Application

Activate the environment:

conda activate semantic3d

Open the project directory:

cd C:\Users\abhis\Desktop\Semantic3DTwin

Run Flask:

python app.py

Open the local address printed in the terminal, normally:

http://127.0.0.1:5000

The application intentionally uses:

debug=False

to avoid Flask's development reloader loading large AI models multiple times.

9. Object Detection

The project uses:

yolov8n.pt

YOLO uses its original COCO class names. For example:

couch

is preserved as couch and is not renamed to sofa.

Detection confidence is handled by YOLO during prediction.

10. Depth Estimation

The project uses:

depth-anything/Depth-Anything-V2-Small-hf

through Hugging Face Transformers.

Depth Anything V2 produces monocular relative depth.

Therefore:

depth values are not physical meters

object distances are relative

point-cloud scale is approximate

room dimensions must not be presented as survey-grade measurements

11. 3D Reconstruction

RGB and relative depth are converted into an Open3D RGB-D representation.

A point cloud is generated using an approximate pinhole camera model.

The reconstruction is intended for:

visualization

spatial reasoning

semantic mapping

digital-twin demonstration

It is not intended for precision surveying or engineering measurement.

12. Multi-Frame Registration

For video reconstruction, multiple point clouds are aligned using:

FPFH
  +
RANSAC
  +
ICP

FPFH

Fast Point Feature Histograms provide geometric feature descriptors.

RANSAC

RANSAC performs coarse global alignment using feature correspondences.

ICP

Iterative Closest Point refines the transformation between point clouds.

The registered frames are merged into a larger room-level point cloud.

13. Semantic Fusion

Objects detected in multiple frames may refer to the same physical object.

Step 32 performs semantic fusion and deduplication using:

object class

registered 3D position

proximity

confidence

frame observations

The result is a fused semantic object map.

14. Semantic Scene Graph

Step 33 converts fused semantic objects into a graph.

Nodes

Nodes represent detected semantic objects.

Example:

OBJ-001 -> couch
OBJ-002 -> tv
OBJ-003 -> chair

Edges

Edges represent spatial relationships such as:

near
very_near
left_of
right_of
above
below
in_front_of
behind
approximately_same_height
horizontal_neighbor

Relationship thresholds are in relative 3D units, not meters.

15. Room-Level Intelligence

Step 35 performs semantic room classification.

Possible room categories include:

Bedroom

Living Room

Dining Room

Kitchen

Office

Study Room

Bathroom

General Indoor Space

The classifier uses semantic evidence from detected objects and contextual rules.

For example:

couch + tv
    ->
Living Room evidence

and:

bed
    ->
Bedroom evidence

The displayed confidence is a heuristic confidence score, not a calibrated neural-network probability.

The system does not claim that Step 35 is a separately trained room-classification model.

16. Interactive Dashboard

The video dashboard provides:

merged semantic 3D twin

Three.js point-cloud viewer

semantic object markers

object labels

camera controls

scene graph

relationship filtering

closest-object pair

object connectivity

Room Intelligence

final report downloads

The dashboard status for the completed pipeline is:

STEP 36 COMPLETE

17. Final Digital Twin Report

Step 36 generates:

outputs/final_reports/

with a file similar to:

room_final_digital_twin_report.json

The report combines:

project
source
pipeline
reconstruction
semantic_intelligence
spatial_intelligence
room_intelligence
asset_manifest
quality_summary
technical_stack
limitations

This provides a single project-level digital-twin export.

18. Final Validation

Run:

python final_validation.py

The validator checks:

required project files

Python syntax

required Python libraries

Flask pipeline integration

video dashboard integration

expected output folders

YOLO model availability

generated JSON validity

final report schema

relative-depth/non-metric safety notices

A machine-readable validation report is written to:

outputs/validation/final_validation_report.json

Optional cache cleanup:

python final_validation.py --cleanup

This removes only Python cache artifacts such as:

__pycache__/
*.pyc
*.pyo

It does not delete project outputs.

19. Important Limitations

Relative Depth

The system uses monocular depth estimation and does not recover guaranteed real-world metric scale.

Camera Intrinsics

Approximate camera intrinsics are used when actual camera calibration parameters are unavailable.

Object Detection

YOLOv8n is restricted to supported COCO object categories and may miss objects outside those classes.

Registration

Point-cloud registration quality depends on:

sufficient overlap between video frames

stable camera movement

scene geometry

lighting

visual texture

depth quality

Room Classification

Room classification is based on semantic heuristics and detected object context.

It is not a calibrated probabilistic room-recognition model.

20. Recommended Video Capture

For better reconstruction:

Move the camera slowly.

Avoid sudden rotation.

Maintain overlap between consecutive views.

Keep the room well illuminated.

Avoid excessive motion blur.

Capture visible furniture and meaningful semantic objects.

Avoid recording mostly blank walls.

Keep the camera reasonably stable.

21. Example Final Workflow

1. Start Flask application
2. Upload indoor room video
3. Extract representative frames
4. Detect objects with YOLOv8n
5. Estimate relative depth
6. Create per-frame point clouds
7. Register frames
8. Fuse point clouds
9. Fuse semantic objects
10. Build semantic scene graph
11. Infer spatial relationships
12. Predict room category
13. Display interactive dashboard
14. Export final digital twin report
15. Run final validation

22. Academic Demonstration Points

During demonstration or viva, emphasize these contributions:

integration of 2D object detection with monocular depth

conversion of RGB + relative depth into 3D point clouds

multi-frame point-cloud registration

semantic fusion across video frames

semantic scene-graph construction

spatial relationship reasoning

room-level semantic inference

browser-based interactive 3D visualization

unified digital-twin JSON export

Also clearly state:

The system generates an approximate semantic 3D reconstruction using monocular relative depth. It does not claim metrically accurate room measurements.

23. Final Submission Checklist

Before submission:

[ ] Activate semantic3d environment
[ ] Confirm Python 3.11.x
[ ] Run python final_validation.py
[ ] Ensure there are no FAIL results
[ ] Process one representative test video
[ ] Verify merged point cloud loads
[ ] Verify semantic markers appear
[ ] Verify Scene Graph tab works
[ ] Verify Room Intelligence works
[ ] Verify Final Digital Twin Report downloads
[ ] Keep final_validation_report.json
[ ] Remove only unnecessary cache files
[ ] Keep README.md and requirements.txt
[ ] Keep project limitations in report/presentation
[ ] Create final ZIP backup

24. Project Status

Image Reconstruction                 COMPLETE
Video Processing                     COMPLETE
3D Multi-Frame Registration          COMPLETE
Point-Cloud Fusion                   COMPLETE
Semantic Fusion                      COMPLETE
Semantic Scene Graph                 COMPLETE
Spatial Intelligence                 COMPLETE
Interactive 3D Dashboard             COMPLETE
Room Intelligence                    COMPLETE
Final Digital Twin Report            COMPLETE
Testing and Validation               COMPLETE
Documentation                        COMPLETE

Final Project Status: Ready for final verification and submission packaging.