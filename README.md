# Semantic 3D Reconstruction for Indoor Digital Twins

## Overview

**Semantic 3D Reconstruction for Indoor Digital Twins** is a Computer Vision and 3D Reconstruction project that converts an indoor **image or video into an approximate semantic 3D digital twin**.

The system combines object detection, monocular depth estimation, 3D point-cloud reconstruction, multi-frame registration, semantic object fusion, spatial reasoning, room classification, and interactive browser-based 3D visualization.

> **Note:** The generated depth and 3D coordinates are relative and non-metric. The project is designed for visualization, semantic understanding, and spatial reasoning rather than accurate real-world measurement.

---

## Project Workflow

```text
Image / Video Input
        ↓
OpenCV Preprocessing
        ↓
YOLOv8n Object Detection
        ↓
Depth Anything V2 Depth Estimation
        ↓
Open3D 3D Point-Cloud Reconstruction
        ↓
Per-Frame Semantic Object Mapping
        ↓
FPFH + RANSAC + ICP Registration
        ↓
Merged Room Point Cloud
        ↓
Semantic Object Fusion & Deduplication
        ↓
Semantic Scene Graph
        ↓
Spatial Relationship Analysis
        ↓
Room-Level Intelligence
        ↓
Final Digital Twin Report
        ↓
Flask + Three.js Interactive Dashboard
```

---

## Main Features

* **YOLOv8n Object Detection** for identifying indoor objects.
* **Depth Anything V2** for estimating relative depth from RGB images.
* **Open3D** for generating and processing 3D point clouds.
* **FPFH + RANSAC + ICP** for aligning multiple video frames.
* **Semantic Fusion** for removing duplicate objects detected across frames.
* **Scene Graph Generation** for representing objects and their relationships.
* Spatial relations such as `near`, `left_of`, `right_of`, `above`, `below`, `in_front_of`, and `behind`.
* **Room Classification** using detected objects and contextual rules.
* **Three.js 3D Viewer** for interactive browser-based point-cloud visualization.
* Automatic **Digital Twin JSON Report** generation.
* Final project validation and output verification.

---

## Technologies Used

| Technology              | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| Python                  | Main programming language                      |
| Flask                   | Web application backend                        |
| OpenCV                  | Image and video processing                     |
| YOLOv8n                 | Object detection                               |
| Depth Anything V2       | Monocular depth estimation                     |
| PyTorch                 | Deep-learning runtime                          |
| Open3D                  | 3D reconstruction and point-cloud registration |
| NumPy                   | Numerical processing                           |
| Three.js                | Interactive 3D visualization                   |
| HTML / CSS / JavaScript | Web dashboard                                  |

---

## How the System Works

For an **image**, the system detects objects using YOLOv8n and estimates relative depth using Depth Anything V2. The RGB image and depth information are converted into an Open3D point cloud. Detected objects are then mapped into the reconstructed 3D scene.

For a **video**, representative frames are processed independently. Their point clouds are aligned using **FPFH feature matching, RANSAC global registration, and ICP refinement**. The aligned frames are merged to create a larger room-level 3D reconstruction.

Objects detected repeatedly across multiple frames are combined using semantic class, registered 3D position, proximity, confidence, and frame observations.

The fused objects are converted into a **semantic scene graph**, where objects represent nodes and spatial relationships represent edges.

Finally, the system analyzes the detected object combinations to infer room categories such as:

* Bedroom
* Living Room
* Dining Room
* Kitchen
* Office
* Study Room
* Bathroom
* General Indoor Space

The final result is displayed using a **Flask + Three.js interactive dashboard** containing the reconstructed point cloud, semantic markers, scene graph, spatial relationships, room intelligence, and downloadable digital-twin report.

---
