import os
import json
import copy
import cv2
import torch
import numpy as np
import open3d as o3d

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
    flash
)

from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO

from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)

import torch.nn.functional as F


# ============================================================
# STEP 33
# IMPORT SPATIAL INTELLIGENCE MODULE
# ============================================================

from spatial_intelligence import (
    build_scene_graph
)


# ============================================================
# STEP 35
# IMPORT ROOM INTELLIGENCE MODULE
# ============================================================

from room_intelligence import (
    build_room_intelligence
)


# ============================================================
# STEP 36
# IMPORT FINAL REPORT MODULE
# ============================================================

from final_report import (
    save_final_digital_twin_report
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = (
    "semantic3d-digital-twin"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# IMAGE DIRECTORIES
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "uploads"
)

DETECTION_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "detections"
)

DEPTH_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "depth"
)

POINTCLOUD_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "pointclouds"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "reports"
)

DIGITAL_TWIN_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "digital_twin"
)


# ============================================================
# VIDEO DIRECTORIES
# ============================================================

VIDEO_UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "videos"
)

VIDEO_FRAME_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "video_frames"
)

VIDEO_DETECTION_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "video_detections"
)

VIDEO_DEPTH_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "video_depth"
)

VIDEO_POINTCLOUD_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "video_pointclouds"
)

VIDEO_REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "video_reports"
)


# ============================================================
# STEP 31 DIRECTORIES
# ============================================================

VIDEO_FUSION_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "video_fusion"
)

VIDEO_TRANSFORM_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "video_transforms"
)


# ============================================================
# STEP 32 DIRECTORY
# ============================================================

VIDEO_SEMANTIC_TWIN_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "video_semantic_twin"
)


# ============================================================
# STEP 33 DIRECTORY
# ============================================================

SCENE_GRAPH_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "scene_graphs"
)


# ============================================================
# STEP 35 DIRECTORY
# ============================================================

ROOM_INTELLIGENCE_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "room_intelligence"
)


# ============================================================
# STEP 36 DIRECTORY
# ============================================================

FINAL_REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "final_reports"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

ALL_FOLDERS = [

    UPLOAD_FOLDER,
    DETECTION_FOLDER,
    DEPTH_FOLDER,
    POINTCLOUD_FOLDER,
    REPORT_FOLDER,
    DIGITAL_TWIN_FOLDER,

    VIDEO_UPLOAD_FOLDER,
    VIDEO_FRAME_FOLDER,
    VIDEO_DETECTION_FOLDER,
    VIDEO_DEPTH_FOLDER,
    VIDEO_POINTCLOUD_FOLDER,
    VIDEO_REPORT_FOLDER,

    VIDEO_FUSION_FOLDER,
    VIDEO_TRANSFORM_FOLDER,

    VIDEO_SEMANTIC_TWIN_FOLDER,

    SCENE_GRAPH_FOLDER,

    ROOM_INTELLIGENCE_FOLDER,

    FINAL_REPORT_FOLDER
]


for folder in ALL_FOLDERS:

    os.makedirs(
        folder,
        exist_ok=True
    )


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER

app.config[
    "VIDEO_UPLOAD_FOLDER"
] = VIDEO_UPLOAD_FOLDER


# ============================================================
# FILE TYPES
# ============================================================

ALLOWED_IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"Using device: {device}"
)


# ============================================================
# YOLO
# ============================================================

print(
    "Loading YOLOv8n..."
)


yolo_model = YOLO(
    "yolov8n.pt"
)


print(
    "YOLOv8n loaded."
)


# ============================================================
# DEPTH ANYTHING V2
# ============================================================

DEPTH_MODEL_NAME = (
    "depth-anything/"
    "Depth-Anything-V2-Small-hf"
)


print(
    "Loading Depth Anything V2..."
)


depth_processor = (
    AutoImageProcessor
    .from_pretrained(
        DEPTH_MODEL_NAME
    )
)


depth_model = (
    AutoModelForDepthEstimation
    .from_pretrained(
        DEPTH_MODEL_NAME
    )
    .to(device)
)


depth_model.eval()


print(
    "Depth Anything V2 loaded."
)


# ============================================================
# VALIDATION
# ============================================================

def allowed_image(
    filename
):

    return (
        "." in filename
        and
        filename
        .rsplit(".", 1)[1]
        .lower()
        in ALLOWED_IMAGE_EXTENSIONS
    )


def allowed_video(
    filename
):

    return (
        "." in filename
        and
        filename
        .rsplit(".", 1)[1]
        .lower()
        in ALLOWED_VIDEO_EXTENSIONS
    )


# ============================================================
# DEPTH GENERATION
# ============================================================

def generate_depth_map(
    image_path,
    depth_output_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    width = image.width
    height = image.height


    inputs = depth_processor(
        images=image,
        return_tensors="pt"
    )


    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }


    with torch.no_grad():

        outputs = depth_model(
            **inputs
        )


        predicted_depth = (
            outputs.predicted_depth
        )


    predicted_depth = (
        F.interpolate(

            predicted_depth.unsqueeze(1),

            size=(
                height,
                width
            ),

            mode="bicubic",

            align_corners=False
        )
        .squeeze()
        .cpu()
        .numpy()
    )


    depth_min = float(
        predicted_depth.min()
    )

    depth_max = float(
        predicted_depth.max()
    )


    if depth_max > depth_min:

        normalized_depth = (
            (
                predicted_depth
                - depth_min
            )
            /
            (
                depth_max
                - depth_min
            )
        )

    else:

        normalized_depth = (
            np.zeros_like(
                predicted_depth
            )
        )


    depth_uint16 = (
        normalized_depth
        * 65535
    ).astype(
        np.uint16
    )


    cv2.imwrite(
        depth_output_path,
        depth_uint16
    )


    return depth_uint16


# ============================================================
# POINT CLOUD
# ============================================================

def generate_point_cloud(
    image_path,
    depth_uint16,
    output_path
):

    image = cv2.imread(
        image_path
    )


    if image is None:

        raise ValueError(
            "Unable to read image."
        )


    height, width = (
        image.shape[:2]
    )


    if depth_uint16.shape != (
        height,
        width
    ):

        depth_uint16 = cv2.resize(

            depth_uint16,

            (
                width,
                height
            ),

            interpolation=(
                cv2.INTER_NEAREST
            )
        )


    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    color_o3d = (
        o3d.geometry.Image(
            image_rgb
        )
    )


    depth_o3d = (
        o3d.geometry.Image(
            depth_uint16
        )
    )


    rgbd = (
        o3d.geometry
        .RGBDImage
        .create_from_color_and_depth(

            color_o3d,
            depth_o3d,

            depth_scale=65535.0,

            depth_trunc=1.0,

            convert_rgb_to_intensity=False
        )
    )


    fx = width * 0.8
    fy = height * 0.8

    cx = width / 2
    cy = height / 2


    intrinsic = (
        o3d.camera
        .PinholeCameraIntrinsic(

            width,
            height,

            fx,
            fy,

            cx,
            cy
        )
    )


    point_cloud = (
        o3d.geometry
        .PointCloud
        .create_from_rgbd_image(

            rgbd,
            intrinsic
        )
    )


    point_cloud.transform(
        [
            [1, 0, 0, 0],

            [0, -1, 0, 0],

            [0, 0, -1, 0],

            [0, 0, 0, 1]
        ]
    )


    if len(
        point_cloud.points
    ) > 0:

        point_cloud = (
            point_cloud
            .voxel_down_sample(
                0.01
            )
        )


    o3d.io.write_point_cloud(
        output_path,
        point_cloud
    )


    return point_cloud


# ============================================================
# SEMANTIC OBJECTS
# ============================================================

def generate_semantic_objects(
    yolo_results,
    depth_uint16,
    width,
    height
):

    semantic_objects = []


    fx = width * 0.8
    fy = height * 0.8

    cx = width / 2
    cy = height / 2


    if not yolo_results:

        return semantic_objects


    result = yolo_results[0]


    if result.boxes is None:

        return semantic_objects


    for box in result.boxes:

        coordinates = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )


        x1 = int(
            coordinates[0]
        )

        y1 = int(
            coordinates[1]
        )

        x2 = int(
            coordinates[2]
        )

        y2 = int(
            coordinates[3]
        )


        x1 = max(
            0,
            min(
                x1,
                width - 1
            )
        )


        y1 = max(
            0,
            min(
                y1,
                height - 1
            )
        )


        x2 = max(
            0,
            min(
                x2,
                width - 1
            )
        )


        y2 = max(
            0,
            min(
                y2,
                height - 1
            )
        )


        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )


        center_x = max(
            0,
            min(
                center_x,
                width - 1
            )
        )


        center_y = max(
            0,
            min(
                center_y,
                height - 1
            )
        )


        confidence = float(
            box.conf[0]
            .cpu()
            .item()
        )


        class_id = int(
            box.cls[0]
            .cpu()
            .item()
        )


        object_name = (
            yolo_model.names[
                class_id
            ]
        )


        raw_depth = float(
            depth_uint16[
                center_y,
                center_x
            ]
        )


        normalized_depth = (
            raw_depth
            /
            65535.0
        )


        X = (
            (
                center_x
                - cx
            )
            *
            normalized_depth
            /
            fx
        )


        Y = (
            (
                center_y
                - cy
            )
            *
            normalized_depth
            /
            fy
        )


        Z = normalized_depth


        semantic_objects.append(
            {

                "object":
                    object_name,

                "class_id":
                    class_id,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "bounding_box": {

                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2
                },

                "center_pixel": {

                    "x":
                        center_x,

                    "y":
                        center_y
                },

                "depth_normalized":
                    round(
                        normalized_depth,
                        4
                    ),

                "position_3d": {

                    "x":
                        round(
                            float(X),
                            6
                        ),

                    "y":
                        round(
                            float(Y),
                            6
                        ),

                    "z":
                        round(
                            float(Z),
                            6
                        )
                }
            }
        )


    return semantic_objects


# ============================================================
# IMAGE STATISTICS
# ============================================================

def generate_room_statistics(
    semantic_objects
):

    total = len(
        semantic_objects
    )


    counts = {}


    for obj in semantic_objects:

        name = obj[
            "object"
        ]


        counts[name] = (
            counts.get(
                name,
                0
            )
            + 1
        )


    if total == 0:

        return {

            "total_objects":
                0,

            "unique_object_types":
                0,

            "object_counts":
                {},

            "average_confidence":
                0.0,

            "closest_object":
                None
        }


    confidence_values = [

        obj[
            "confidence"
        ]

        for obj
        in semantic_objects
    ]


    closest_object = min(

        semantic_objects,

        key=lambda item:
            item[
                "depth_normalized"
            ]
    )


    return {

        "total_objects":
            total,

        "unique_object_types":
            len(
                counts
            ),

        "object_counts":
            counts,

        "average_confidence":
            round(
                float(
                    np.mean(
                        confidence_values
                    )
                ),
                4
            ),

        "closest_object":
            closest_object
    }


# ============================================================
# IMAGE DIGITAL TWIN
# ============================================================

def generate_digital_twin(
    filename,
    width,
    height,
    objects,
    statistics,
    pointcloud_filename
):

    return {

        "project": {

            "name":
                "Semantic 3D Indoor Digital Twin",

            "version":
                "1.0"
        },

        "source": {

            "image":
                filename,

            "width":
                width,

            "height":
                height
        },

        "reconstruction": {

            "object_detector":
                "YOLOv8n",

            "depth_model":
                "Depth Anything V2 Small",

            "point_cloud_engine":
                "Open3D",

            "point_cloud":
                pointcloud_filename,

            "depth_type":
                "relative",

            "metric_scale":
                False
        },

        "objects":
            objects,

        "statistics":
            statistics,

        "notice":
            (
                "Depth and 3D coordinates "
                "are relative estimates."
            )
    }


# ============================================================
# IMAGE PIPELINE
# ============================================================

def process_image(
    image_path,
    filename
):

    stem = os.path.splitext(
        filename
    )[0]


    image = cv2.imread(
        image_path
    )


    if image is None:

        raise ValueError(
            "Invalid image."
        )


    height, width = (
        image.shape[:2]
    )


    yolo_results = (
        yolo_model.predict(

            source=image_path,

            conf=0.25,

            verbose=False
        )
    )


    detection_filename = (
        f"{stem}_detected.jpg"
    )


    cv2.imwrite(

        os.path.join(
            DETECTION_FOLDER,
            detection_filename
        ),

        yolo_results[0].plot()
    )


    depth_filename = (
        f"{stem}_depth.png"
    )


    depth_uint16 = (
        generate_depth_map(

            image_path,

            os.path.join(
                DEPTH_FOLDER,
                depth_filename
            )
        )
    )


    pointcloud_filename = (
        f"{stem}.ply"
    )


    generate_point_cloud(

        image_path,

        depth_uint16,

        os.path.join(
            POINTCLOUD_FOLDER,
            pointcloud_filename
        )
    )


    semantic_objects = (
        generate_semantic_objects(

            yolo_results,

            depth_uint16,

            width,

            height
        )
    )


    room_statistics = (
        generate_room_statistics(
            semantic_objects
        )
    )


    semantic_filename = (
        f"{stem}_semantic.json"
    )


    with open(

        os.path.join(
            REPORT_FOLDER,
            semantic_filename
        ),

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(
            semantic_objects,
            file,
            indent=4
        )


    digital_twin = (
        generate_digital_twin(

            filename,

            width,

            height,

            semantic_objects,

            room_statistics,

            pointcloud_filename
        )
    )


    digital_twin_filename = (
        f"{stem}_digital_twin.json"
    )


    with open(

        os.path.join(
            DIGITAL_TWIN_FOLDER,
            digital_twin_filename
        ),

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(
            digital_twin,
            file,
            indent=4
        )


    return {

        "image_filename":
            filename,

        "detection_filename":
            detection_filename,

        "depth_filename":
            depth_filename,

        "pointcloud_filename":
            pointcloud_filename,

        "semantic_filename":
            semantic_filename,

        "digital_twin_filename":
            digital_twin_filename,

        "semantic_objects":
            semantic_objects,

        "room_statistics":
            room_statistics,

        "digital_twin":
            digital_twin
    }


# ============================================================
# VIDEO FRAME PIPELINE
# ============================================================

def process_video_frame(
    frame_path,
    video_stem,
    frame_number
):

    frame = cv2.imread(
        frame_path
    )


    if frame is None:

        return None


    height, width = (
        frame.shape[:2]
    )


    frame_id = (
        f"{video_stem}_"
        f"frame_"
        f"{frame_number:06d}"
    )


    yolo_results = (
        yolo_model.predict(

            source=frame_path,

            conf=0.25,

            verbose=False
        )
    )


    detection_filename = (
        f"{frame_id}_detected.jpg"
    )


    cv2.imwrite(

        os.path.join(
            VIDEO_DETECTION_FOLDER,
            detection_filename
        ),

        yolo_results[0].plot()
    )


    depth_filename = (
        f"{frame_id}_depth.png"
    )


    depth_uint16 = (
        generate_depth_map(

            frame_path,

            os.path.join(
                VIDEO_DEPTH_FOLDER,
                depth_filename
            )
        )
    )


    pointcloud_filename = (
        f"{frame_id}.ply"
    )


    generate_point_cloud(

        frame_path,

        depth_uint16,

        os.path.join(
            VIDEO_POINTCLOUD_FOLDER,
            pointcloud_filename
        )
    )


    objects = (
        generate_semantic_objects(

            yolo_results,

            depth_uint16,

            width,

            height
        )
    )


    return {

        "frame_number":
            frame_number,

        "frame_filename":
            os.path.basename(
                frame_path
            ),

        "detection_filename":
            detection_filename,

        "depth_filename":
            depth_filename,

        "pointcloud_filename":
            pointcloud_filename,

        "width":
            width,

        "height":
            height,

        "objects":
            objects,

        "object_count":
            len(
                objects
            )
    }


# ============================================================
# STEP 31
# POINT CLOUD PREPROCESSING
# ============================================================

def preprocess_point_cloud(
    point_cloud,
    voxel_size
):

    downsampled = (
        point_cloud
        .voxel_down_sample(
            voxel_size
        )
    )


    if len(
        downsampled.points
    ) < 10:

        return (
            downsampled,
            None
        )


    downsampled.estimate_normals(

        o3d.geometry
        .KDTreeSearchParamHybrid(

            radius=(
                voxel_size * 2
            ),

            max_nn=30
        )
    )


    fpfh = (
        o3d.pipelines.registration
        .compute_fpfh_feature(

            downsampled,

            o3d.geometry
            .KDTreeSearchParamHybrid(

                radius=(
                    voxel_size * 5
                ),

                max_nn=100
            )
        )
    )


    return (
        downsampled,
        fpfh
    )


# ============================================================
# STEP 31
# RANSAC
# ============================================================

def global_registration(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    voxel_size
):

    threshold = (
        voxel_size
        * 1.5
    )


    return (
        o3d.pipelines.registration
        .registration_ransac_based_on_feature_matching(

            source_down,
            target_down,

            source_fpfh,
            target_fpfh,

            True,

            threshold,

            o3d.pipelines.registration
            .TransformationEstimationPointToPoint(
                False
            ),

            3,

            [
                o3d.pipelines.registration
                .CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),

                o3d.pipelines.registration
                .CorrespondenceCheckerBasedOnDistance(
                    threshold
                )
            ],

            o3d.pipelines.registration
            .RANSACConvergenceCriteria(
                50000,
                0.999
            )
        )
    )


# ============================================================
# STEP 31
# ICP
# ============================================================

def refine_registration(
    source,
    target,
    initial_transform,
    voxel_size
):

    source_small = (
        source.voxel_down_sample(
            voxel_size
        )
    )


    target_small = (
        target.voxel_down_sample(
            voxel_size
        )
    )


    if (
        len(
            source_small.points
        ) < 10
        or
        len(
            target_small.points
        ) < 10
    ):

        return None


    source_small.estimate_normals(

        o3d.geometry
        .KDTreeSearchParamHybrid(

            radius=(
                voxel_size * 2
            ),

            max_nn=30
        )
    )


    target_small.estimate_normals(

        o3d.geometry
        .KDTreeSearchParamHybrid(

            radius=(
                voxel_size * 2
            ),

            max_nn=30
        )
    )


    return (
        o3d.pipelines.registration
        .registration_icp(

            source_small,

            target_small,

            voxel_size * 1.5,

            initial_transform,

            o3d.pipelines.registration
            .TransformationEstimationPointToPlane(),

            o3d.pipelines.registration
            .ICPConvergenceCriteria(
                max_iteration=60
            )
        )
    )


# ============================================================
# STEP 31
# PAIR REGISTRATION
# ============================================================

def register_point_cloud_pair(
    source,
    target,
    voxel_size=0.03
):

    try:

        (
            source_down,
            source_fpfh
        ) = preprocess_point_cloud(
            source,
            voxel_size
        )


        (
            target_down,
            target_fpfh
        ) = preprocess_point_cloud(
            target,
            voxel_size
        )


        if (
            source_fpfh is None
            or
            target_fpfh is None
        ):

            return {

                "success":
                    False,

                "transformation":
                    np.eye(4),

                "fitness":
                    0.0,

                "rmse":
                    0.0,

                "method":
                    "insufficient_points"
            }


        ransac_result = (
            global_registration(

                source_down,

                target_down,

                source_fpfh,

                target_fpfh,

                voxel_size
            )
        )


        icp_result = (
            refine_registration(

                source,

                target,

                ransac_result.transformation,

                voxel_size
            )
        )


        if icp_result is None:

            return {

                "success":
                    False,

                "transformation":
                    ransac_result.transformation,

                "fitness":
                    float(
                        ransac_result.fitness
                    ),

                "rmse":
                    float(
                        ransac_result.inlier_rmse
                    ),

                "method":
                    "RANSAC"
            }


        fitness = float(
            icp_result.fitness
        )


        return {

            "success":
                fitness >= 0.05,

            "transformation":
                icp_result.transformation,

            "fitness":
                fitness,

            "rmse":
                float(
                    icp_result.inlier_rmse
                ),

            "method":
                "FPFH_RANSAC_ICP"
        }


    except Exception as error:

        print(
            "Registration error:",
            error
        )


        return {

            "success":
                False,

            "transformation":
                np.eye(4),

            "fitness":
                0.0,

            "rmse":
                0.0,

            "method":
                "registration_failed"
        }


# ============================================================
# STEP 31
# MULTI-FRAME FUSION
# ============================================================

def register_and_merge_point_clouds(
    processed_frames,
    video_stem
):

    print()
    print(
        "=============================================="
    )
    print(
        "STEP 31: POINT CLOUD REGISTRATION"
    )
    print(
        "=============================================="
    )


    point_clouds = []
    valid_frames = []


    for frame in processed_frames:

        path = os.path.join(

            VIDEO_POINTCLOUD_FOLDER,

            frame[
                "pointcloud_filename"
            ]
        )


        cloud = (
            o3d.io.read_point_cloud(
                path
            )
        )


        if len(
            cloud.points
        ) >= 10:

            point_clouds.append(
                cloud
            )

            valid_frames.append(
                frame
            )


    if not point_clouds:

        return {

            "success":
                False,

            "merged_pointcloud_filename":
                None,

            "transform_filename":
                None,

            "registered_frames":
                0,

            "attempted_frames":
                0,

            "merged_point_count":
                0,

            "average_fitness":
                0.0,

            "average_rmse":
                0.0,

            "registration_results":
                []
        }


    maximum_frames = 10


    point_clouds = (
        point_clouds[
            :maximum_frames
        ]
    )


    valid_frames = (
        valid_frames[
            :maximum_frames
        ]
    )


    merged_cloud = (
        copy.deepcopy(
            point_clouds[0]
        )
    )


    accumulated = [
        np.eye(4)
    ]


    registration_results = [

        {

            "frame_number":
                valid_frames[0][
                    "frame_number"
                ],

            "reference_frame":
                None,

            "registered":
                True,

            "fitness":
                1.0,

            "rmse":
                0.0,

            "method":
                "reference_frame",

            "transformation":
                np.eye(4).tolist()
        }
    ]


    registered_count = 1


    for index in range(
        1,
        len(
            point_clouds
        )
    ):

        print(
            f"Registering "
            f"{index + 1}/"
            f"{len(point_clouds)}"
        )


        result = (
            register_point_cloud_pair(

                point_clouds[index],

                point_clouds[
                    index - 1
                ]
            )
        )


        previous_transform = (
            accumulated[
                index - 1
            ]
        )


        if result[
            "success"
        ]:

            global_transform = (
                previous_transform
                @
                result[
                    "transformation"
                ]
            )


            registered_count += 1


            aligned = (
                copy.deepcopy(
                    point_clouds[index]
                )
            )


            aligned.transform(
                global_transform
            )


            merged_cloud += aligned

        else:

            global_transform = (
                previous_transform.copy()
            )


        accumulated.append(
            global_transform
        )


        registration_results.append(
            {

                "frame_number":
                    valid_frames[index][
                        "frame_number"
                    ],

                "reference_frame":
                    valid_frames[
                        index - 1
                    ][
                        "frame_number"
                    ],

                "registered":
                    bool(
                        result[
                            "success"
                        ]
                    ),

                "fitness":
                    round(
                        float(
                            result[
                                "fitness"
                            ]
                        ),
                        6
                    ),

                "rmse":
                    round(
                        float(
                            result[
                                "rmse"
                            ]
                        ),
                        6
                    ),

                "method":
                    result[
                        "method"
                    ],

                "transformation":
                    global_transform.tolist()
            }
        )


    merged_cloud = (
        merged_cloud
        .voxel_down_sample(
            0.008
        )
    )


    if len(
        merged_cloud.points
    ) > 20:

        try:

            merged_cloud, _ = (
                merged_cloud
                .remove_statistical_outlier(

                    nb_neighbors=20,

                    std_ratio=2.0
                )
            )

        except Exception:

            pass


    merged_filename = (
        f"{video_stem}_"
        f"merged_room.ply"
    )


    o3d.io.write_point_cloud(

        os.path.join(
            VIDEO_FUSION_FOLDER,
            merged_filename
        ),

        merged_cloud
    )


    transform_filename = (
        f"{video_stem}_"
        f"camera_transforms.json"
    )


    with open(

        os.path.join(
            VIDEO_TRANSFORM_FOLDER,
            transform_filename
        ),

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            {

                "reference_coordinate_system":
                    "first_processed_frame",

                "metric_scale":
                    False,

                "frames":
                    registration_results
            },

            file,

            indent=4
        )


    successful = [

        item

        for item
        in registration_results

        if (
            item[
                "reference_frame"
            ]
            is not None
            and
            item[
                "registered"
            ]
        )
    ]


    average_fitness = (

        float(
            np.mean(
                [
                    item[
                        "fitness"
                    ]
                    for item
                    in successful
                ]
            )
        )

        if successful

        else 0.0
    )


    average_rmse = (

        float(
            np.mean(
                [
                    item[
                        "rmse"
                    ]
                    for item
                    in successful
                ]
            )
        )

        if successful

        else 0.0
    )


    print(
        "Step 31 complete."
    )


    return {

        "success":
            True,

        "merged_pointcloud_filename":
            merged_filename,

        "transform_filename":
            transform_filename,

        "registered_frames":
            registered_count,

        "attempted_frames":
            len(
                point_clouds
            ),

        "merged_point_count":
            len(
                merged_cloud.points
            ),

        "average_fitness":
            round(
                average_fitness,
                6
            ),

        "average_rmse":
            round(
                average_rmse,
                6
            ),

        "registration_results":
            registration_results
    }


# ============================================================
# STEP 32
# TRANSFORM SEMANTIC POSITION
# ============================================================

def transform_semantic_position(
    position,
    transformation
):

    local_point = np.array(

        [

            position["x"],

            -position["y"],

            -position["z"],

            1.0
        ],

        dtype=np.float64
    )


    global_point = (
        transformation
        @
        local_point
    )


    return {

        "x":
            float(
                global_point[0]
            ),

        "y":
            float(
                global_point[1]
            ),

        "z":
            float(
                global_point[2]
            )
    }


# ============================================================
# STEP 32
# COLLECT GLOBAL SEMANTIC OBJECTS
# ============================================================

def collect_registered_semantic_objects(
    processed_frames,
    registration_results
):

    transform_map = {}


    for item in registration_results:

        if item[
            "registered"
        ]:

            transform_map[
                item[
                    "frame_number"
                ]
            ] = np.array(

                item[
                    "transformation"
                ],

                dtype=np.float64
            )


    output = []


    for frame in processed_frames:

        frame_number = (
            frame[
                "frame_number"
            ]
        )


        if frame_number not in transform_map:

            continue


        transformation = (
            transform_map[
                frame_number
            ]
        )


        for obj in frame[
            "objects"
        ]:

            global_position = (
                transform_semantic_position(

                    obj[
                        "position_3d"
                    ],

                    transformation
                )
            )


            output.append(
                {

                    "object":
                        obj[
                            "object"
                        ],

                    "class_id":
                        obj.get(
                            "class_id"
                        ),

                    "confidence":
                        obj[
                            "confidence"
                        ],

                    "frame_number":
                        frame_number,

                    "timestamp_seconds":
                        frame.get(
                            "timestamp_seconds",
                            0
                        ),

                    "global_position_3d": {

                        "x":
                            round(
                                global_position[
                                    "x"
                                ],
                                6
                            ),

                        "y":
                            round(
                                global_position[
                                    "y"
                                ],
                                6
                            ),

                        "z":
                            round(
                                global_position[
                                    "z"
                                ],
                                6
                            )
                    }
                }
            )


    return output


# ============================================================
# STEP 32
# DISTANCE
# ============================================================

def semantic_distance(
    position_a,
    position_b
):

    a = np.array(
        [
            position_a[
                "x"
            ],

            position_a[
                "y"
            ],

            position_a[
                "z"
            ]
        ]
    )


    b = np.array(
        [
            position_b[
                "x"
            ],

            position_b[
                "y"
            ],

            position_b[
                "z"
            ]
        ]
    )


    return float(
        np.linalg.norm(
            a - b
        )
    )


# ============================================================
# STEP 32
# DEDUPLICATE
# ============================================================

def deduplicate_semantic_objects(
    objects,
    threshold=0.12
):

    clusters = []


    sorted_objects = sorted(

        objects,

        key=lambda item:
            item[
                "confidence"
            ],

        reverse=True
    )


    for detection in sorted_objects:

        best_cluster = None
        best_distance = None


        position = detection[
            "global_position_3d"
        ]


        for cluster in clusters:

            if (
                cluster[
                    "object"
                ]
                !=
                detection[
                    "object"
                ]
            ):

                continue


            distance = (
                semantic_distance(

                    position,

                    cluster[
                        "centroid"
                    ]
                )
            )


            if (
                distance <= threshold
                and
                (
                    best_distance is None
                    or
                    distance < best_distance
                )
            ):

                best_cluster = cluster
                best_distance = distance


        if best_cluster is None:

            clusters.append(
                {

                    "object":
                        detection[
                            "object"
                        ],

                    "class_id":
                        detection.get(
                            "class_id"
                        ),

                    "centroid":
                        dict(
                            position
                        ),

                    "positions":
                        [
                            position
                        ],

                    "confidences":
                        [
                            detection[
                                "confidence"
                            ]
                        ],

                    "frames":
                        [
                            detection[
                                "frame_number"
                            ]
                        ],

                    "timestamps":
                        [
                            detection[
                                "timestamp_seconds"
                            ]
                        ]
                }
            )


        else:

            best_cluster[
                "positions"
            ].append(
                position
            )


            best_cluster[
                "confidences"
            ].append(
                detection[
                    "confidence"
                ]
            )


            best_cluster[
                "frames"
            ].append(
                detection[
                    "frame_number"
                ]
            )


            best_cluster[
                "timestamps"
            ].append(
                detection[
                    "timestamp_seconds"
                ]
            )


            array = np.array(

                [

                    [
                        p["x"],
                        p["y"],
                        p["z"]
                    ]

                    for p in best_cluster[
                        "positions"
                    ]
                ]
            )


            centroid = (
                array.mean(
                    axis=0
                )
            )


            best_cluster[
                "centroid"
            ] = {

                "x":
                    float(
                        centroid[0]
                    ),

                "y":
                    float(
                        centroid[1]
                    ),

                "z":
                    float(
                        centroid[2]
                    )
            }


    fused = []


    for index, cluster in enumerate(
        clusters,
        start=1
    ):

        unique_frames = sorted(
            set(
                cluster[
                    "frames"
                ]
            )
        )


        confidences = (
            cluster[
                "confidences"
            ]
        )


        fused.append(
            {

                "semantic_id":
                    f"OBJ-{index:03d}",

                "object":
                    cluster[
                        "object"
                    ],

                "class_id":
                    cluster[
                        "class_id"
                    ],

                "position_3d": {

                    "x":
                        round(
                            cluster[
                                "centroid"
                            ][
                                "x"
                            ],
                            6
                        ),

                    "y":
                        round(
                            cluster[
                                "centroid"
                            ][
                                "y"
                            ],
                            6
                        ),

                    "z":
                        round(
                            cluster[
                                "centroid"
                            ][
                                "z"
                            ],
                            6
                        )
                },

                "occurrences":
                    len(
                        cluster[
                            "positions"
                        ]
                    ),

                "observed_in_frames":
                    unique_frames,

                "frame_observation_count":
                    len(
                        unique_frames
                    ),

                "average_confidence":
                    round(
                        float(
                            np.mean(
                                confidences
                            )
                        ),
                        4
                    ),

                "maximum_confidence":
                    round(
                        float(
                            np.max(
                                confidences
                            )
                        ),
                        4
                    ),

                "minimum_confidence":
                    round(
                        float(
                            np.min(
                                confidences
                            )
                        ),
                        4
                    ),

                "first_timestamp":
                    round(
                        float(
                            min(
                                cluster[
                                    "timestamps"
                                ]
                            )
                        ),
                        3
                    ),

                "last_timestamp":
                    round(
                        float(
                            max(
                                cluster[
                                    "timestamps"
                                ]
                            )
                        ),
                        3
                    )
            }
        )


    return fused


# ============================================================
# STEP 32
# STATISTICS
# ============================================================

def semantic_fusion_statistics(
    registered_objects,
    fused_objects
):

    counts = {}


    for obj in fused_objects:

        name = obj[
            "object"
        ]


        counts[name] = (
            counts.get(
                name,
                0
            )
            + 1
        )


    raw_count = len(
        registered_objects
    )


    final_count = len(
        fused_objects
    )


    removed = max(
        raw_count
        - final_count,
        0
    )


    reduction = (

        removed
        /
        raw_count
        *
        100

        if raw_count

        else 0
    )


    return {

        "registered_detections":
            raw_count,

        "fused_semantic_objects":
            final_count,

        "duplicates_removed":
            removed,

        "deduplication_reduction_percent":
            round(
                reduction,
                2
            ),

        "unique_object_types":
            len(
                counts
            ),

        "fused_object_counts":
            counts
    }


# ============================================================
# STEP 32
# BUILD SEMANTIC DIGITAL TWIN
# ============================================================

def generate_video_semantic_twin(
    video_filename,
    processed_frames,
    fusion_result,
    global_object_counts
):

    print()
    print(
        "=============================================="
    )
    print(
        "STEP 32: SEMANTIC FUSION"
    )
    print(
        "=============================================="
    )


    registered_objects = (
        collect_registered_semantic_objects(

            processed_frames,

            fusion_result.get(
                "registration_results",
                []
            )
        )
    )


    fused_objects = (
        deduplicate_semantic_objects(
            registered_objects,
            0.12
        )
    )


    statistics = (
        semantic_fusion_statistics(

            registered_objects,

            fused_objects
        )
    )


    stem = os.path.splitext(
        video_filename
    )[0]


    filename = (
        f"{stem}_"
        f"semantic_digital_twin.json"
    )


    semantic_twin = {

        "project": {

            "name":
                "Semantic 3D Indoor Digital Twin",

            "phase":
                "Semantic Fusion",

            "version":
                "4.0"
        },

        "source": {

            "video":
                video_filename
        },

        "reconstruction": {

            "merged_point_cloud":
                fusion_result.get(
                    "merged_pointcloud_filename"
                ),

            "camera_transforms":
                fusion_result.get(
                    "transform_filename"
                ),

            "metric_scale":
                False
        },

        "semantic_fusion": {

            "distance_threshold_relative":
                0.12,

            "statistics":
                statistics,

            "input_detection_counts":
                global_object_counts
        },

        "objects":
            fused_objects,

        "notice":
            (
                "Semantic positions use "
                "relative monocular depth."
            )
    }


    with open(

        os.path.join(
            VIDEO_SEMANTIC_TWIN_FOLDER,
            filename
        ),

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(
            semantic_twin,
            file,
            indent=4
        )


    print(
        "Registered detections:",
        len(
            registered_objects
        )
    )


    print(
        "Final semantic objects:",
        len(
            fused_objects
        )
    )


    print(
        "Duplicates removed:",
        statistics[
            "duplicates_removed"
        ]
    )


    print(
        "Step 32 complete."
    )


    return {

        "semantic_twin_filename":
            filename,

        "semantic_twin":
            semantic_twin,

        "registered_semantic_objects":
            registered_objects,

        "fused_semantic_objects":
            fused_objects,

        "semantic_statistics":
            statistics
    }


# ============================================================
# STEP 33
# GENERATE SCENE GRAPH
# ============================================================

def generate_scene_graph_file(
    video_filename,
    fused_semantic_objects
):

    print()
    print(
        "=============================================="
    )
    print(
        "STEP 33: SEMANTIC SCENE GRAPH"
    )
    print(
        "=============================================="
    )


    scene_graph = (
        build_scene_graph(
            fused_semantic_objects
        )
    )


    video_stem = os.path.splitext(
        video_filename
    )[0]


    scene_graph_filename = (
        f"{video_stem}_"
        f"scene_graph.json"
    )


    scene_graph_path = (
        os.path.join(
            SCENE_GRAPH_FOLDER,
            scene_graph_filename
        )
    )


    with open(
        scene_graph_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            scene_graph,
            file,
            indent=4
        )


    print(
        "Scene Graph Nodes:",
        scene_graph[
            "statistics"
        ][
            "total_nodes"
        ]
    )


    print(
        "Scene Graph Edges:",
        scene_graph[
            "statistics"
        ][
            "total_edges"
        ]
    )


    print(
        "Relationship Types:",
        scene_graph[
            "statistics"
        ][
            "relationship_types"
        ]
    )


    closest_pair = (
        scene_graph.get(
            "closest_object_pair"
        )
    )


    if closest_pair:

        print(
            "Closest pair:",
            closest_pair[
                "object_a"
            ],
            "<->",
            closest_pair[
                "object_b"
            ]
        )


    print(
        "Scene graph file:",
        scene_graph_filename
    )


    print(
        "Step 33 complete."
    )


    print(
        "=============================================="
    )

    print()


    return {

        "scene_graph_filename":
            scene_graph_filename,

        "scene_graph":
            scene_graph,

        "scene_graph_statistics":
            scene_graph[
                "statistics"
            ],

        "closest_object_pair":
            scene_graph.get(
                "closest_object_pair"
            ),

        "object_connectivity":
            scene_graph.get(
                "object_connectivity",
                []
            )
    }


# ============================================================
# STEP 35
# GENERATE ROOM INTELLIGENCE
# ============================================================

def generate_room_intelligence_file(
    video_filename,
    fused_semantic_objects,
    scene_graph
):

    print()
    print(
        "=============================================="
    )
    print(
        "STEP 35: ROOM-LEVEL INTELLIGENCE"
    )
    print(
        "=============================================="
    )


    room_intelligence = (
        build_room_intelligence(
            fused_semantic_objects,
            scene_graph
        )
    )


    video_stem = os.path.splitext(
        video_filename
    )[0]


    room_intelligence_filename = (
        f"{video_stem}_"
        f"room_intelligence.json"
    )


    room_intelligence_path = (
        os.path.join(
            ROOM_INTELLIGENCE_FOLDER,
            room_intelligence_filename
        )
    )


    with open(
        room_intelligence_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            room_intelligence,
            file,
            indent=4
        )


    predicted_room = (
        room_intelligence.get(
            "predicted_room",
            {}
        )
    )


    print(
        "Predicted Room:",
        predicted_room.get(
            "display_name",
            "Unknown"
        )
    )


    print(
        "Confidence:",
        round(
            float(
                predicted_room.get(
                    "confidence",
                    0.0
                )
            )
            * 100,
            2
        ),
        "%"
    )


    print(
        "Evidence Quality:",
        predicted_room.get(
            "evidence_quality",
            "unknown"
        )
    )


    print(
        "Room intelligence file:",
        room_intelligence_filename
    )


    print(
        "Step 35 complete."
    )


    print(
        "=============================================="
    )

    print()


    return {

        "room_intelligence_filename":
            room_intelligence_filename,

        "room_intelligence":
            room_intelligence,

        "predicted_room":
            predicted_room,

        "semantic_evidence":
            room_intelligence.get(
                "semantic_evidence",
                []
            ),

        "inferred_activities":
            room_intelligence.get(
                "inferred_activities",
                []
            ),

        "alternative_predictions":
            room_intelligence.get(
                "alternative_predictions",
                []
            ),

        "room_scores":
            room_intelligence.get(
                "all_room_scores",
                []
            )
    }


# ============================================================
# VIDEO PIPELINE
# ============================================================

def process_video(
    video_path,
    filename
):

    video_stem = os.path.splitext(
        filename
    )[0]


    capture = (
        cv2.VideoCapture(
            video_path
        )
    )


    if not capture.isOpened():

        raise ValueError(
            "Unable to open video."
        )


    fps = capture.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:

        fps = 30


    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    duration = (
        total_frames
        /
        fps
        if fps
        else 0
    )


    frame_interval = max(
        int(
            round(
                fps
            )
        ),
        1
    )


    max_processed_frames = 20


    frame_index = 0
    processed_count = 0


    processed_frames = []


    object_counts = {}


    confidences = []


    while capture.isOpened():

        success, frame = (
            capture.read()
        )


        if not success:

            break


        if (
            frame_index
            %
            frame_interval
            ==
            0
        ):

            frame_filename = (
                f"{video_stem}_"
                f"frame_"
                f"{frame_index:06d}.jpg"
            )


            frame_path = os.path.join(
                VIDEO_FRAME_FOLDER,
                frame_filename
            )


            cv2.imwrite(
                frame_path,
                frame
            )


            result = process_video_frame(

                frame_path,

                video_stem,

                frame_index
            )


            if result:

                result[
                    "timestamp_seconds"
                ] = round(
                    frame_index
                    /
                    fps,
                    3
                )


                processed_frames.append(
                    result
                )


                for obj in result[
                    "objects"
                ]:

                    name = obj[
                        "object"
                    ]


                    object_counts[name] = (
                        object_counts.get(
                            name,
                            0
                        )
                        + 1
                    )


                    confidences.append(
                        obj[
                            "confidence"
                        ]
                    )


                processed_count += 1


            if (
                processed_count
                >=
                max_processed_frames
            ):

                break


        frame_index += 1


    capture.release()


    total_detections = sum(

        frame[
            "object_count"
        ]

        for frame
        in processed_frames
    )


    average_confidence = (

        float(
            np.mean(
                confidences
            )
        )

        if confidences

        else 0.0
    )


    # ========================================================
    # STEP 31
    # ========================================================

    fusion_result = (
        register_and_merge_point_clouds(

            processed_frames,

            video_stem
        )
    )


    # ========================================================
    # STEP 32
    # ========================================================

    semantic_result = (
        generate_video_semantic_twin(

            filename,

            processed_frames,

            fusion_result,

            object_counts
        )
    )


    # ========================================================
    # STEP 33
    # AUTOMATIC SCENE GRAPH
    # ========================================================

    scene_graph_result = (
        generate_scene_graph_file(

            filename,

            semantic_result[
                "fused_semantic_objects"
            ]
        )
    )


    # ========================================================
    # STEP 35
    # ROOM-LEVEL INTELLIGENCE
    # ========================================================

    room_intelligence_result = (
        generate_room_intelligence_file(

            filename,

            semantic_result[
                "fused_semantic_objects"
            ],

            scene_graph_result[
                "scene_graph"
            ]
        )
    )


    # ========================================================
    # COMPLETE VIDEO REPORT
    # ========================================================

    report = {

        "project": {

            "name":
                "Semantic 3D Indoor Digital Twin",

            "version":
                "6.0",

            "phase":
                (
                    "Room-Level Intelligence "
                    "and Spatial Reasoning"
                )
        },

        "source": {

            "video":
                filename,

            "width":
                width,

            "height":
                height,

            "fps":
                round(
                    float(fps),
                    3
                ),

            "total_frames":
                total_frames,

            "duration_seconds":
                round(
                    float(duration),
                    3
                )
        },

        "statistics": {

            "processed_frames":
                len(
                    processed_frames
                ),

            "total_detections":
                total_detections,

            "unique_object_types":
                len(
                    object_counts
                ),

            "object_counts":
                object_counts,

            "average_confidence":
                round(
                    average_confidence,
                    4
                )
        },

        "registration": {

            "registered_frames":
                fusion_result.get(
                    "registered_frames",
                    0
                ),

            "attempted_frames":
                fusion_result.get(
                    "attempted_frames",
                    0
                ),

            "average_fitness":
                fusion_result.get(
                    "average_fitness",
                    0
                ),

            "average_rmse":
                fusion_result.get(
                    "average_rmse",
                    0
                ),

            "merged_point_cloud":
                fusion_result.get(
                    "merged_pointcloud_filename"
                )
        },

        "semantic_fusion": {

            "semantic_twin_file":
                semantic_result[
                    "semantic_twin_filename"
                ],

            "statistics":
                semantic_result[
                    "semantic_statistics"
                ]
        },

        # ====================================================
        # STEP 33 DATA
        # ====================================================

        "spatial_intelligence": {

            "scene_graph_file":
                scene_graph_result[
                    "scene_graph_filename"
                ],

            "statistics":
                scene_graph_result[
                    "scene_graph_statistics"
                ],

            "closest_object_pair":
                scene_graph_result[
                    "closest_object_pair"
                ],

            "object_connectivity":
                scene_graph_result[
                    "object_connectivity"
                ]
        },

        # ====================================================
        # STEP 35 DATA
        # ====================================================

        "room_intelligence": {

            "room_intelligence_file":
                room_intelligence_result[
                    "room_intelligence_filename"
                ],

            "predicted_room":
                room_intelligence_result[
                    "predicted_room"
                ],

            "semantic_evidence":
                room_intelligence_result[
                    "semantic_evidence"
                ],

            "inferred_activities":
                room_intelligence_result[
                    "inferred_activities"
                ],

            "alternative_predictions":
                room_intelligence_result[
                    "alternative_predictions"
                ]
        },

        "frames":
            processed_frames,

        "registration_results":
            fusion_result.get(
                "registration_results",
                []
            ),

        "fused_semantic_objects":
            semantic_result[
                "fused_semantic_objects"
            ],

        "notice":
            (
                "Depth, 3D coordinates, "
                "spatial distances and scene "
                "relationships are relative "
                "monocular reconstruction "
                "estimates and not metric "
                "room measurements."
            )
    }


    report_filename = (
        f"{video_stem}_"
        f"video_analysis.json"
    )


    with open(

        os.path.join(
            VIDEO_REPORT_FOLDER,
            report_filename
        ),

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )


    # ========================================================
    # STEP 36
    # FINAL DIGITAL TWIN REPORT & EXPORT
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "STEP 36: FINAL DIGITAL TWIN REPORT"
    )
    print(
        "=============================================="
    )


    final_report_filename = (
        f"{video_stem}_"
        f"final_digital_twin_report.json"
    )


    final_report_path = os.path.join(
        FINAL_REPORT_FOLDER,
        final_report_filename
    )


    final_digital_twin_report = (
        save_final_digital_twin_report(
            video_filename=filename,
            processed_frames=processed_frames,
            fusion_result=fusion_result,
            semantic_result=semantic_result,
            scene_graph_result=scene_graph_result,
            room_intelligence_result=(
                room_intelligence_result
            ),
            output_path=final_report_path,
            video_report_filename=report_filename
        )
    )


    print(
        "Final report file:",
        final_report_filename
    )

    print(
        "Pipeline completeness:",
        final_digital_twin_report
        .get(
            "quality_summary",
            {}
        )
        .get(
            "pipeline_completeness_percent",
            0
        ),
        "%"
    )

    print(
        "Step 36 complete."
    )

    print(
        "=============================================="
    )
    print()


    return {

        "video_filename":
            filename,

        "video_report_filename":
            report_filename,

        "video_report":
            report,

        "processed_frames":
            processed_frames,

        "processed_frame_count":
            len(
                processed_frames
            ),

        "total_detections":
            total_detections,

        "unique_object_types":
            len(
                object_counts
            ),

        "object_counts":
            object_counts,

        "average_confidence":
            round(
                average_confidence,
                4
            ),

        "fps":
            round(
                float(fps),
                3
            ),

        "duration":
            round(
                float(duration),
                3
            ),

        "total_frames":
            total_frames,

        "fusion_result":
            fusion_result,

        "merged_pointcloud_filename":
            fusion_result.get(
                "merged_pointcloud_filename"
            ),

        "transform_filename":
            fusion_result.get(
                "transform_filename"
            ),

        "semantic_fusion_result":
            semantic_result,

        "semantic_twin_filename":
            semantic_result[
                "semantic_twin_filename"
            ],

        "fused_semantic_objects":
            semantic_result[
                "fused_semantic_objects"
            ],

        "semantic_statistics":
            semantic_result[
                "semantic_statistics"
            ],

        # ====================================================
        # STEP 33 TEMPLATE VARIABLES
        # ====================================================

        "scene_graph_filename":
            scene_graph_result[
                "scene_graph_filename"
            ],

        "scene_graph":
            scene_graph_result[
                "scene_graph"
            ],

        "scene_graph_statistics":
            scene_graph_result[
                "scene_graph_statistics"
            ],

        "closest_object_pair":
            scene_graph_result[
                "closest_object_pair"
            ],

        "object_connectivity":
            scene_graph_result[
                "object_connectivity"
            ],

        # ====================================================
        # STEP 35 TEMPLATE VARIABLES
        # ====================================================

        "room_intelligence_filename":
            room_intelligence_result[
                "room_intelligence_filename"
            ],

        "room_intelligence":
            room_intelligence_result[
                "room_intelligence"
            ],

        "predicted_room":
            room_intelligence_result[
                "predicted_room"
            ],

        "semantic_evidence":
            room_intelligence_result[
                "semantic_evidence"
            ],

        "inferred_activities":
            room_intelligence_result[
                "inferred_activities"
            ],

        "alternative_predictions":
            room_intelligence_result[
                "alternative_predictions"
            ],

        "room_scores":
            room_intelligence_result[
                "room_scores"
            ],

        # ====================================================
        # STEP 36 TEMPLATE VARIABLES
        # ====================================================

        "final_report_filename":
            final_report_filename,

        "final_digital_twin_report":
            final_digital_twin_report
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=[
        "POST"
    ]
)
def upload_image():

    image_file = (

        request.files.get(
            "room_image"
        )

        or

        request.files.get(
            "image"
        )
    )


    if (
        image_file is None
        or
        image_file.filename == ""
    ):

        flash(
            "Please select an image."
        )

        return redirect(
            url_for(
                "index"
            )
        )


    if not allowed_image(
        image_file.filename
    ):

        flash(
            "Supported formats: "
            "JPG, JPEG and PNG."
        )

        return redirect(
            url_for(
                "index"
            )
        )


    filename = secure_filename(
        image_file.filename
    )


    path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    image_file.save(
        path
    )


    try:

        result = process_image(
            path,
            filename
        )

    except Exception as error:

        print(
            "Image error:",
            error
        )


        flash(
            f"Image processing failed: "
            f"{error}"
        )


        return redirect(
            url_for(
                "index"
            )
        )


    return render_template(

        "results.html",

        image_filename=
            result[
                "image_filename"
            ],

        detection_filename=
            result[
                "detection_filename"
            ],

        depth_filename=
            result[
                "depth_filename"
            ],

        pointcloud_filename=
            result[
                "pointcloud_filename"
            ],

        semantic_filename=
            result[
                "semantic_filename"
            ],

        digital_twin_filename=
            result[
                "digital_twin_filename"
            ],

        semantic_objects=
            result[
                "semantic_objects"
            ],

        detected_objects=
            result[
                "semantic_objects"
            ],

        room_statistics=
            result[
                "room_statistics"
            ],

        digital_twin=
            result[
                "digital_twin"
            ]
    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

@app.route(
    "/upload-video",
    methods=[
        "POST"
    ]
)
def upload_video():

    video_file = (

        request.files.get(
            "room_video"
        )

        or

        request.files.get(
            "video"
        )
    )


    if (
        video_file is None
        or
        video_file.filename == ""
    ):

        flash(
            "Please select a video."
        )

        return redirect(
            url_for(
                "index"
            )
        )


    if not allowed_video(
        video_file.filename
    ):

        flash(
            "Supported formats: "
            "MP4, AVI, MOV, MKV and WEBM."
        )

        return redirect(
            url_for(
                "index"
            )
        )


    filename = secure_filename(
        video_file.filename
    )


    path = os.path.join(
        VIDEO_UPLOAD_FOLDER,
        filename
    )


    video_file.save(
        path
    )


    try:

        result = process_video(
            path,
            filename
        )

    except Exception as error:

        print(
            "Video error:",
            error
        )


        flash(
            f"Video processing failed: "
            f"{error}"
        )


        return redirect(
            url_for(
                "index"
            )
        )


    return render_template(

        "video_results.html",

        video_filename=
            result[
                "video_filename"
            ],

        video_report_filename=
            result[
                "video_report_filename"
            ],

        video_report=
            result[
                "video_report"
            ],

        processed_frames=
            result[
                "processed_frames"
            ],

        processed_frame_count=
            result[
                "processed_frame_count"
            ],

        total_detections=
            result[
                "total_detections"
            ],

        unique_object_types=
            result[
                "unique_object_types"
            ],

        object_counts=
            result[
                "object_counts"
            ],

        average_confidence=
            result[
                "average_confidence"
            ],

        fps=
            result[
                "fps"
            ],

        duration=
            result[
                "duration"
            ],

        total_frames=
            result[
                "total_frames"
            ],

        fusion_result=
            result[
                "fusion_result"
            ],

        merged_pointcloud_filename=
            result[
                "merged_pointcloud_filename"
            ],

        transform_filename=
            result[
                "transform_filename"
            ],

        semantic_fusion_result=
            result[
                "semantic_fusion_result"
            ],

        semantic_twin_filename=
            result[
                "semantic_twin_filename"
            ],

        fused_semantic_objects=
            result[
                "fused_semantic_objects"
            ],

        semantic_statistics=
            result[
                "semantic_statistics"
            ],

        # ====================================================
        # STEP 33
        # ====================================================

        scene_graph_filename=
            result[
                "scene_graph_filename"
            ],

        scene_graph=
            result[
                "scene_graph"
            ],

        scene_graph_statistics=
            result[
                "scene_graph_statistics"
            ],

        closest_object_pair=
            result[
                "closest_object_pair"
            ],

        object_connectivity=
            result[
                "object_connectivity"
            ],

        # ====================================================
        # STEP 35
        # ====================================================

        room_intelligence_filename=
            result[
                "room_intelligence_filename"
            ],

        room_intelligence=
            result[
                "room_intelligence"
            ],

        predicted_room=
            result[
                "predicted_room"
            ],

        semantic_evidence=
            result[
                "semantic_evidence"
            ],

        inferred_activities=
            result[
                "inferred_activities"
            ],

        alternative_predictions=
            result[
                "alternative_predictions"
            ],

        room_scores=
            result[
                "room_scores"
            ],

        # ====================================================
        # STEP 36
        # ====================================================

        final_report_filename=
            result[
                "final_report_filename"
            ],

        final_digital_twin_report=
            result[
                "final_digital_twin_report"
            ]
    )


# ============================================================
# IMAGE FILE ROUTES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def serve_upload(
    filename
):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


@app.route(
    "/detections/<filename>"
)
def serve_detection(
    filename
):

    return send_from_directory(
        DETECTION_FOLDER,
        filename
    )


@app.route(
    "/depth/<filename>"
)
def serve_depth(
    filename
):

    return send_from_directory(
        DEPTH_FOLDER,
        filename
    )


@app.route(
    "/pointclouds/<filename>"
)
def serve_pointcloud(
    filename
):

    return send_from_directory(
        POINTCLOUD_FOLDER,
        filename
    )


@app.route(
    "/semantic/<filename>"
)
def serve_semantic(
    filename
):

    return send_from_directory(
        REPORT_FOLDER,
        filename
    )


@app.route(
    "/digital-twin/<filename>"
)
def serve_digital_twin(
    filename
):

    return send_from_directory(
        DIGITAL_TWIN_FOLDER,
        filename
    )


# ============================================================
# VIDEO FILE ROUTES
# ============================================================

@app.route(
    "/videos/<filename>"
)
def serve_video(
    filename
):

    return send_from_directory(
        VIDEO_UPLOAD_FOLDER,
        filename
    )


@app.route(
    "/video-frames/<filename>"
)
def serve_video_frame(
    filename
):

    return send_from_directory(
        VIDEO_FRAME_FOLDER,
        filename
    )


@app.route(
    "/video-detections/<filename>"
)
def serve_video_detection(
    filename
):

    return send_from_directory(
        VIDEO_DETECTION_FOLDER,
        filename
    )


@app.route(
    "/video-depth/<filename>"
)
def serve_video_depth(
    filename
):

    return send_from_directory(
        VIDEO_DEPTH_FOLDER,
        filename
    )


@app.route(
    "/video-pointclouds/<filename>"
)
def serve_video_pointcloud(
    filename
):

    return send_from_directory(
        VIDEO_POINTCLOUD_FOLDER,
        filename
    )


@app.route(
    "/video-reports/<filename>"
)
def serve_video_report(
    filename
):

    return send_from_directory(
        VIDEO_REPORT_FOLDER,
        filename
    )


# ============================================================
# STEP 31 ROUTES
# ============================================================

@app.route(
    "/video-fusion/<filename>"
)
def serve_video_fusion(
    filename
):

    return send_from_directory(
        VIDEO_FUSION_FOLDER,
        filename
    )


@app.route(
    "/video-transforms/<filename>"
)
def serve_video_transform(
    filename
):

    return send_from_directory(
        VIDEO_TRANSFORM_FOLDER,
        filename
    )


# ============================================================
# STEP 32 ROUTE
# ============================================================

@app.route(
    "/video-semantic-twin/<filename>"
)
def serve_video_semantic_twin(
    filename
):

    return send_from_directory(
        VIDEO_SEMANTIC_TWIN_FOLDER,
        filename
    )


# ============================================================
# STEP 33 ROUTE
# ============================================================

@app.route(
    "/scene-graphs/<filename>"
)
def serve_scene_graph(
    filename
):

    return send_from_directory(
        SCENE_GRAPH_FOLDER,
        filename
    )


# ============================================================
# STEP 35 ROUTE
# ============================================================

@app.route(
    "/room-intelligence/<filename>"
)
def serve_room_intelligence(
    filename
):

    return send_from_directory(
        ROOM_INTELLIGENCE_FOLDER,
        filename
    )


# ============================================================
# STEP 36 ROUTE
# ============================================================

@app.route(
    "/final-reports/<filename>"
)
def serve_final_report(
    filename
):

    return send_from_directory(
        FINAL_REPORT_FOLDER,
        filename
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=============================================="
    )

    print(
        " Semantic 3D Indoor Digital Twin Platform"
    )

    print(
        "=============================================="
    )

    print(
        "Image Pipeline         : READY"
    )

    print(
        "Video Pipeline         : READY"
    )

    print(
        "Step 31 Registration   : READY"
    )

    print(
        "Step 31 Fusion         : READY"
    )

    print(
        "Step 32 Semantic Fusion: READY"
    )

    print(
        "Step 33 Scene Graph    : READY"
    )

    print(
        "Step 33 Spatial Intel. : READY"
    )

    print(
        "Step 35 Room Intel.    : READY"
    )

    print(
        "Step 35 Classification : READY"
    )

    print(
        "Step 36 Final Report    : READY"
    )

    print(
        "Step 36 Export          : READY"
    )

    print(
        f"Device                 : {device}"
    )

    print(
        "Server                 : "
        "http://127.0.0.1:5000"
    )

    print(
        "=============================================="
    )

    print()


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False
    )