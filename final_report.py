import json
import os
from datetime import datetime


# ============================================================
# STEP 36
# FINAL DIGITAL TWIN REPORT & EXPORT
# ============================================================

"""
Final Digital Twin Report Builder

This module combines outputs from the completed pipeline:

Step 31 - Multi-frame point-cloud registration and fusion
Step 32 - Semantic fusion and deduplication
Step 33 - Semantic scene graph / spatial intelligence
Step 35 - Room-level intelligence

The generated report is a project-level JSON export that can
be archived, downloaded, or used by the final dashboard.

IMPORTANT:
Depth and reconstructed 3D coordinates are based on monocular
relative depth. They are not metric measurements in meters.
"""


FINAL_REPORT_VERSION = "1.0"


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_dict(value):
    return value if isinstance(value, dict) else {}


def safe_list(value):
    return value if isinstance(value, list) else []


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# RECONSTRUCTION SUMMARY
# ============================================================

def build_reconstruction_summary(
    processed_frames,
    fusion_result
):
    processed_frames = safe_list(processed_frames)
    fusion_result = safe_dict(fusion_result)

    registration_results = safe_list(
        fusion_result.get("registration_results")
    )

    successful_registrations = [
        item
        for item in registration_results
        if item.get("registered") is True
    ]

    return {
        "processed_frame_count":
            len(processed_frames),

        "registered_frames":
            safe_int(
                fusion_result.get("registered_frames")
            ),

        "attempted_frames":
            safe_int(
                fusion_result.get("attempted_frames")
            ),

        "merged_point_count":
            safe_int(
                fusion_result.get("merged_point_count")
            ),

        "average_registration_fitness":
            round(
                safe_float(
                    fusion_result.get("average_fitness")
                ),
                6
            ),

        "average_registration_rmse":
            round(
                safe_float(
                    fusion_result.get("average_rmse")
                ),
                6
            ),

        "successful_registration_records":
            len(successful_registrations),

        "merged_point_cloud_file":
            fusion_result.get(
                "merged_pointcloud_filename"
            ),

        "camera_transform_file":
            fusion_result.get(
                "transform_filename"
            ),

        "registration_method":
            "FPFH + RANSAC + ICP",

        "coordinate_scale":
            "relative",

        "metric_scale":
            False
    }


# ============================================================
# SEMANTIC SUMMARY
# ============================================================

def build_semantic_summary(
    semantic_result
):
    semantic_result = safe_dict(
        semantic_result
    )

    statistics = safe_dict(
        semantic_result.get(
            "semantic_statistics"
        )
    )

    fused_objects = safe_list(
        semantic_result.get(
            "fused_semantic_objects"
        )
    )

    return {
        "registered_detections":
            safe_int(
                statistics.get(
                    "registered_detections"
                )
            ),

        "fused_semantic_objects":
            safe_int(
                statistics.get(
                    "fused_semantic_objects",
                    len(fused_objects)
                )
            ),

        "duplicates_removed":
            safe_int(
                statistics.get(
                    "duplicates_removed"
                )
            ),

        "deduplication_reduction_percent":
            round(
                safe_float(
                    statistics.get(
                        "deduplication_reduction_percent"
                    )
                ),
                2
            ),

        "unique_object_types":
            safe_int(
                statistics.get(
                    "unique_object_types"
                )
            ),

        "object_counts":
            safe_dict(
                statistics.get(
                    "fused_object_counts"
                )
            ),

        "semantic_twin_file":
            semantic_result.get(
                "semantic_twin_filename"
            ),

        "objects":
            fused_objects
    }


# ============================================================
# SPATIAL INTELLIGENCE SUMMARY
# ============================================================

def build_spatial_summary(
    scene_graph_result
):
    scene_graph_result = safe_dict(
        scene_graph_result
    )

    scene_graph = safe_dict(
        scene_graph_result.get(
            "scene_graph"
        )
    )

    statistics = safe_dict(
        scene_graph_result.get(
            "scene_graph_statistics"
        )
    )

    if not statistics:
        statistics = safe_dict(
            scene_graph.get(
                "statistics"
            )
        )

    return {
        "scene_graph_file":
            scene_graph_result.get(
                "scene_graph_filename"
            ),

        "total_nodes":
            safe_int(
                statistics.get(
                    "total_nodes"
                )
            ),

        "total_edges":
            safe_int(
                statistics.get(
                    "total_edges"
                )
            ),

        "relationship_types":
            statistics.get(
                "relationship_types",
                0
            ),

        "relationship_counts":
            safe_dict(
                statistics.get(
                    "relationship_counts"
                )
            ),

        "closest_object_pair":
            scene_graph_result.get(
                "closest_object_pair",
                scene_graph.get(
                    "closest_object_pair"
                )
            ),

        "object_connectivity":
            safe_list(
                scene_graph_result.get(
                    "object_connectivity",
                    scene_graph.get(
                        "object_connectivity",
                        []
                    )
                )
            ),

        "scene_graph":
            scene_graph
    }


# ============================================================
# ROOM INTELLIGENCE SUMMARY
# ============================================================

def build_room_summary(
    room_intelligence_result
):
    room_intelligence_result = safe_dict(
        room_intelligence_result
    )

    room_intelligence = safe_dict(
        room_intelligence_result.get(
            "room_intelligence",
            room_intelligence_result
        )
    )

    predicted_room = safe_dict(
        room_intelligence.get(
            "predicted_room"
        )
    )

    return {
        "room_intelligence_file":
            room_intelligence_result.get(
                "room_intelligence_filename"
            ),

        "predicted_room_type":
            predicted_room.get(
                "room_type",
                "general_indoor_space"
            ),

        "predicted_room_name":
            predicted_room.get(
                "display_name",
                "General Indoor Space"
            ),

        "confidence":
            round(
                safe_float(
                    predicted_room.get(
                        "confidence"
                    )
                ),
                4
            ),

        "evidence_quality":
            predicted_room.get(
                "evidence_quality",
                "insufficient"
            ),

        "purpose":
            predicted_room.get(
                "purpose"
            ),

        "reasoning":
            predicted_room.get(
                "reasoning"
            ),

        "semantic_evidence":
            safe_list(
                room_intelligence.get(
                    "semantic_evidence"
                )
            ),

        "inferred_activities":
            safe_list(
                room_intelligence.get(
                    "inferred_activities"
                )
            ),

        "alternative_predictions":
            safe_list(
                room_intelligence.get(
                    "alternative_predictions"
                )
            ),

        "all_room_scores":
            safe_list(
                room_intelligence.get(
                    "all_room_scores"
                )
            ),

        "classification_method":
            room_intelligence.get(
                "classification_method"
            ),

        "model_based_classifier":
            bool(
                room_intelligence.get(
                    "model_based_classifier",
                    False
                )
            )
    }


# ============================================================
# ASSET MANIFEST
# ============================================================

def build_asset_manifest(
    video_filename,
    fusion_result,
    semantic_result,
    scene_graph_result,
    room_intelligence_result,
    video_report_filename=None
):
    fusion_result = safe_dict(
        fusion_result
    )

    semantic_result = safe_dict(
        semantic_result
    )

    scene_graph_result = safe_dict(
        scene_graph_result
    )

    room_intelligence_result = safe_dict(
        room_intelligence_result
    )

    return {
        "source_video":
            video_filename,

        "video_processing_report":
            video_report_filename,

        "merged_point_cloud":
            fusion_result.get(
                "merged_pointcloud_filename"
            ),

        "camera_transforms":
            fusion_result.get(
                "transform_filename"
            ),

        "semantic_digital_twin":
            semantic_result.get(
                "semantic_twin_filename"
            ),

        "semantic_scene_graph":
            scene_graph_result.get(
                "scene_graph_filename"
            ),

        "room_intelligence":
            room_intelligence_result.get(
                "room_intelligence_filename"
            )
    }


# ============================================================
# QUALITY / COMPLETENESS SUMMARY
# ============================================================

def build_quality_summary(
    reconstruction,
    semantic,
    spatial,
    room
):
    completed_components = []

    if reconstruction.get(
        "merged_point_cloud_file"
    ):
        completed_components.append(
            "3d_reconstruction"
        )

    if semantic.get(
        "semantic_twin_file"
    ):
        completed_components.append(
            "semantic_fusion"
        )

    if spatial.get(
        "scene_graph_file"
    ):
        completed_components.append(
            "spatial_intelligence"
        )

    if room.get(
        "room_intelligence_file"
    ):
        completed_components.append(
            "room_intelligence"
        )

    total_components = 4

    completeness_percent = (
        len(completed_components)
        /
        total_components
        *
        100
    )

    return {
        "completed_components":
            completed_components,

        "completed_component_count":
            len(completed_components),

        "expected_component_count":
            total_components,

        "pipeline_completeness_percent":
            round(
                completeness_percent,
                2
            ),

        "has_fused_geometry":
            bool(
                reconstruction.get(
                    "merged_point_cloud_file"
                )
            ),

        "has_semantic_objects":
            semantic.get(
                "fused_semantic_objects",
                0
            ) > 0,

        "has_scene_graph":
            spatial.get(
                "total_nodes",
                0
            ) > 0,

        "has_room_prediction":
            bool(
                room.get(
                    "predicted_room_name"
                )
            )
    }


# ============================================================
# FINAL REPORT BUILDER
# ============================================================

def build_final_digital_twin_report(
    video_filename,
    processed_frames,
    fusion_result,
    semantic_result,
    scene_graph_result,
    room_intelligence_result,
    video_report_filename=None
):
    reconstruction = (
        build_reconstruction_summary(
            processed_frames,
            fusion_result
        )
    )

    semantic = (
        build_semantic_summary(
            semantic_result
        )
    )

    spatial = (
        build_spatial_summary(
            scene_graph_result
        )
    )

    room = (
        build_room_summary(
            room_intelligence_result
        )
    )

    assets = (
        build_asset_manifest(
            video_filename,
            fusion_result,
            semantic_result,
            scene_graph_result,
            room_intelligence_result,
            video_report_filename
        )
    )

    quality = (
        build_quality_summary(
            reconstruction,
            semantic,
            spatial,
            room
        )
    )

    report = {
        "project": {
            "name":
                "Semantic 3D Indoor Digital Twin",

            "report_type":
                "Final Digital Twin Report",

            "report_version":
                FINAL_REPORT_VERSION,

            "pipeline_stage":
                "Step 36 - Final Report & Export"
        },

        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "source": {
            "video":
                video_filename
        },

        "pipeline": {
            "step_31":
                "Multi-frame registration and point-cloud fusion",

            "step_32":
                "Semantic fusion and object deduplication",

            "step_33":
                "Semantic scene graph and spatial relationships",

            "step_34":
                "Interactive spatial intelligence dashboard",

            "step_35":
                "Room-level intelligence and room classification",

            "step_36":
                "Final digital twin report and export"
        },

        "reconstruction":
            reconstruction,

        "semantic_intelligence":
            semantic,

        "spatial_intelligence":
            spatial,

        "room_intelligence":
            room,

        "asset_manifest":
            assets,

        "quality_summary":
            quality,

        "technical_stack": [
            "Python",
            "Flask",
            "OpenCV",
            "YOLOv8n",
            "Depth Anything V2",
            "Open3D",
            "FPFH",
            "RANSAC",
            "ICP",
            "Three.js"
        ],

        "limitations": [
            (
                "Depth Anything V2 provides monocular "
                "relative depth rather than metric depth."
            ),
            (
                "Reconstructed coordinates and scene-graph "
                "distance thresholds are relative and must "
                "not be interpreted as meters."
            ),
            (
                "Point-cloud registration quality depends on "
                "frame overlap, visual geometry and scene texture."
            ),
            (
                "Semantic object detection depends on YOLOv8n "
                "COCO classes and may miss unsupported objects."
            ),
            (
                "Room classification is rule-based semantic "
                "reasoning and is not a separately trained "
                "room-classification neural network."
            )
        ],

        "notice":
            (
                "This digital twin is an approximate semantic "
                "3D reconstruction intended for visualization, "
                "spatial reasoning and indoor scene analysis. "
                "It does not provide survey-grade metric geometry."
            )
    }

    return report


# ============================================================
# SAVE FINAL REPORT
# ============================================================

def save_final_digital_twin_report(
    video_filename,
    processed_frames,
    fusion_result,
    semantic_result,
    scene_graph_result,
    room_intelligence_result,
    output_path,
    video_report_filename=None
):
    report = (
        build_final_digital_twin_report(
            video_filename,
            processed_frames,
            fusion_result,
            semantic_result,
            scene_graph_result,
            room_intelligence_result,
            video_report_filename
        )
    )

    output_directory = (
        os.path.dirname(
            output_path
        )
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )

    return report


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    demo_processed_frames = [
        {
            "frame_number": 0
        },
        {
            "frame_number": 30
        },
        {
            "frame_number": 60
        }
    ]

    demo_fusion_result = {
        "registered_frames": 3,
        "attempted_frames": 3,
        "merged_point_count": 24567,
        "average_fitness": 0.72,
        "average_rmse": 0.018,
        "merged_pointcloud_filename":
            "demo_merged_room.ply",
        "transform_filename":
            "demo_camera_transforms.json",
        "registration_results": [
            {
                "frame_number": 0,
                "registered": True
            },
            {
                "frame_number": 30,
                "registered": True
            },
            {
                "frame_number": 60,
                "registered": True
            }
        ]
    }

    demo_semantic_result = {
        "semantic_twin_filename":
            "demo_semantic_digital_twin.json",

        "semantic_statistics": {
            "registered_detections": 12,
            "fused_semantic_objects": 4,
            "duplicates_removed": 8,
            "deduplication_reduction_percent": 66.67,
            "unique_object_types": 4,
            "fused_object_counts": {
                "couch": 1,
                "tv": 1,
                "chair": 1,
                "remote": 1
            }
        },

        "fused_semantic_objects": [
            {
                "semantic_id": "OBJ-001",
                "object": "couch"
            },
            {
                "semantic_id": "OBJ-002",
                "object": "tv"
            },
            {
                "semantic_id": "OBJ-003",
                "object": "chair"
            },
            {
                "semantic_id": "OBJ-004",
                "object": "remote"
            }
        ]
    }

    demo_scene_graph_result = {
        "scene_graph_filename":
            "demo_scene_graph.json",

        "scene_graph_statistics": {
            "total_nodes": 4,
            "total_edges": 8,
            "relationship_types": 3,
            "relationship_counts": {
                "near": 4,
                "left_of": 2,
                "right_of": 2
            }
        },

        "closest_object_pair": {
            "object_a": "couch",
            "object_b": "remote",
            "distance_relative": 0.08
        },

        "object_connectivity": [
            {
                "semantic_id": "OBJ-001",
                "object": "couch",
                "relationship_count": 3
            }
        ],

        "scene_graph": {
            "nodes": [],
            "edges": []
        }
    }

    demo_room_intelligence_result = {
        "room_intelligence_filename":
            "demo_room_intelligence.json",

        "room_intelligence": {
            "predicted_room": {
                "room_type": "living_room",
                "display_name": "Living Room",
                "confidence": 0.78,
                "evidence_quality": "strong",
                "purpose":
                    "Relaxation, entertainment and social interaction.",
                "reasoning":
                    "A couch and TV strongly support a living-room interpretation."
            },

            "semantic_evidence": [
                {
                    "object": "couch",
                    "importance": "strong"
                },
                {
                    "object": "tv",
                    "importance": "strong"
                }
            ],

            "inferred_activities": [
                "relaxation",
                "entertainment",
                "social interaction"
            ],

            "alternative_predictions": [
                {
                    "display_name": "Office",
                    "confidence": 0.12
                }
            ],

            "all_room_scores": [
                {
                    "display_name": "Living Room",
                    "confidence": 0.78
                },
                {
                    "display_name": "Office",
                    "confidence": 0.12
                }
            ],

            "classification_method":
                "Semantic object weighted heuristic classification",

            "model_based_classifier":
                False
        }
    }

    demo_report = (
        build_final_digital_twin_report(
            video_filename="demo_room.mp4",
            processed_frames=demo_processed_frames,
            fusion_result=demo_fusion_result,
            semantic_result=demo_semantic_result,
            scene_graph_result=demo_scene_graph_result,
            room_intelligence_result=demo_room_intelligence_result,
            video_report_filename="demo_video_report.json"
        )
    )

    print(
        json.dumps(
            demo_report,
            indent=4
        )
    )
