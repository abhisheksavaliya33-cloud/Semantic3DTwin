import math
import json


# ============================================================
# SEMANTIC 3D SPATIAL INTELLIGENCE
# ============================================================

"""
Step 33
Semantic Scene Graph + Spatial Relationship Analysis

IMPORTANT:

The 3D positions used by this project come from
monocular relative depth.

Therefore all spatial thresholds in this file
use RELATIVE 3D UNITS.

They are NOT meters.
"""


# ============================================================
# CONFIGURATION
# ============================================================

NEAR_THRESHOLD = 0.25

VERY_NEAR_THRESHOLD = 0.12

SAME_LEVEL_THRESHOLD = 0.10

ALIGNMENT_THRESHOLD = 0.12


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# GET POSITION
# ============================================================

def get_position(
    semantic_object
):

    position = semantic_object.get(
        "position_3d",
        {}
    )


    return {

        "x":
            safe_float(
                position.get(
                    "x",
                    0.0
                )
            ),

        "y":
            safe_float(
                position.get(
                    "y",
                    0.0
                )
            ),

        "z":
            safe_float(
                position.get(
                    "z",
                    0.0
                )
            )
    }


# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

def calculate_distance(
    object_a,
    object_b
):

    position_a = get_position(
        object_a
    )

    position_b = get_position(
        object_b
    )


    dx = (
        position_a["x"]
        -
        position_b["x"]
    )

    dy = (
        position_a["y"]
        -
        position_b["y"]
    )

    dz = (
        position_a["z"]
        -
        position_b["z"]
    )


    return math.sqrt(
        (
            dx * dx
        )
        +
        (
            dy * dy
        )
        +
        (
            dz * dz
        )
    )


# ============================================================
# HORIZONTAL DISTANCE
# ============================================================

def calculate_horizontal_distance(
    object_a,
    object_b
):

    position_a = get_position(
        object_a
    )

    position_b = get_position(
        object_b
    )


    dx = (
        position_a["x"]
        -
        position_b["x"]
    )


    dz = (
        position_a["z"]
        -
        position_b["z"]
    )


    return math.sqrt(
        (
            dx * dx
        )
        +
        (
            dz * dz
        )
    )


# ============================================================
# CREATE RELATIONSHIP
# ============================================================

def create_relationship(
    source,
    target,
    relationship,
    distance,
    strength
):

    return {

        "source_id":
            source.get(
                "semantic_id"
            ),

        "source_object":
            source.get(
                "object"
            ),

        "target_id":
            target.get(
                "semantic_id"
            ),

        "target_object":
            target.get(
                "object"
            ),

        "relationship":
            relationship,

        "distance_relative":
            round(
                distance,
                5
            ),

        "strength":
            strength
    }


# ============================================================
# DETERMINE SPATIAL RELATIONSHIPS
# ============================================================

def determine_relationships(
    source,
    target
):

    relationships = []


    source_position = get_position(
        source
    )

    target_position = get_position(
        target
    )


    distance = calculate_distance(
        source,
        target
    )


    horizontal_distance = (
        calculate_horizontal_distance(
            source,
            target
        )
    )


    dx = (
        target_position["x"]
        -
        source_position["x"]
    )

    dy = (
        target_position["y"]
        -
        source_position["y"]
    )

    dz = (
        target_position["z"]
        -
        source_position["z"]
    )


    # ========================================================
    # NEAR / VERY NEAR
    # ========================================================

    if (
        distance
        <= VERY_NEAR_THRESHOLD
    ):

        relationships.append(
            create_relationship(
                source,
                target,
                "very_near",
                distance,
                "strong"
            )
        )

    elif (
        distance
        <= NEAR_THRESHOLD
    ):

        relationships.append(
            create_relationship(
                source,
                target,
                "near",
                distance,
                "medium"
            )
        )


    # ========================================================
    # LEFT / RIGHT
    # ========================================================

    if abs(dx) > ALIGNMENT_THRESHOLD:

        if dx > 0:

            relationships.append(
                create_relationship(
                    source,
                    target,
                    "left_of",
                    distance,
                    "directional"
                )
            )

        else:

            relationships.append(
                create_relationship(
                    source,
                    target,
                    "right_of",
                    distance,
                    "directional"
                )
            )


    # ========================================================
    # ABOVE / BELOW
    # ========================================================

    if abs(dy) > SAME_LEVEL_THRESHOLD:

        if dy > 0:

            relationships.append(
                create_relationship(
                    source,
                    target,
                    "below",
                    distance,
                    "directional"
                )
            )

        else:

            relationships.append(
                create_relationship(
                    source,
                    target,
                    "above",
                    distance,
                    "directional"
                )
            )


    # ========================================================
    # FRONT / BEHIND
    # ========================================================

    if abs(dz) > ALIGNMENT_THRESHOLD:

        if dz < 0:

            relationships.append(
                create_relationship(
                    source,
                    target,
                    "behind",
                    distance,
                    "directional"
                )
            )

        else:

            relationships.append(
                create_relationship(
                    source,
                    target,
                    "in_front_of",
                    distance,
                    "directional"
                )
            )


    # ========================================================
    # SAME APPROXIMATE HEIGHT
    # ========================================================

    if (
        abs(dy)
        <= SAME_LEVEL_THRESHOLD
    ):

        relationships.append(
            create_relationship(
                source,
                target,
                "approximately_same_height",
                distance,
                "weak"
            )
        )


    # ========================================================
    # HORIZONTAL PROXIMITY
    # ========================================================

    if (
        horizontal_distance
        <= NEAR_THRESHOLD
    ):

        relationships.append(
            create_relationship(
                source,
                target,
                "horizontal_neighbor",
                distance,
                "medium"
            )
        )


    return relationships


# ============================================================
# BUILD SCENE NODES
# ============================================================

def build_scene_nodes(
    fused_semantic_objects
):

    nodes = []


    for semantic_object in (
        fused_semantic_objects
    ):

        nodes.append(
            {

                "id":
                    semantic_object.get(
                        "semantic_id"
                    ),

                "label":
                    semantic_object.get(
                        "object"
                    ),

                "class_id":
                    semantic_object.get(
                        "class_id"
                    ),

                "position_3d":
                    get_position(
                        semantic_object
                    ),

                "occurrences":
                    semantic_object.get(
                        "occurrences",
                        1
                    ),

                "frame_observation_count":
                    semantic_object.get(
                        "frame_observation_count",
                        1
                    ),

                "average_confidence":
                    semantic_object.get(
                        "average_confidence",
                        0.0
                    ),

                "maximum_confidence":
                    semantic_object.get(
                        "maximum_confidence",
                        0.0
                    )
            }
        )


    return nodes


# ============================================================
# BUILD SCENE EDGES
# ============================================================

def build_scene_edges(
    fused_semantic_objects
):

    edges = []


    total_objects = len(
        fused_semantic_objects
    )


    for source_index in range(
        total_objects
    ):

        source = (
            fused_semantic_objects[
                source_index
            ]
        )


        for target_index in range(
            source_index + 1,
            total_objects
        ):

            target = (
                fused_semantic_objects[
                    target_index
                ]
            )


            relationships = (
                determine_relationships(
                    source,
                    target
                )
            )


            edges.extend(
                relationships
            )


            # =================================================
            # ADD INVERSE DIRECTIONAL RELATIONSHIPS
            # =================================================

            reverse_relationships = []


            for relationship in (
                relationships
            ):

                relation_name = (
                    relationship[
                        "relationship"
                    ]
                )


                inverse_relation = None


                if relation_name == "left_of":

                    inverse_relation = (
                        "right_of"
                    )


                elif relation_name == "right_of":

                    inverse_relation = (
                        "left_of"
                    )


                elif relation_name == "above":

                    inverse_relation = (
                        "below"
                    )


                elif relation_name == "below":

                    inverse_relation = (
                        "above"
                    )


                elif (
                    relation_name
                    ==
                    "in_front_of"
                ):

                    inverse_relation = (
                        "behind"
                    )


                elif relation_name == "behind":

                    inverse_relation = (
                        "in_front_of"
                    )


                elif relation_name == "near":

                    inverse_relation = (
                        "near"
                    )


                elif (
                    relation_name
                    ==
                    "very_near"
                ):

                    inverse_relation = (
                        "very_near"
                    )


                elif (
                    relation_name
                    ==
                    "approximately_same_height"
                ):

                    inverse_relation = (
                        "approximately_same_height"
                    )


                elif (
                    relation_name
                    ==
                    "horizontal_neighbor"
                ):

                    inverse_relation = (
                        "horizontal_neighbor"
                    )


                if inverse_relation:

                    reverse_relationships.append(
                        create_relationship(
                            target,
                            source,
                            inverse_relation,
                            relationship[
                                "distance_relative"
                            ],
                            relationship[
                                "strength"
                            ]
                        )
                    )


            edges.extend(
                reverse_relationships
            )


    return edges


# ============================================================
# RELATIONSHIP STATISTICS
# ============================================================

def generate_relationship_statistics(
    edges
):

    relationship_counts = {}


    for edge in edges:

        relationship = edge[
            "relationship"
        ]


        relationship_counts[
            relationship
        ] = (
            relationship_counts.get(
                relationship,
                0
            )
            + 1
        )


    return {

        "total_relationships":
            len(
                edges
            ),

        "relationship_types":
            len(
                relationship_counts
            ),

        "relationship_counts":
            relationship_counts
    }


# ============================================================
# OBJECT CONNECTIVITY
# ============================================================

def calculate_object_connectivity(
    nodes,
    edges
):

    connectivity = {}


    for node in nodes:

        connectivity[
            node["id"]
        ] = 0


    for edge in edges:

        source_id = edge[
            "source_id"
        ]


        if source_id in connectivity:

            connectivity[
                source_id
            ] += 1


    connectivity_list = []


    for node in nodes:

        object_id = node[
            "id"
        ]


        connectivity_list.append(
            {
                "semantic_id":
                    object_id,

                "object":
                    node[
                        "label"
                    ],

                "relationship_count":
                    connectivity.get(
                        object_id,
                        0
                    )
            }
        )


    connectivity_list.sort(
        key=lambda item:
            item[
                "relationship_count"
            ],
        reverse=True
    )


    return connectivity_list


# ============================================================
# FIND CLOSEST OBJECT PAIR
# ============================================================

def find_closest_object_pair(
    fused_semantic_objects
):

    if len(
        fused_semantic_objects
    ) < 2:

        return None


    closest_pair = None

    closest_distance = None


    for i in range(
        len(
            fused_semantic_objects
        )
    ):

        for j in range(
            i + 1,
            len(
                fused_semantic_objects
            )
        ):

            object_a = (
                fused_semantic_objects[i]
            )

            object_b = (
                fused_semantic_objects[j]
            )


            distance = calculate_distance(
                object_a,
                object_b
            )


            if (
                closest_distance is None
                or
                distance
                <
                closest_distance
            ):

                closest_distance = (
                    distance
                )

                closest_pair = {

                    "object_a_id":
                        object_a.get(
                            "semantic_id"
                        ),

                    "object_a":
                        object_a.get(
                            "object"
                        ),

                    "object_b_id":
                        object_b.get(
                            "semantic_id"
                        ),

                    "object_b":
                        object_b.get(
                            "object"
                        ),

                    "distance_relative":
                        round(
                            distance,
                            5
                        )
                }


    return closest_pair


# ============================================================
# BUILD COMPLETE SCENE GRAPH
# ============================================================

def build_scene_graph(
    fused_semantic_objects
):

    nodes = build_scene_nodes(
        fused_semantic_objects
    )


    edges = build_scene_edges(
        fused_semantic_objects
    )


    relationship_statistics = (
        generate_relationship_statistics(
            edges
        )
    )


    object_connectivity = (
        calculate_object_connectivity(
            nodes,
            edges
        )
    )


    closest_pair = (
        find_closest_object_pair(
            fused_semantic_objects
        )
    )


    scene_graph = {

        "scene_graph_version":
            "1.0",

        "coordinate_system":
            (
                "registered fused "
                "relative 3D coordinates"
            ),

        "metric_scale":
            False,

        "thresholds": {

            "very_near_relative":
                VERY_NEAR_THRESHOLD,

            "near_relative":
                NEAR_THRESHOLD,

            "same_level_relative":
                SAME_LEVEL_THRESHOLD,

            "alignment_relative":
                ALIGNMENT_THRESHOLD
        },

        "statistics": {

            "total_nodes":
                len(
                    nodes
                ),

            "total_edges":
                len(
                    edges
                ),

            **relationship_statistics
        },

        "closest_object_pair":
            closest_pair,

        "object_connectivity":
            object_connectivity,

        "nodes":
            nodes,

        "edges":
            edges,

        "notice":
            (
                "Spatial relationships are "
                "computed from monocular "
                "relative-depth coordinates. "
                "Thresholds and distances are "
                "relative values, not meters."
            )
    }


    return scene_graph


# ============================================================
# SAVE SCENE GRAPH
# ============================================================

def save_scene_graph(
    fused_semantic_objects,
    output_path
):

    scene_graph = build_scene_graph(
        fused_semantic_objects
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            scene_graph,
            file,
            indent=4
        )


    return scene_graph


# ============================================================
# OPTIONAL TEST
# ============================================================

if __name__ == "__main__":

    demo_objects = [

        {
            "semantic_id":
                "OBJ-001",

            "object":
                "chair",

            "position_3d": {
                "x": 0.10,
                "y": -0.10,
                "z": -0.50
            },

            "occurrences":
                4,

            "frame_observation_count":
                4,

            "average_confidence":
                0.86,

            "maximum_confidence":
                0.93
        },


        {
            "semantic_id":
                "OBJ-002",

            "object":
                "table",

            "position_3d": {
                "x": 0.24,
                "y": -0.08,
                "z": -0.55
            },

            "occurrences":
                5,

            "frame_observation_count":
                5,

            "average_confidence":
                0.90,

            "maximum_confidence":
                0.95
        },


        {
            "semantic_id":
                "OBJ-003",

            "object":
                "tv",

            "position_3d": {
                "x": -0.35,
                "y": 0.14,
                "z": -0.72
            },

            "occurrences":
                3,

            "frame_observation_count":
                3,

            "average_confidence":
                0.81,

            "maximum_confidence":
                0.88
        }

    ]


    graph = build_scene_graph(
        demo_objects
    )


    print(
        json.dumps(
            graph,
            indent=4
        )
    )