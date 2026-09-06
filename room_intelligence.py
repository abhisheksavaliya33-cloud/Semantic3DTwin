import json


# ============================================================
# STEP 35
# ROOM-LEVEL INTELLIGENCE
# ============================================================

"""
Room Intelligence Module

This module classifies the reconstructed room
using the fused semantic objects generated in
Step 32.

It produces:

1. Predicted room type
2. Confidence score
3. Supporting detected objects
4. Competing room predictions
5. Room-purpose interpretation
6. Semantic evidence
7. Room intelligence summary

IMPORTANT:
The classification is heuristic / semantic.
It is NOT a separately trained room-classification model.
"""


# ============================================================
# ROOM KNOWLEDGE BASE
# ============================================================

ROOM_PROFILES = {

    "bedroom": {

        "strong_objects": {
            "bed": 10,
            "pillow": 6
        },

        "supporting_objects": {
            "chair": 1,
            "tv": 2,
            "laptop": 1,
            "book": 1,
            "clock": 1,
            "cell phone": 1,
            "remote": 1
        },

        "purpose":
            (
                "Primarily intended for sleeping, "
                "resting and personal activities."
            )
    },


    "living_room": {

        "strong_objects": {
            "couch": 9,
            "tv": 7
        },

        "supporting_objects": {
            "chair": 3,
            "remote": 3,
            "book": 1,
            "potted plant": 1,
            "clock": 1,
            "vase": 1
        },

        "purpose":
            (
                "Primarily intended for relaxation, "
                "entertainment and social interaction."
            )
    },


    "dining_room": {

        "strong_objects": {
            "dining table": 9
        },

        "supporting_objects": {
            "chair": 4,
            "bottle": 1,
            "cup": 1,
            "wine glass": 1,
            "fork": 2,
            "knife": 2,
            "spoon": 2,
            "bowl": 1
        },

        "purpose":
            (
                "Primarily intended for eating, "
                "serving meals and group dining."
            )
    },


    "kitchen": {

        "strong_objects": {
            "refrigerator": 10,
            "oven": 9,
            "microwave": 8,
            "toaster": 6,
            "sink": 8
        },

        "supporting_objects": {
            "bottle": 1,
            "cup": 1,
            "bowl": 1,
            "fork": 2,
            "knife": 2,
            "spoon": 2,
            "dining table": 2,
            "chair": 1
        },

        "purpose":
            (
                "Primarily intended for preparing, "
                "cooking and handling food."
            )
    },


    "office": {

        "strong_objects": {
            "laptop": 8,
            "keyboard": 7,
            "mouse": 7
        },

        "supporting_objects": {
            "chair": 4,
            "tv": 1,
            "book": 2,
            "cell phone": 1,
            "clock": 1,
            "cup": 1
        },

        "purpose":
            (
                "Primarily intended for work, "
                "study, computing and productivity."
            )
    },


    "study_room": {

        "strong_objects": {
            "book": 6,
            "laptop": 6
        },

        "supporting_objects": {
            "chair": 4,
            "cell phone": 1,
            "clock": 1,
            "cup": 1,
            "keyboard": 3,
            "mouse": 3
        },

        "purpose":
            (
                "Primarily intended for studying, "
                "reading and academic work."
            )
    },


    "bathroom": {

        "strong_objects": {
            "toilet": 10,
            "sink": 6
        },

        "supporting_objects": {
            "bottle": 1
        },

        "purpose":
            (
                "Primarily intended for hygiene "
                "and personal sanitation."
            )
    },


    "general_indoor_space": {

        "strong_objects": {},

        "supporting_objects": {
            "chair": 1,
            "potted plant": 1,
            "clock": 1,
            "book": 1
        },

        "purpose":
            (
                "An indoor space whose exact "
                "functional category cannot be "
                "determined confidently from "
                "the available semantic evidence."
            )
    }
}


# ============================================================
# ROOM LABEL
# ============================================================

ROOM_DISPLAY_NAMES = {

    "bedroom":
        "Bedroom",

    "living_room":
        "Living Room",

    "dining_room":
        "Dining Room",

    "kitchen":
        "Kitchen",

    "office":
        "Office",

    "study_room":
        "Study Room",

    "bathroom":
        "Bathroom",

    "general_indoor_space":
        "General Indoor Space"
}


# ============================================================
# NORMALIZE OBJECT NAME
# ============================================================

def normalize_object_name(
    object_name
):

    if object_name is None:

        return ""


    return (
        str(
            object_name
        )
        .strip()
        .lower()
    )


# ============================================================
# OBJECT INVENTORY
# ============================================================

def build_object_inventory(
    fused_semantic_objects
):

    inventory = {}


    for semantic_object in (
        fused_semantic_objects
    ):

        name = normalize_object_name(
            semantic_object.get(
                "object"
            )
        )


        if not name:

            continue


        inventory[name] = (
            inventory.get(
                name,
                0
            )
            + 1
        )


    return inventory


# ============================================================
# CONFIDENCE-WEIGHTED OBJECT INVENTORY
# ============================================================

def build_confidence_inventory(
    fused_semantic_objects
):

    confidence_inventory = {}


    for semantic_object in (
        fused_semantic_objects
    ):

        name = normalize_object_name(
            semantic_object.get(
                "object"
            )
        )


        confidence = float(
            semantic_object.get(
                "average_confidence",
                0.0
            )
        )


        confidence_inventory[
            name
        ] = (
            confidence_inventory.get(
                name,
                0.0
            )
            +
            confidence
        )


    return confidence_inventory


# ============================================================
# SCORE ONE ROOM
# ============================================================

def score_room_profile(
    room_name,
    profile,
    object_inventory,
    confidence_inventory
):

    raw_score = 0.0

    maximum_possible_score = 0.0

    evidence = []


    # ========================================================
    # STRONG OBJECTS
    # ========================================================

    for (
        object_name,
        weight
    ) in profile[
        "strong_objects"
    ].items():

        maximum_possible_score += (
            weight
        )


        count = object_inventory.get(
            object_name,
            0
        )


        if count <= 0:

            continue


        confidence_total = (
            confidence_inventory.get(
                object_name,
                0.0
            )
        )


        average_object_confidence = (

            confidence_total
            /
            count

            if count > 0

            else 0.0
        )


        confidence_factor = (
            0.5
            +
            (
                average_object_confidence
                *
                0.5
            )
        )


        object_score = (
            weight
            *
            confidence_factor
        )


        # Multiple instances give only
        # a small bonus.
        if count > 1:

            object_score *= min(
                1.25,
                1.0
                +
                (
                    0.05
                    *
                    (
                        count - 1
                    )
                )
            )


        raw_score += (
            object_score
        )


        evidence.append(
            {

                "object":
                    object_name,

                "count":
                    count,

                "importance":
                    "strong",

                "weight":
                    weight,

                "average_confidence":
                    round(
                        average_object_confidence,
                        4
                    ),

                "score_contribution":
                    round(
                        object_score,
                        4
                    )
            }
        )


    # ========================================================
    # SUPPORTING OBJECTS
    # ========================================================

    for (
        object_name,
        weight
    ) in profile[
        "supporting_objects"
    ].items():

        maximum_possible_score += (
            weight
        )


        count = object_inventory.get(
            object_name,
            0
        )


        if count <= 0:

            continue


        confidence_total = (
            confidence_inventory.get(
                object_name,
                0.0
            )
        )


        average_object_confidence = (

            confidence_total
            /
            count

            if count > 0

            else 0.0
        )


        confidence_factor = (
            0.5
            +
            (
                average_object_confidence
                *
                0.5
            )
        )


        object_score = (
            weight
            *
            confidence_factor
        )


        if count > 1:

            object_score *= min(
                1.20,
                1.0
                +
                (
                    0.04
                    *
                    (
                        count - 1
                    )
                )
            )


        raw_score += (
            object_score
        )


        evidence.append(
            {

                "object":
                    object_name,

                "count":
                    count,

                "importance":
                    "supporting",

                "weight":
                    weight,

                "average_confidence":
                    round(
                        average_object_confidence,
                        4
                    ),

                "score_contribution":
                    round(
                        object_score,
                        4
                    )
            }
        )


    evidence.sort(

        key=lambda item:
            item[
                "score_contribution"
            ],

        reverse=True
    )


    normalized_score = (

        raw_score
        /
        maximum_possible_score

        if maximum_possible_score > 0

        else 0.0
    )


    return {

        "room_type":
            room_name,

        "display_name":
            ROOM_DISPLAY_NAMES.get(
                room_name,
                room_name
            ),

        "raw_score":
            round(
                raw_score,
                4
            ),

        "normalized_score":
            round(
                normalized_score,
                4
            ),

        "evidence":
            evidence,

        "purpose":
            profile[
                "purpose"
            ]
    }


# ============================================================
# SPECIAL CONTEXT BONUSES
# ============================================================

def apply_context_rules(
    room_scores,
    object_inventory
):

    score_map = {

        item[
            "room_type"
        ]:
            item

        for item in room_scores
    }


    # ========================================================
    # LIVING ROOM CONTEXT
    # couch + tv
    # ========================================================

    if (
        object_inventory.get(
            "couch",
            0
        ) > 0
        and
        object_inventory.get(
            "tv",
            0
        ) > 0
    ):

        score_map[
            "living_room"
        ][
            "raw_score"
        ] += 5.0


        score_map[
            "living_room"
        ][
            "context_bonus"
        ] = (
            score_map[
                "living_room"
            ].get(
                "context_bonus",
                0.0
            )
            +
            5.0
        )


    # ========================================================
    # BEDROOM CONTEXT
    # bed dominates
    # ========================================================

    if (
        object_inventory.get(
            "bed",
            0
        ) > 0
    ):

        score_map[
            "bedroom"
        ][
            "raw_score"
        ] += 5.0


        score_map[
            "bedroom"
        ][
            "context_bonus"
        ] = (
            score_map[
                "bedroom"
            ].get(
                "context_bonus",
                0.0
            )
            +
            5.0
        )


    # ========================================================
    # DINING CONTEXT
    # table + chairs
    # ========================================================

    if (
        object_inventory.get(
            "dining table",
            0
        ) > 0
        and
        object_inventory.get(
            "chair",
            0
        ) >= 2
    ):

        score_map[
            "dining_room"
        ][
            "raw_score"
        ] += 4.0


        score_map[
            "dining_room"
        ][
            "context_bonus"
        ] = (
            score_map[
                "dining_room"
            ].get(
                "context_bonus",
                0.0
            )
            +
            4.0
        )


    # ========================================================
    # OFFICE CONTEXT
    # laptop + chair
    # ========================================================

    if (
        object_inventory.get(
            "laptop",
            0
        ) > 0
        and
        object_inventory.get(
            "chair",
            0
        ) > 0
    ):

        score_map[
            "office"
        ][
            "raw_score"
        ] += 3.0


        score_map[
            "office"
        ][
            "context_bonus"
        ] = (
            score_map[
                "office"
            ].get(
                "context_bonus",
                0.0
            )
            +
            3.0
        )


    # ========================================================
    # COMPUTER WORKSTATION
    # keyboard + mouse
    # ========================================================

    if (
        object_inventory.get(
            "keyboard",
            0
        ) > 0
        and
        object_inventory.get(
            "mouse",
            0
        ) > 0
    ):

        score_map[
            "office"
        ][
            "raw_score"
        ] += 5.0


        score_map[
            "office"
        ][
            "context_bonus"
        ] = (
            score_map[
                "office"
            ].get(
                "context_bonus",
                0.0
            )
            +
            5.0
        )


    # ========================================================
    # KITCHEN APPLIANCE COMBINATION
    # ========================================================

    kitchen_appliances = [

        "refrigerator",
        "oven",
        "microwave",
        "toaster",
        "sink"
    ]


    kitchen_hits = sum(

        1

        for item
        in kitchen_appliances

        if object_inventory.get(
            item,
            0
        ) > 0
    )


    if kitchen_hits >= 2:

        kitchen_bonus = (
            min(
                kitchen_hits,
                4
            )
            *
            2.0
        )


        score_map[
            "kitchen"
        ][
            "raw_score"
        ] += (
            kitchen_bonus
        )


        score_map[
            "kitchen"
        ][
            "context_bonus"
        ] = (
            score_map[
                "kitchen"
            ].get(
                "context_bonus",
                0.0
            )
            +
            kitchen_bonus
        )


    # ========================================================
    # BATHROOM CONTEXT
    # toilet strongly identifies bathroom
    # ========================================================

    if (
        object_inventory.get(
            "toilet",
            0
        ) > 0
    ):

        score_map[
            "bathroom"
        ][
            "raw_score"
        ] += 5.0


        score_map[
            "bathroom"
        ][
            "context_bonus"
        ] = (
            score_map[
                "bathroom"
            ].get(
                "context_bonus",
                0.0
            )
            +
            5.0
        )


    return list(
        score_map.values()
    )


# ============================================================
# CALCULATE PREDICTION CONFIDENCE
# ============================================================

def calculate_prediction_confidences(
    room_scores
):

    total_score = sum(

        max(
            score[
                "raw_score"
            ],
            0.0
        )

        for score
        in room_scores
    )


    if total_score <= 0:

        for score in room_scores:

            score[
                "prediction_confidence"
            ] = 0.0


        return room_scores


    for score in room_scores:

        confidence = (

            max(
                score[
                    "raw_score"
                ],
                0.0
            )
            /
            total_score
        )


        score[
            "prediction_confidence"
        ] = round(
            confidence,
            4
        )


    return room_scores


# ============================================================
# ROOM EVIDENCE QUALITY
# ============================================================

def determine_evidence_quality(
    predicted_room
):

    evidence = predicted_room.get(
        "evidence",
        []
    )


    strong_evidence = [

        item

        for item
        in evidence

        if item[
            "importance"
        ] == "strong"
    ]


    supporting_evidence = [

        item

        for item
        in evidence

        if item[
            "importance"
        ] == "supporting"
    ]


    if len(
        strong_evidence
    ) >= 2:

        return "strong"


    if len(
        strong_evidence
    ) == 1:

        return "moderate"


    if len(
        supporting_evidence
    ) >= 3:

        return "moderate"


    if evidence:

        return "weak"


    return "insufficient"


# ============================================================
# GENERATE HUMAN READABLE REASONING
# ============================================================

def generate_room_reasoning(
    predicted_room,
    object_inventory
):

    room_name = predicted_room[
        "display_name"
    ]


    evidence = predicted_room.get(
        "evidence",
        []
    )


    if not evidence:

        return (
            "The system could not identify enough "
            "room-specific semantic objects to "
            "classify the space confidently."
        )


    top_evidence = evidence[
        :4
    ]


    evidence_descriptions = []


    for item in top_evidence:

        object_name = item[
            "object"
        ]


        count = item[
            "count"
        ]


        if count > 1:

            description = (
                f"{count} detected "
                f"{object_name} objects"
            )

        else:

            description = (
                f"a detected "
                f"{object_name}"
            )


        evidence_descriptions.append(
            description
        )


    if len(
        evidence_descriptions
    ) == 1:

        evidence_text = (
            evidence_descriptions[0]
        )

    elif len(
        evidence_descriptions
    ) == 2:

        evidence_text = (
            evidence_descriptions[0]
            +
            " and "
            +
            evidence_descriptions[1]
        )

    else:

        evidence_text = (
            ", ".join(
                evidence_descriptions[
                    :-1
                ]
            )
            +
            ", and "
            +
            evidence_descriptions[
                -1
            ]
        )


    return (
        f"The room is classified as a "
        f"{room_name} because the fused "
        f"semantic map contains {evidence_text}. "
        f"These objects are commonly associated "
        f"with the predicted room function."
    )


# ============================================================
# DETERMINE ROOM ACTIVITIES
# ============================================================

def infer_room_activities(
    predicted_room_type,
    object_inventory
):

    activities = []


    activity_map = {

        "bedroom": [
            "sleeping",
            "resting",
            "personal activities"
        ],

        "living_room": [
            "relaxation",
            "entertainment",
            "social interaction"
        ],

        "dining_room": [
            "eating",
            "serving meals",
            "group dining"
        ],

        "kitchen": [
            "food preparation",
            "cooking",
            "food storage"
        ],

        "office": [
            "computer work",
            "professional tasks",
            "productivity"
        ],

        "study_room": [
            "reading",
            "studying",
            "academic work"
        ],

        "bathroom": [
            "personal hygiene",
            "sanitation"
        ],

        "general_indoor_space": [
            "general indoor activity"
        ]
    }


    activities.extend(

        activity_map.get(
            predicted_room_type,
            [
                "general indoor activity"
            ]
        )
    )


    # ========================================================
    # OBJECT-BASED EXTRA ACTIVITIES
    # ========================================================

    if (
        object_inventory.get(
            "tv",
            0
        ) > 0
    ):

        activities.append(
            "media viewing"
        )


    if (
        object_inventory.get(
            "laptop",
            0
        ) > 0
    ):

        activities.append(
            "computer use"
        )


    if (
        object_inventory.get(
            "book",
            0
        ) > 0
    ):

        activities.append(
            "reading"
        )


    if (
        object_inventory.get(
            "dining table",
            0
        ) > 0
    ):

        activities.append(
            "table-based activities"
        )


    if (
        object_inventory.get(
            "bed",
            0
        ) > 0
    ):

        activities.append(
            "sleeping"
        )


    # Remove duplicates
    unique_activities = []


    for activity in activities:

        if activity not in (
            unique_activities
        ):

            unique_activities.append(
                activity
            )


    return unique_activities


# ============================================================
# CLASSIFY ROOM
# ============================================================

def classify_room(
    fused_semantic_objects
):

    object_inventory = (
        build_object_inventory(
            fused_semantic_objects
        )
    )


    confidence_inventory = (
        build_confidence_inventory(
            fused_semantic_objects
        )
    )


    room_scores = []


    for (
        room_name,
        profile
    ) in ROOM_PROFILES.items():

        room_score = (
            score_room_profile(

                room_name,

                profile,

                object_inventory,

                confidence_inventory
            )
        )


        room_scores.append(
            room_score
        )


    room_scores = (
        apply_context_rules(

            room_scores,

            object_inventory
        )
    )


    room_scores = (
        calculate_prediction_confidences(
            room_scores
        )
    )


    room_scores.sort(

        key=lambda item:
            item[
                "raw_score"
            ],

        reverse=True
    )


    # ========================================================
    # NO USEFUL SEMANTIC EVIDENCE
    # ========================================================

    if (
        not room_scores
        or
        room_scores[0][
            "raw_score"
        ]
        <= 0
    ):

        predicted_room = {

            "room_type":
                "general_indoor_space",

            "display_name":
                "General Indoor Space",

            "raw_score":
                0.0,

            "normalized_score":
                0.0,

            "prediction_confidence":
                0.0,

            "evidence":
                [],

            "purpose":
                ROOM_PROFILES[
                    "general_indoor_space"
                ][
                    "purpose"
                ]
        }


    else:

        predicted_room = (
            room_scores[0]
        )


    # ========================================================
    # LOW CONFIDENCE FALLBACK
    # ========================================================

    prediction_confidence = float(
        predicted_room.get(
            "prediction_confidence",
            0.0
        )
    )


    if (
        prediction_confidence
        < 0.25
        and
        predicted_room[
            "raw_score"
        ] < 4
    ):

        predicted_room = {

            "room_type":
                "general_indoor_space",

            "display_name":
                "General Indoor Space",

            "raw_score":
                predicted_room[
                    "raw_score"
                ],

            "normalized_score":
                predicted_room[
                    "normalized_score"
                ],

            "prediction_confidence":
                prediction_confidence,

            "evidence":
                predicted_room[
                    "evidence"
                ],

            "purpose":
                ROOM_PROFILES[
                    "general_indoor_space"
                ][
                    "purpose"
                ]
        }


    evidence_quality = (
        determine_evidence_quality(
            predicted_room
        )
    )


    reasoning = (
        generate_room_reasoning(

            predicted_room,

            object_inventory
        )
    )


    inferred_activities = (
        infer_room_activities(

            predicted_room[
                "room_type"
            ],

            object_inventory
        )
    )


    # ========================================================
    # TOP ALTERNATIVE PREDICTIONS
    # ========================================================

    alternatives = []


    for score in room_scores:

        if (
            score[
                "room_type"
            ]
            ==
            predicted_room[
                "room_type"
            ]
        ):

            continue


        alternatives.append(
            {

                "room_type":
                    score[
                        "room_type"
                    ],

                "display_name":
                    score[
                        "display_name"
                    ],

                "score":
                    round(
                        score[
                            "raw_score"
                        ],
                        4
                    ),

                "confidence":
                    round(
                        score.get(
                            "prediction_confidence",
                            0.0
                        ),
                        4
                    )
            }
        )


        if len(
            alternatives
        ) >= 3:

            break


    return {

        "predicted_room": {

            "room_type":
                predicted_room[
                    "room_type"
                ],

            "display_name":
                predicted_room[
                    "display_name"
                ],

            "confidence":
                round(
                    predicted_room.get(
                        "prediction_confidence",
                        0.0
                    ),
                    4
                ),

            "score":
                round(
                    predicted_room[
                        "raw_score"
                    ],
                    4
                ),

            "evidence_quality":
                evidence_quality,

            "purpose":
                predicted_room[
                    "purpose"
                ],

            "reasoning":
                reasoning
        },


        "semantic_evidence":
            predicted_room.get(
                "evidence",
                []
            ),


        "object_inventory":
            object_inventory,


        "inferred_activities":
            inferred_activities,


        "alternative_predictions":
            alternatives,


        "all_room_scores": [

            {

                "room_type":
                    score[
                        "room_type"
                    ],

                "display_name":
                    score[
                        "display_name"
                    ],

                "score":
                    round(
                        score[
                            "raw_score"
                        ],
                        4
                    ),

                "confidence":
                    round(
                        score.get(
                            "prediction_confidence",
                            0.0
                        ),
                        4
                    ),

                "context_bonus":
                    round(
                        score.get(
                            "context_bonus",
                            0.0
                        ),
                        4
                    )
            }

            for score in room_scores
        ],


        "classification_method":
            (
                "Semantic object weighted "
                "heuristic classification"
            ),


        "model_based_classifier":
            False,


        "notice":
            (
                "The predicted room type is inferred "
                "from detected semantic objects and "
                "rule-based object associations. "
                "It is not produced by a separately "
                "trained room-classification neural "
                "network."
            )
    }


# ============================================================
# COMPLETE ROOM INTELLIGENCE
# ============================================================

def build_room_intelligence(
    fused_semantic_objects,
    scene_graph=None
):

    classification = (
        classify_room(
            fused_semantic_objects
        )
    )


    scene_statistics = {}


    if scene_graph:

        scene_statistics = (
            scene_graph.get(
                "statistics",
                {}
            )
        )


    most_connected_object = None


    if scene_graph:

        connectivity = (
            scene_graph.get(
                "object_connectivity",
                []
            )
        )


        if connectivity:

            most_connected_object = (
                connectivity[0]
            )


    room_intelligence = {

        "room_intelligence_version":
            "1.0",

        "predicted_room":
            classification[
                "predicted_room"
            ],

        "classification_method":
            classification[
                "classification_method"
            ],

        "model_based_classifier":
            classification[
                "model_based_classifier"
            ],

        "semantic_evidence":
            classification[
                "semantic_evidence"
            ],

        "object_inventory":
            classification[
                "object_inventory"
            ],

        "inferred_activities":
            classification[
                "inferred_activities"
            ],

        "alternative_predictions":
            classification[
                "alternative_predictions"
            ],

        "all_room_scores":
            classification[
                "all_room_scores"
            ],

        "scene_context": {

            "semantic_object_count":
                len(
                    fused_semantic_objects
                ),

            "scene_graph_nodes":
                scene_statistics.get(
                    "total_nodes",
                    0
                ),

            "scene_graph_edges":
                scene_statistics.get(
                    "total_edges",
                    0
                ),

            "relationship_types":
                scene_statistics.get(
                    "relationship_types",
                    0
                ),

            "most_connected_object":
                most_connected_object
        },

        "notice":
            classification[
                "notice"
            ]
    }


    return room_intelligence


# ============================================================
# SAVE ROOM INTELLIGENCE
# ============================================================

def save_room_intelligence(
    fused_semantic_objects,
    output_path,
    scene_graph=None
):

    room_intelligence = (
        build_room_intelligence(

            fused_semantic_objects,

            scene_graph
        )
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            room_intelligence,
            file,
            indent=4
        )


    return room_intelligence


# ============================================================
# TEST MODULE
# ============================================================

if __name__ == "__main__":

    demo_objects = [

        {
            "semantic_id":
                "OBJ-001",

            "object":
                "couch",

            "average_confidence":
                0.91,

            "position_3d": {
                "x": 0.1,
                "y": -0.1,
                "z": -0.6
            }
        },


        {
            "semantic_id":
                "OBJ-002",

            "object":
                "tv",

            "average_confidence":
                0.88,

            "position_3d": {
                "x": -0.2,
                "y": 0.1,
                "z": -0.8
            }
        },


        {
            "semantic_id":
                "OBJ-003",

            "object":
                "chair",

            "average_confidence":
                0.84,

            "position_3d": {
                "x": 0.3,
                "y": -0.1,
                "z": -0.5
            }
        },


        {
            "semantic_id":
                "OBJ-004",

            "object":
                "remote",

            "average_confidence":
                0.77,

            "position_3d": {
                "x": 0.15,
                "y": -0.05,
                "z": -0.57
            }
        }

    ]


    result = build_room_intelligence(
        demo_objects
    )


    print(
        json.dumps(
            result,
            indent=4
        )
    )