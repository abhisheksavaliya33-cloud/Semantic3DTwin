import argparse
import ast
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime


# ============================================================
# STEP 37
# FINAL PROJECT TESTING, VALIDATION & CLEANUP
# ============================================================

"""
Semantic 3D Indoor Digital Twin
Step 37 - Final Project Testing, Validation & Cleanup

Run:

    python final_validation.py

Optional safe cleanup:

    python final_validation.py --cleanup

This script does NOT run YOLO, Depth Anything V2, Open3D
registration, or Flask itself. It validates the project structure,
Python syntax, required imports/packages, Flask routes, template
integration, generated output files, JSON readability, and final
pipeline completeness.

The --cleanup option only removes Python cache artifacts:
    __pycache__/
    *.pyc
    *.pyo

It does NOT delete uploaded videos, generated point clouds,
reports, semantic outputs, or trained/downloaded model files.
"""


PROJECT_NAME = "Semantic 3D Indoor Digital Twin"
VALIDATION_VERSION = "1.0"


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUTS_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

VALIDATION_OUTPUT_DIR = os.path.join(
    OUTPUTS_DIR,
    "validation"
)

VALIDATION_REPORT_PATH = os.path.join(
    VALIDATION_OUTPUT_DIR,
    "final_validation_report.json"
)


# ============================================================
# REQUIRED PROJECT FILES
# ============================================================

REQUIRED_FILES = [
    "app.py",
    "spatial_intelligence.py",
    "room_intelligence.py",
    "final_report.py",
    os.path.join(
        "templates",
        "video_results.html"
    )
]


# ============================================================
# EXPECTED OUTPUT DIRECTORIES
# ============================================================

EXPECTED_OUTPUT_DIRECTORIES = [
    os.path.join(
        "outputs",
        "video_frames"
    ),
    os.path.join(
        "outputs",
        "video_detections"
    ),
    os.path.join(
        "outputs",
        "video_depth"
    ),
    os.path.join(
        "outputs",
        "video_pointclouds"
    ),
    os.path.join(
        "outputs",
        "video_fusion"
    ),
    os.path.join(
        "outputs",
        "video_transforms"
    ),
    os.path.join(
        "outputs",
        "video_semantic_twin"
    ),
    os.path.join(
        "outputs",
        "scene_graphs"
    ),
    os.path.join(
        "outputs",
        "room_intelligence"
    ),
    os.path.join(
        "outputs",
        "final_reports"
    )
]


# ============================================================
# REQUIRED PYTHON PACKAGES
# ============================================================

REQUIRED_PACKAGES = {
    "flask":
        "Flask",

    "cv2":
        "OpenCV",

    "torch":
        "PyTorch",

    "numpy":
        "NumPy",

    "open3d":
        "Open3D",

    "PIL":
        "Pillow",

    "ultralytics":
        "Ultralytics YOLO",

    "transformers":
        "Transformers"
}


# ============================================================
# APP.PY EXPECTATIONS
# ============================================================

EXPECTED_APP_IMPORT_MARKERS = [
    "from spatial_intelligence import",
    "from room_intelligence import",
    "from final_report import"
]

EXPECTED_APP_ROUTE_MARKERS = [
    "/scene-graphs/<filename>",
    "/room-intelligence/<filename>",
    "/final-reports/<filename>"
]

EXPECTED_APP_FUNCTION_MARKERS = [
    "def generate_scene_graph_file",
    "def generate_room_intelligence_file",
    "def process_video"
]

# Step 36 can be integrated either through a helper function
# or by directly calling save_final_digital_twin_report(...)
# inside process_video().
EXPECTED_STEP36_IMPLEMENTATION_MARKERS = [
    "save_final_digital_twin_report(",
    "def generate_final_report_file"
]

EXPECTED_PIPELINE_MARKERS = [
    "STEP 31",
    "STEP 32",
    "STEP 33",
    "STEP 35",
    "STEP 36"
]


# ============================================================
# TEMPLATE EXPECTATIONS
# ============================================================

EXPECTED_TEMPLATE_MARKERS = [
    "STEP 36 COMPLETE",
    "Final Output",
    "Process Overview",
    "3D Digital Twin",
    "Room Intelligence",
    "Final Digital Twin Report",
    "serve_room_intelligence",
    "serve_final_report"
]


# ============================================================
# FINAL REPORT EXPECTATIONS
# ============================================================

FINAL_REPORT_REQUIRED_KEYS = [
    "project",
    "source",
    "pipeline",
    "reconstruction",
    "semantic_intelligence",
    "spatial_intelligence",
    "room_intelligence",
    "asset_manifest",
    "quality_summary",
    "technical_stack",
    "limitations"
]


# ============================================================
# RESULT HELPERS
# ============================================================

def make_result(
    name,
    status,
    message,
    details=None
):
    return {
        "name":
            name,

        "status":
            status,

        "message":
            message,

        "details":
            details
    }


def status_symbol(
    status
):
    mapping = {
        "PASS":
            "[PASS]",

        "WARN":
            "[WARN]",

        "FAIL":
            "[FAIL]",

        "INFO":
            "[INFO]"
    }

    return mapping.get(
        status,
        "[INFO]"
    )


# ============================================================
# FILE HELPERS
# ============================================================

def read_text_file(
    path
):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


def list_json_files(
    directory
):
    if not os.path.isdir(
        directory
    ):
        return []

    return sorted(
        [
            os.path.join(
                directory,
                filename
            )
            for filename
            in os.listdir(
                directory
            )
            if filename.lower().endswith(
                ".json"
            )
        ],
        key=os.path.getmtime,
        reverse=True
    )


# ============================================================
# TEST 1
# PROJECT FILE STRUCTURE
# ============================================================

def validate_required_files():
    missing = []
    present = []

    for relative_path in REQUIRED_FILES:

        absolute_path = os.path.join(
            BASE_DIR,
            relative_path
        )

        if os.path.isfile(
            absolute_path
        ):
            present.append(
                relative_path
            )

        else:
            missing.append(
                relative_path
            )

    if missing:
        return make_result(
            "Required project files",
            "FAIL",
            (
                "One or more required project files "
                "are missing."
            ),
            {
                "present":
                    present,

                "missing":
                    missing
            }
        )

    return make_result(
        "Required project files",
        "PASS",
        "All required Step 33-37 project files are present.",
        {
            "files":
                present
        }
    )


# ============================================================
# TEST 2
# PYTHON SYNTAX
# ============================================================

def validate_python_syntax():
    python_files = [
        "app.py",
        "spatial_intelligence.py",
        "room_intelligence.py",
        "final_report.py",
        "final_validation.py"
    ]

    errors = []
    checked = []

    for filename in python_files:

        path = os.path.join(
            BASE_DIR,
            filename
        )

        if not os.path.isfile(
            path
        ):
            errors.append(
                {
                    "file":
                        filename,

                    "error":
                        "File not found."
                }
            )
            continue

        try:
            source = read_text_file(
                path
            )

            ast.parse(
                source,
                filename=filename
            )

            checked.append(
                filename
            )

        except SyntaxError as error:

            errors.append(
                {
                    "file":
                        filename,

                    "line":
                        error.lineno,

                    "column":
                        error.offset,

                    "error":
                        error.msg
                }
            )

        except Exception as error:

            errors.append(
                {
                    "file":
                        filename,

                    "error":
                        str(error)
                }
            )

    if errors:
        return make_result(
            "Python syntax",
            "FAIL",
            "Python syntax validation found errors.",
            {
                "checked":
                    checked,

                "errors":
                    errors
            }
        )

    return make_result(
        "Python syntax",
        "PASS",
        "All project Python files passed AST syntax validation.",
        {
            "checked":
                checked
        }
    )


# ============================================================
# TEST 3
# REQUIRED PACKAGES
# ============================================================

def validate_required_packages():
    installed = {}
    missing = {}

    for module_name, display_name in REQUIRED_PACKAGES.items():

        try:
            spec = importlib.util.find_spec(
                module_name
            )

        except Exception:
            spec = None

        if spec is None:
            missing[
                module_name
            ] = display_name

        else:
            installed[
                module_name
            ] = display_name

    if missing:
        return make_result(
            "Python packages",
            "FAIL",
            (
                "One or more required Python packages "
                "are not available in the active environment."
            ),
            {
                "installed":
                    installed,

                "missing":
                    missing,

                "python_executable":
                    sys.executable
            }
        )

    return make_result(
        "Python packages",
        "PASS",
        "All required Python packages are available.",
        {
            "installed":
                installed,

            "python_executable":
                sys.executable
        }
    )


# ============================================================
# TEST 4
# APP.PY INTEGRATION
# ============================================================

def validate_app_integration():
    app_path = os.path.join(
        BASE_DIR,
        "app.py"
    )

    if not os.path.isfile(
        app_path
    ):
        return make_result(
            "app.py integration",
            "FAIL",
            "app.py was not found."
        )

    source = read_text_file(
        app_path
    )

    missing_imports = [
        marker
        for marker
        in EXPECTED_APP_IMPORT_MARKERS
        if marker not in source
    ]

    missing_routes = [
        marker
        for marker
        in EXPECTED_APP_ROUTE_MARKERS
        if marker not in source
    ]

    missing_functions = [
        marker
        for marker
        in EXPECTED_APP_FUNCTION_MARKERS
        if marker not in source
    ]

    has_step36_implementation = any(
        marker in source
        for marker
        in EXPECTED_STEP36_IMPLEMENTATION_MARKERS
    )

    missing_pipeline_markers = [
        marker
        for marker
        in EXPECTED_PIPELINE_MARKERS
        if marker not in source
    ]

    issues = {
        "missing_import_markers":
            missing_imports,

        "missing_routes":
            missing_routes,

        "missing_functions":
            missing_functions,

        "missing_step36_implementation":
            [] if has_step36_implementation else EXPECTED_STEP36_IMPLEMENTATION_MARKERS,

        "missing_pipeline_markers":
            missing_pipeline_markers
    }

    has_issues = any(
        issues.values()
    )

    if has_issues:
        return make_result(
            "app.py integration",
            "FAIL",
            (
                "app.py is missing one or more expected "
                "pipeline integrations."
            ),
            issues
        )

    return make_result(
        "app.py integration",
        "PASS",
        (
            "app.py contains the expected Step 33, "
            "Step 35 and Step 36 integrations."
        ),
        {
            "scene_graph_route":
                True,

            "room_intelligence_route":
                True,

            "final_report_route":
                True
        }
    )


# ============================================================
# TEST 5
# DASHBOARD TEMPLATE
# ============================================================

def validate_dashboard_template():
    template_path = os.path.join(
        BASE_DIR,
        "templates",
        "video_results.html"
    )

    if not os.path.isfile(
        template_path
    ):
        return make_result(
            "Video dashboard",
            "FAIL",
            "templates/video_results.html was not found."
        )

    source = read_text_file(
        template_path
    )

    missing = [
        marker
        for marker
        in EXPECTED_TEMPLATE_MARKERS
        if marker not in source
    ]

    if missing:
        return make_result(
            "Video dashboard",
            "FAIL",
            (
                "The simplified final dashboard is missing one or more "
                "required Step 36 output/process integration markers."
            ),
            {
                "missing":
                    missing
            }
        )

    return make_result(
        "Video dashboard",
        "PASS",
        (
            "The simplified final dashboard contains the expected "
            "Final Output, Process Overview, 3D Digital Twin, "
            "Room Intelligence and Final Report integrations."
        )
    )


# ============================================================
# TEST 6
# OUTPUT DIRECTORY STRUCTURE
# ============================================================

def validate_output_directories():
    present = []
    missing = []

    for relative_path in EXPECTED_OUTPUT_DIRECTORIES:

        absolute_path = os.path.join(
            BASE_DIR,
            relative_path
        )

        if os.path.isdir(
            absolute_path
        ):
            present.append(
                relative_path
            )

        else:
            missing.append(
                relative_path
            )

    if missing:
        return make_result(
            "Output directories",
            "WARN",
            (
                "Some output directories are not present yet. "
                "This can be normal before the corresponding "
                "pipeline stage has processed a video."
            ),
            {
                "present":
                    present,

                "missing":
                    missing
            }
        )

    return make_result(
        "Output directories",
        "PASS",
        "All expected video output directories exist.",
        {
            "directories":
                present
        }
    )


# ============================================================
# TEST 7
# YOLO MODEL FILE
# ============================================================

def validate_yolo_model_file():
    model_path = os.path.join(
        BASE_DIR,
        "yolov8n.pt"
    )

    if os.path.isfile(
        model_path
    ):
        return make_result(
            "YOLO model",
            "PASS",
            "Local yolov8n.pt model file is present.",
            {
                "path":
                    model_path,

                "size_bytes":
                    os.path.getsize(
                        model_path
                    )
            }
        )

    return make_result(
        "YOLO model",
        "WARN",
        (
            "yolov8n.pt is not stored in the project root. "
            "Ultralytics may download it automatically when "
            "the application starts if network access is available."
        ),
        {
            "expected_path":
                model_path
        }
    )


# ============================================================
# TEST 8
# GENERATED JSON OUTPUTS
# ============================================================

def validate_generated_json_outputs():
    output_groups = {
        "semantic_twin":
            os.path.join(
                BASE_DIR,
                "outputs",
                "video_semantic_twin"
            ),

        "scene_graph":
            os.path.join(
                BASE_DIR,
                "outputs",
                "scene_graphs"
            ),

        "room_intelligence":
            os.path.join(
                BASE_DIR,
                "outputs",
                "room_intelligence"
            ),

        "final_reports":
            os.path.join(
                BASE_DIR,
                "outputs",
                "final_reports"
            )
    }

    summary = {}
    invalid_files = []

    for group_name, directory in output_groups.items():

        files = list_json_files(
            directory
        )

        summary[
            group_name
        ] = {
            "count":
                len(files),

            "latest_file":
                (
                    os.path.basename(
                        files[0]
                    )
                    if files
                    else None
                )
        }

        for path in files[:5]:

            try:
                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as file:
                    json.load(
                        file
                    )

            except Exception as error:

                invalid_files.append(
                    {
                        "file":
                            path,

                        "error":
                            str(error)
                    }
                )

    if invalid_files:
        return make_result(
            "Generated JSON outputs",
            "FAIL",
            "One or more generated JSON files are invalid.",
            {
                "summary":
                    summary,

                "invalid_files":
                    invalid_files
            }
        )

    final_count = summary[
        "final_reports"
    ][
        "count"
    ]

    if final_count == 0:
        return make_result(
            "Generated JSON outputs",
            "WARN",
            (
                "No Step 36 final report has been generated yet. "
                "Process a video once with the final application."
            ),
            {
                "summary":
                    summary
            }
        )

    return make_result(
        "Generated JSON outputs",
        "PASS",
        (
            "Generated semantic, scene, room and final-report "
            "JSON files are readable."
        ),
        {
            "summary":
                summary
        }
    )


# ============================================================
# TEST 9
# LATEST FINAL REPORT SCHEMA
# ============================================================

def validate_latest_final_report():
    directory = os.path.join(
        BASE_DIR,
        "outputs",
        "final_reports"
    )

    files = list_json_files(
        directory
    )

    if not files:
        return make_result(
            "Latest final report schema",
            "WARN",
            (
                "No final digital twin report exists yet. "
                "Run one video through Steps 31-36 first."
            )
        )

    latest_path = files[0]

    try:
        with open(
            latest_path,
            "r",
            encoding="utf-8"
        ) as file:
            report = json.load(
                file
            )

    except Exception as error:
        return make_result(
            "Latest final report schema",
            "FAIL",
            "The latest final report cannot be read.",
            {
                "file":
                    latest_path,

                "error":
                    str(error)
            }
        )

    missing_keys = [
        key
        for key
        in FINAL_REPORT_REQUIRED_KEYS
        if key not in report
    ]

    if missing_keys:
        return make_result(
            "Latest final report schema",
            "FAIL",
            (
                "The latest final report is missing "
                "required top-level sections."
            ),
            {
                "file":
                    latest_path,

                "missing_keys":
                    missing_keys
            }
        )

    quality_summary = report.get(
        "quality_summary",
        {}
    )

    completeness = quality_summary.get(
        "pipeline_completeness_percent"
    )

    return make_result(
        "Latest final report schema",
        "PASS",
        "The latest final report contains all required sections.",
        {
            "file":
                latest_path,

            "pipeline_completeness_percent":
                completeness
        }
    )


# ============================================================
# TEST 10
# RELATIVE DEPTH / NON-METRIC SAFETY
# ============================================================

def validate_non_metric_disclaimers():
    files_to_check = [
        os.path.join(
            BASE_DIR,
            "app.py"
        ),
        os.path.join(
            BASE_DIR,
            "final_report.py"
        ),
        os.path.join(
            BASE_DIR,
            "templates",
            "video_results.html"
        )
    ]

    evidence = {}
    failures = []

    for path in files_to_check:

        relative_path = os.path.relpath(
            path,
            BASE_DIR
        )

        if not os.path.isfile(
            path
        ):
            failures.append(
                relative_path
            )
            continue

        source = read_text_file(
            path
        ).lower()

        has_relative = (
            "relative" in source
        )

        has_metric_warning = (
            "metric_scale" in source
            or
            "not meters" in source
            or
            "not metric" in source
            or
            "relative depth" in source
        )

        evidence[
            relative_path
        ] = {
            "mentions_relative":
                has_relative,

            "contains_non_metric_notice":
                has_metric_warning
        }

        if not (
            has_relative
            and
            has_metric_warning
        ):
            failures.append(
                relative_path
            )

    if failures:
        return make_result(
            "Relative-depth safety notices",
            "WARN",
            (
                "One or more project files do not clearly "
                "contain relative/non-metric reconstruction notices."
            ),
            {
                "evidence":
                    evidence,

                "needs_review":
                    failures
            }
        )

    return make_result(
        "Relative-depth safety notices",
        "PASS",
        (
            "The project clearly identifies monocular depth "
            "and 3D coordinates as relative/non-metric."
        ),
        {
            "evidence":
                evidence
        }
    )


# ============================================================
# SAFE CLEANUP
# ============================================================

def cleanup_python_cache():
    removed = []
    errors = []

    for root, directories, files in os.walk(
        BASE_DIR
    ):

        if ".git" in directories:
            directories.remove(
                ".git"
            )

        cache_directories = [
            directory
            for directory
            in list(
                directories
            )
            if directory == "__pycache__"
        ]

        for directory in cache_directories:

            path = os.path.join(
                root,
                directory
            )

            try:
                shutil.rmtree(
                    path
                )

                removed.append(
                    path
                )

                directories.remove(
                    directory
                )

            except Exception as error:

                errors.append(
                    {
                        "path":
                            path,

                        "error":
                            str(error)
                    }
                )

        for filename in files:

            if filename.endswith(
                (
                    ".pyc",
                    ".pyo"
                )
            ):

                path = os.path.join(
                    root,
                    filename
                )

                try:
                    os.remove(
                        path
                    )

                    removed.append(
                        path
                    )

                except Exception as error:

                    errors.append(
                        {
                            "path":
                                path,

                            "error":
                                str(error)
                        }
                    )

    if errors:
        return make_result(
            "Safe cleanup",
            "WARN",
            (
                "Cleanup completed with some cache files "
                "that could not be removed."
            ),
            {
                "removed_count":
                    len(removed),

                "removed":
                    removed,

                "errors":
                    errors
            }
        )

    return make_result(
        "Safe cleanup",
        "PASS",
        (
            "Python cache cleanup completed. "
            "No project data or generated outputs were deleted."
        ),
        {
            "removed_count":
                len(removed),

            "removed":
                removed
        }
    )


# ============================================================
# OVERALL STATUS
# ============================================================

def calculate_overall_status(
    results
):
    fail_count = sum(
        1
        for result
        in results
        if result[
            "status"
        ] == "FAIL"
    )

    warning_count = sum(
        1
        for result
        in results
        if result[
            "status"
        ] == "WARN"
    )

    pass_count = sum(
        1
        for result
        in results
        if result[
            "status"
        ] == "PASS"
    )

    if fail_count > 0:
        status = "FAIL"

    elif warning_count > 0:
        status = "PASS_WITH_WARNINGS"

    else:
        status = "PASS"

    return {
        "status":
            status,

        "pass_count":
            pass_count,

        "warning_count":
            warning_count,

        "fail_count":
            fail_count,

        "total_checks":
            len(
                results
            )
    }


# ============================================================
# VALIDATION RUNNER
# ============================================================

def run_validation(
    cleanup=False
):
    results = []

    tests = [
        validate_required_files,
        validate_python_syntax,
        validate_required_packages,
        validate_app_integration,
        validate_dashboard_template,
        validate_output_directories,
        validate_yolo_model_file,
        validate_generated_json_outputs,
        validate_latest_final_report,
        validate_non_metric_disclaimers
    ]

    for test in tests:

        try:
            result = test()

        except Exception as error:

            result = make_result(
                test.__name__,
                "FAIL",
                "Unexpected validation error.",
                {
                    "error":
                        str(error)
                }
            )

        results.append(
            result
        )

    if cleanup:
        results.append(
            cleanup_python_cache()
        )

    overall = calculate_overall_status(
        results
    )

    report = {
        "project":
            PROJECT_NAME,

        "validation_version":
            VALIDATION_VERSION,

        "step":
            "Step 37 - Final Project Testing, Validation & Cleanup",

        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "base_directory":
            BASE_DIR,

        "python_version":
            sys.version,

        "python_executable":
            sys.executable,

        "overall":
            overall,

        "checks":
            results,

        "submission_readiness": {
            "code_structure_valid":
                overall[
                    "fail_count"
                ] == 0,

            "recommended_final_actions": [
                (
                    "Run one representative room video "
                    "through the complete application."
                ),
                (
                    "Confirm the final dashboard loads the "
                    "merged 3D point cloud and semantic markers."
                ),
                (
                    "Confirm the simplified Process Overview and "
                    "Final Output sections display correctly."
                ),
                (
                    "Confirm Room Intelligence displays a "
                    "reasonable semantic room prediction."
                ),
                (
                    "Download and open the Step 36 final "
                    "digital twin JSON report."
                ),
                (
                    "Keep the relative-depth/non-metric "
                    "limitation in the final report and presentation."
                )
            ]
        }
    }

    os.makedirs(
        VALIDATION_OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        VALIDATION_REPORT_PATH,
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
# TERMINAL OUTPUT
# ============================================================

def print_report(
    report
):
    print()
    print(
        "============================================================"
    )
    print(
        " STEP 37 - FINAL PROJECT VALIDATION"
    )
    print(
        "============================================================"
    )
    print()

    for result in report[
        "checks"
    ]:

        print(
            f"{status_symbol(result['status'])} "
            f"{result['name']}"
        )

        print(
            f"       {result['message']}"
        )

        print()

    overall = report[
        "overall"
    ]

    print(
        "------------------------------------------------------------"
    )

    print(
        "Overall status :",
        overall[
            "status"
        ]
    )

    print(
        "Passed         :",
        overall[
            "pass_count"
        ]
    )

    print(
        "Warnings       :",
        overall[
            "warning_count"
        ]
    )

    print(
        "Failed         :",
        overall[
            "fail_count"
        ]
    )

    print(
        "Total checks   :",
        overall[
            "total_checks"
        ]
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        "Validation report:"
    )

    print(
        VALIDATION_REPORT_PATH
    )

    print()

    if overall[
        "status"
    ] == "PASS":

        print(
            "PROJECT STATUS: READY FOR FINAL SUBMISSION"
        )

    elif overall[
        "status"
    ] == "PASS_WITH_WARNINGS":

        print(
            "PROJECT STATUS: FUNCTIONALLY READY - REVIEW WARNINGS"
        )

    else:

        print(
            "PROJECT STATUS: FIX FAILED CHECKS BEFORE SUBMISSION"
        )

    print()

    print(
        "IMPORTANT: reconstructed depth and 3D coordinates "
        "remain relative/non-metric."
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Semantic 3D Indoor "
            "Digital Twin project."
        )
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "Safely remove Python __pycache__, "
            ".pyc and .pyo files."
        )
    )

    arguments = parser.parse_args()

    report = run_validation(
        cleanup=arguments.cleanup
    )

    print_report(
        report
    )


if __name__ == "__main__":
    main()
