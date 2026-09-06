import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime


# ============================================================
# STEP 39 - FINAL SUBMISSION PACKAGING
# Semantic 3D Reconstruction for Indoor Digital Twins
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_NAME = "Semantic3DTwin"
PACKAGE_NAME = "Semantic3DTwin_Final_Submission"

SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
PACKAGE_DIR = os.path.join(SUBMISSION_DIR, PACKAGE_NAME)
ZIP_PATH = os.path.join(SUBMISSION_DIR, PACKAGE_NAME + ".zip")


# ============================================================
# FILES THAT MUST BE INCLUDED
# ============================================================

REQUIRED_FILES = [
    "app.py",
    "spatial_intelligence.py",
    "room_intelligence.py",
    "final_report.py",
    "final_validation.py",
    "README.md",
    "requirements.txt",
    "yolov8n.pt",
]

REQUIRED_TEMPLATES = [
    "index.html",
    "results.html",
    "video_results.html",
]


# ============================================================
# OUTPUT FOLDERS WORTH KEEPING IN SUBMISSION
# ============================================================

OUTPUT_FOLDERS = [
    "detections",
    "depth",
    "pointclouds",
    "reports",
    "digital_twin",
    "video_frames",
    "video_detections",
    "video_depth",
    "video_pointclouds",
    "video_fusion",
    "video_transforms",
    "video_semantic_twin",
    "scene_graphs",
    "room_intelligence",
    "final_reports",
    "validation",
]


# ============================================================
# EXCLUSIONS
# ============================================================

EXCLUDED_DIRECTORY_NAMES = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "submission",
}

EXCLUDED_FILE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
}

EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def safe_remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def copy_file(relative_path, required=True):
    source = os.path.join(BASE_DIR, relative_path)
    destination = os.path.join(PACKAGE_DIR, relative_path)

    if not os.path.isfile(source):
        if required:
            raise FileNotFoundError(
                f"Required file is missing: {relative_path}"
            )
        return False

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    return True


def should_exclude_file(filename):
    if filename in EXCLUDED_FILE_NAMES:
        return True

    extension = os.path.splitext(filename)[1].lower()

    return extension in EXCLUDED_FILE_EXTENSIONS


def copy_directory_filtered(source, destination):
    if not os.path.isdir(source):
        return 0

    copied = 0

    for root, directories, files in os.walk(source):
        directories[:] = [
            directory
            for directory in directories
            if directory not in EXCLUDED_DIRECTORY_NAMES
        ]

        relative_root = os.path.relpath(root, source)

        if relative_root == ".":
            target_root = destination
        else:
            target_root = os.path.join(destination, relative_root)

        os.makedirs(target_root, exist_ok=True)

        for filename in files:
            if should_exclude_file(filename):
                continue

            source_file = os.path.join(root, filename)
            target_file = os.path.join(target_root, filename)

            shutil.copy2(source_file, target_file)
            copied += 1

    return copied


# ============================================================
# PRE-SUBMISSION VALIDATION
# ============================================================

def run_validation():
    validation_script = os.path.join(
        BASE_DIR,
        "final_validation.py"
    )

    if not os.path.isfile(validation_script):
        return False, "final_validation.py is missing."

    print_header("1. RUNNING FINAL VALIDATION")

    result = subprocess.run(
        [sys.executable, validation_script],
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    validation_report = os.path.join(
        BASE_DIR,
        "outputs",
        "validation",
        "final_validation_report.json"
    )

    if not os.path.isfile(validation_report):
        return False, (
            "Validation report was not generated. "
            "Check final_validation.py."
        )

    import json

    try:
        with open(
            validation_report,
            "r",
            encoding="utf-8"
        ) as file:
            report = json.load(file)

        status = (
            report
            .get("overall", {})
            .get("status", "UNKNOWN")
        )

    except Exception as error:
        return False, (
            "Could not read validation report: "
            + str(error)
        )

    if status == "FAIL":
        return False, (
            "Final validation contains failed checks. "
            "Fix them before packaging."
        )

    if status == "PASS_WITH_WARNINGS":
        return True, (
            "Validation passed with warnings. "
            "Review warnings before submission."
        )

    if status == "PASS":
        return True, "Validation passed."

    return False, (
        f"Unknown validation status: {status}"
    )


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

def check_required_files():
    print_header("2. CHECKING REQUIRED PROJECT FILES")

    missing = []

    for relative_path in REQUIRED_FILES:
        path = os.path.join(BASE_DIR, relative_path)

        if os.path.isfile(path):
            print("[PASS]", relative_path)
        else:
            print("[FAIL]", relative_path)
            missing.append(relative_path)

    templates_directory = os.path.join(
        BASE_DIR,
        "templates"
    )

    for filename in REQUIRED_TEMPLATES:
        relative_path = os.path.join(
            "templates",
            filename
        )

        path = os.path.join(
            templates_directory,
            filename
        )

        if os.path.isfile(path):
            print("[PASS]", relative_path)
        else:
            print("[FAIL]", relative_path)
            missing.append(relative_path)

    return missing


# ============================================================
# BUILD CLEAN PACKAGE
# ============================================================

def build_package(include_outputs=True):
    print_header("3. BUILDING CLEAN SUBMISSION FOLDER")

    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    safe_remove(PACKAGE_DIR)
    os.makedirs(PACKAGE_DIR, exist_ok=True)

    copied_count = 0

    for relative_path in REQUIRED_FILES:
        if copy_file(relative_path):
            copied_count += 1

    # Templates
    templates_source = os.path.join(
        BASE_DIR,
        "templates"
    )

    templates_destination = os.path.join(
        PACKAGE_DIR,
        "templates"
    )

    copied_count += copy_directory_filtered(
        templates_source,
        templates_destination
    )

    # Static assets, if the project has any
    static_source = os.path.join(
        BASE_DIR,
        "static"
    )

    static_destination = os.path.join(
        PACKAGE_DIR,
        "static"
    )

    copied_count += copy_directory_filtered(
        static_source,
        static_destination
    )

    # Keep uploads directory empty in final package.
    os.makedirs(
        os.path.join(
            PACKAGE_DIR,
            "uploads"
        ),
        exist_ok=True
    )

    # Preserve output directory structure.
    package_outputs = os.path.join(
        PACKAGE_DIR,
        "outputs"
    )

    os.makedirs(
        package_outputs,
        exist_ok=True
    )

    for folder_name in OUTPUT_FOLDERS:
        source = os.path.join(
            BASE_DIR,
            "outputs",
            folder_name
        )

        destination = os.path.join(
            package_outputs,
            folder_name
        )

        os.makedirs(
            destination,
            exist_ok=True
        )

        if include_outputs:
            copied_count += copy_directory_filtered(
                source,
                destination
            )

    # Add submission information.
    submission_info = f"""Semantic 3D Reconstruction for Indoor Digital Twins
Final Submission Package

Package created: {datetime.now().isoformat(timespec="seconds")}

Recommended environment:
Python 3.11.x
Conda environment: semantic3d

Run:
    conda activate semantic3d
    pip install -r requirements.txt
    python final_validation.py
    python app.py

Important:
The system uses monocular relative depth.
Reconstructed 3D coordinates are approximate/non-metric
and must not be interpreted as accurate measurements in meters.

Pipeline:
Step 31 - Multi-frame registration and fusion
Step 32 - Semantic fusion
Step 33 - Semantic scene graph
Step 34 - Interactive dashboard
Step 35 - Room intelligence
Step 36 - Final digital twin report
Step 37 - Validation and cleanup
Step 38 - Documentation
Step 39 - Final submission packaging
"""

    info_path = os.path.join(
        PACKAGE_DIR,
        "SUBMISSION_INFO.txt"
    )

    with open(
        info_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(submission_info)

    copied_count += 1

    print(
        f"Clean package created with approximately "
        f"{copied_count} files."
    )

    print("Package folder:")
    print(PACKAGE_DIR)


# ============================================================
# ZIP CREATION
# ============================================================

def create_zip():
    print_header("4. CREATING FINAL ZIP")

    safe_remove(ZIP_PATH)

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6
    ) as archive:

        for root, directories, files in os.walk(
            PACKAGE_DIR
        ):
            directories[:] = [
                directory
                for directory in directories
                if directory not in EXCLUDED_DIRECTORY_NAMES
            ]

            for filename in files:
                if should_exclude_file(filename):
                    continue

                full_path = os.path.join(
                    root,
                    filename
                )

                relative_path = os.path.relpath(
                    full_path,
                    SUBMISSION_DIR
                )

                archive.write(
                    full_path,
                    relative_path
                )

    size_mb = os.path.getsize(
        ZIP_PATH
    ) / (1024 * 1024)

    print("ZIP created:")
    print(ZIP_PATH)
    print(f"ZIP size: {size_mb:.2f} MB")


# ============================================================
# VERIFY ZIP
# ============================================================

def verify_zip():
    print_header("5. VERIFYING FINAL ZIP")

    if not os.path.isfile(ZIP_PATH):
        return False, "ZIP file was not created."

    try:
        with zipfile.ZipFile(
            ZIP_PATH,
            "r"
        ) as archive:

            bad_file = archive.testzip()

            if bad_file is not None:
                return False, (
                    f"Corrupt ZIP entry: {bad_file}"
                )

            names = set(
                archive.namelist()
            )

    except Exception as error:
        return False, str(error)

    required_zip_entries = [
        f"{PACKAGE_NAME}/app.py",
        f"{PACKAGE_NAME}/spatial_intelligence.py",
        f"{PACKAGE_NAME}/room_intelligence.py",
        f"{PACKAGE_NAME}/final_report.py",
        f"{PACKAGE_NAME}/final_validation.py",
        f"{PACKAGE_NAME}/README.md",
        f"{PACKAGE_NAME}/requirements.txt",
        f"{PACKAGE_NAME}/yolov8n.pt",
        f"{PACKAGE_NAME}/templates/index.html",
        f"{PACKAGE_NAME}/templates/results.html",
        f"{PACKAGE_NAME}/templates/video_results.html",
        f"{PACKAGE_NAME}/SUBMISSION_INFO.txt",
    ]

    missing = [
        entry
        for entry in required_zip_entries
        if entry not in names
    ]

    if missing:
        return False, (
            "ZIP is missing required files: "
            + ", ".join(missing)
        )

    print("[PASS] ZIP integrity")
    print("[PASS] Required source files")
    print("[PASS] Required templates")
    print("[PASS] README and requirements")
    print("[PASS] YOLO model")
    print("[PASS] Submission information")

    return True, "Final ZIP verified successfully."


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create the final submission package for "
            "Semantic 3D Reconstruction for Indoor Digital Twins."
        )
    )

    parser.add_argument(
        "--without-outputs",
        action="store_true",
        help=(
            "Create a smaller submission ZIP without copying "
            "generated output files. Output folders are still created."
        )
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Skip automatic final_validation.py execution. "
            "Use only if validation has already been run."
        )
    )

    args = parser.parse_args()

    print_header(
        "STEP 39 - FINAL SUBMISSION PACKAGING"
    )

    print("Project:", PROJECT_NAME)
    print("Python :", sys.version.split()[0])
    print("Root   :", BASE_DIR)

    if not args.skip_validation:
        valid, validation_message = run_validation()

        print(validation_message)

        if not valid:
            print()
            print(
                "PACKAGE STATUS: STOPPED"
            )
            print(
                "Fix validation failures before creating "
                "the final submission package."
            )
            sys.exit(1)

    missing = check_required_files()

    if missing:
        print()
        print(
            "PACKAGE STATUS: STOPPED"
        )
        print(
            "Missing required files:"
        )

        for item in missing:
            print(" -", item)

        sys.exit(1)

    build_package(
        include_outputs=not args.without_outputs
    )

    create_zip()

    verified, message = verify_zip()

    print()
    print(message)

    if not verified:
        print()
        print(
            "PACKAGE STATUS: FAILED"
        )
        sys.exit(1)

    print()
    print("=" * 68)
    print("FINAL PROJECT STATUS: SUBMISSION PACKAGE READY")
    print("=" * 68)
    print()
    print("Final ZIP:")
    print(ZIP_PATH)
    print()
    print(
        "The Semantic 3D Reconstruction for Indoor Digital Twins "
        "project has completed Step 39."
    )
    print()


if __name__ == "__main__":
    main()
