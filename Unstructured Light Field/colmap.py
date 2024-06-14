import os
import subprocess

def run_colmap(image_dir, workspace_dir):
    # Step 1: Feature extraction
    feature_extraction_command = [
        "colmap", "feature_extractor",
        "--database_path", os.path.join(workspace_dir, "database.db"),
        "--image_path", image_dir
    ]
    subprocess.run(feature_extraction_command, check=True)

    # Step 2: Exhaustive matching
    exhaustive_matching_command = [
        "colmap", "exhaustive_matcher",
        "--database_path", os.path.join(workspace_dir, "database.db")
    ]
    subprocess.run(exhaustive_matching_command, check=True)

    # Step 3: Sparse reconstruction
    sparse_reconstruction_command = [
        "colmap", "mapper",
        "--database_path", os.path.join(workspace_dir, "database.db"),
        "--image_path", image_dir,
        "--output_path", os.path.join(workspace_dir, "sparse")
    ]
    subprocess.run(sparse_reconstruction_command, check=True)

    # Step 4: Dense reconstruction
    dense_reconstruction_command = [
        "colmap", "image_undistorter",
        "--image_path", image_dir,
        "--input_path", os.path.join(workspace_dir, "sparse", "0"),
        "--output_path", os.path.join(workspace_dir, "dense"),
        "--output_type", "COLMAP"
    ]
    subprocess.run(dense_reconstruction_command, check=True)

    dense_stereo_command = [
        "colmap", "patch_match_stereo",
        "--workspace_path", os.path.join(workspace_dir, "dense"),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ]
    subprocess.run(dense_stereo_command, check=True)

    # Step 5: Convert depth maps to images
    stereo_fusion_command = [
        "colmap", "stereo_fusion",
        "--workspace_path", os.path.join(workspace_dir, "dense"),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", os.path.join(workspace_dir, "dense", "fused.ply")
    ]
    subprocess.run(stereo_fusion_command, check=True)

# Example usage
image_dir = "Images_New"  # Directory containing your images
workspace_dir = "WorkSpace"  # Directory to store intermediate and final results
os.makedirs(workspace_dir, exist_ok=True)
run_colmap(image_dir, workspace_dir)
