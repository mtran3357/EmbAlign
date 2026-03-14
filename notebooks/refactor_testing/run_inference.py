import os
import glob
import joblib
import pandas as pd

from aligner.config import PipelineConfig
from aligner.matcher import SinkhornMatcher, HungarianMatcher
from aligner.transformer import RigidTransformer
from aligner.engine import ModularAlignmentEngine
from aligner.models import EmbryoFrame
from aligner.oracle import DiagnosticLayer
from aligner.runner import InferenceRunner

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MODEL_DIR = "production_models"
INPUT_DIR = "inference_data_v2"    # Folder containing your raw X,Y,Z,D CSVs
OUTPUT_FILE = "final_inference_predictions.csv"

# Microscope Scaling Factors (from your earlier context)
SCALE_XY = 0.1083
SCALE_Z = 0.75

def main():
    print("Initializing Inference Pipeline...")

    # ==========================================
    # 2. LOAD FROZEN MODELS
    # ==========================================
    print(f"Loading Master Atlases from {MODEL_DIR}/...")
    master_atlas = joblib.load(os.path.join(MODEL_DIR, 'master_gp_atlas.pkl'))
    master_slice_db = joblib.load(os.path.join(MODEL_DIR, 'master_slice_db.pkl'))
    
    # Initialize the Oracle. 
    # By passing model_path, it loads the Random Forest instead of training a new one.
    oracle = DiagnosticLayer(model_path=os.path.join(MODEL_DIR, 'production_oracle_v1.pkl'))

    # ==========================================
    # 3. SPIN UP THE ENGINE
    # ==========================================
    # We use the exact same config that generated the Master Atlases
    config = PipelineConfig.v2_1_augmented()
    
    engine = ModularAlignmentEngine(
        config=config,
        atlas=master_atlas,
        slice_db=master_slice_db,
        coarse_matcher=HungarianMatcher(tau=config.tau),
        icp_matcher=SinkhornMatcher(
            epsilon=config.epsilon_refine, 
            stop_thr=config.sinkhorn_stop_thr,
            max_iters=config.sinkhorn_max_iters
        ),
        transformer=RigidTransformer()
    )

    # ==========================================
    # 4. DATA INGESTION
    # ==========================================
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}/. Exiting.")
        return

    print(f"Found {len(csv_files)} files to process.")
    
    unannotated_frames = []
    for idx, file_path in enumerate(csv_files):
        # Extract a clean embryo ID from the filename (e.g., 'Pos20_emb1')
        filename = os.path.basename(file_path)
        if '_emb' in filename:
            # Parses "Pos20_ShortWavelengthCamera_reg1_emb1.csv" -> "Pos20_emb1"
            base_pos = filename.split('_')[0]
            emb_num = filename.split('_emb')[1].replace('.csv', '')
            embryo_id = f"{base_pos}_emb{emb_num}"
        else:
            embryo_id = f"unknown_emb_{idx}"
        
        # Build the EmbryoFrame using the custom method we wrote
        try:
            frame = EmbryoFrame.from_inference_csv(
                filepath=file_path,
                embryo_id=embryo_id,
                time_idx=idx, # Acts as a sequential index if processing a time-lapse
                scale_xy=SCALE_XY, 
                scale_z=SCALE_Z
            )
            unannotated_frames.append(frame)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    # ==========================================
    # 5. RUN INFERENCE
    # ==========================================
    runner = InferenceRunner(engine=engine, oracle=oracle)
    runner.annotate_dataset(unannotated_frames, output_csv=OUTPUT_FILE)

if __name__ == "__main__":
    main()