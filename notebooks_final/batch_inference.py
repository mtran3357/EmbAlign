import os
import glob
import joblib
import argparse
import pandas as pd
from datetime import datetime

from aligner.config import PipelineConfig
from aligner.matcher import SinkhornMatcher, HungarianMatcher
from aligner.transformer import RigidTransformer
from aligner.engine import ModularAlignmentEngine
from aligner.models import EmbryoFrame
from aligner.oracle import DiagnosticLayer
from aligner.runner import InferenceRunner
from aligner.report_builder import HTMLReportBuilder

def parse_args():
    parser = argparse.ArgumentParser(description="EmbAlign Batch Report Generator")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing nuclei CSVs")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory for alignment results")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing .pkl model files")
    parser.add_argument("--scale_xy", type=float, default=0.1048, help="XY scaling factor (default: 0.1048)")
    parser.add_argument("--scale_z", type=float, default=0.75, help="Z scaling factor (default: 0.75)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Frozen Models
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading models from {args.model_dir}...")
    master_atlas = joblib.load(os.path.join(args.model_dir, 'master_gp_atlas.pkl'))
    master_slice_db = joblib.load(os.path.join(args.model_dir, 'master_slice_db.pkl'))
    life_history_df = joblib.load(os.path.join(args.model_dir, 'master_life_history.pkl'))
    oracle = DiagnosticLayer(model_path=os.path.join(args.model_dir, 'production_oracle.pkl'))
    growth_curve = pd.read_csv(os.path.join(args.model_dir, 'empirical_growth_curve.csv'))

    # 2. Initialize Engine & Runner
    config = PipelineConfig.v3_0_production()
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
    
    runner = InferenceRunner(engine=engine, oracle=oracle, life_history_df=life_history_df)
    report_builder = HTMLReportBuilder(growth_df=growth_curve)

    # 3. Process Data Files
    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    summary_list = []

    for file_path in csv_files:
        # Extract embryo ID from filename
        embryo_id = os.path.basename(file_path).replace('.csv', '')
        sample_out_dir = os.path.join(args.output_dir, embryo_id)
        os.makedirs(sample_out_dir, exist_ok=True)

        print(f"Running alignment for: {embryo_id}...")
        
        try:
            # Instantiate Frame
            frame = EmbryoFrame.from_inference_csv(
                filepath=file_path,
                embryo_id=embryo_id,
                time_idx=0, 
                scale_xy=args.scale_xy, 
                scale_z=args.scale_z
            )
            
            # Run Inference
            reports = runner.run_for_report([frame])
            report = reports[0]
            best_res = report['best_result']
            
            # Export Annotated Cell CSV
            diag_df = best_res['diagnostics']
            csv_out = os.path.join(sample_out_dir, f"{embryo_id}_annotated_cells.csv")
            diag_df.to_csv(csv_out, index=False)
            
            # Export HTML Report
            html_out = os.path.join(sample_out_dir, f"{embryo_id}_alignment_report.html")
            report_builder.build_report(report, output_path=html_out)
            
            # Record summary data
            summary_list.append({
                "embryo_id": embryo_id,
                "cell_count": report['num_cells'],
                "confidence_score": best_res.get('mean_confidence', 0.0),
                "folder_path": os.path.abspath(sample_out_dir)
            })

        except Exception as e:
            print(f"Error processing {embryo_id}: {e}")

    # 4. Generate Master Summary Report
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_df = summary_df.sort_values(by="confidence_score", ascending=False)
        summary_path = os.path.join(args.output_dir, "batch_summary_report.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nBatch processing complete. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
    
# python /Users/miles/Documents/UCLA/shah_lab/embryo_aligner/notebooks_final/batch_inference.py \
#     --input_dir /Users/miles/Documents/UCLA/shah_lab/embryo_aligner/notebooks/refactor_testing/inference_data_v2 \
#     --output_dir /Users/miles/Documents/UCLA/shah_lab/embryo_aligner/notebooks/refactor_testing/test_inference \
#     --model_dir /Users/miles/Documents/UCLA/shah_lab/embryo_aligner/notebooks_final/production_models \
#     --scale_xy 0.1048 \
#     --scale_z 0.75