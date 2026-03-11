import pandas as pd
import numpy as np
from tqdm import tqdm

class PipelineEvaluator:
    """
    Strictly handles the calculation of performance metrics by comparing 
    engine predictions against ground truth labels.
    """
    
    @staticmethod
    def evaluate_benchmark(frame_df: pd.DataFrame, cell_df: pd.DataFrame, full_ground_truth_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the raw output from BenchmarkingSuite.run_sweep() and calculates 
        strict frame-level and slice-level accuracy metrics.
        """
        eval_records = []
        
        # Iterate through every frame we attempted to align
        for _, row in tqdm(frame_df.iterrows(), total=len(frame_df), desc="Calculating Metrics"):
            eid = row['embryo_id']
            t_idx = row['time_idx']
            pred_labels = row.get('labels', [])
            
            # 1. Fetch Ground Truth
            gt_subset = full_ground_truth_df[
                (full_ground_truth_df['embryo_id'] == eid) & 
                (full_ground_truth_df['time_idx'] == t_idx) &
                (full_ground_truth_df['valid'] == 1)
            ]
            
            if gt_subset.empty:
                continue
                
            true_labels = gt_subset['cell_name'].str.strip().tolist()
            
            # 2. Filter out "unassigned" from predictions for slice math
            clean_preds = [lbl for lbl in pred_labels if lbl != "unassigned"]
            
            set_true = set(true_labels)
            set_pred = set(clean_preds)
            intersection = set_true.intersection(set_pred)
            
            # 3. Calculate New Slice Metrics
            slice_match = (set_pred == set_true)
            slice_accuracy = len(intersection) / len(set_true) if len(set_true) > 0 else 0.0
            
            # 4. Calculate Positional Accuracy (Did we put the right label on the right physical cell?)
            # We look at the cell_df for this specific frame
            frame_cells = cell_df[
                (cell_df['embryo_id'] == eid) & 
                (cell_df['time_idx'] == t_idx) &
                (cell_df['config_name'] == row['config_name'])
            ]
            
            correct_assignments = frame_cells['is_correct'].sum() if not frame_cells.empty else 0
            positional_accuracy = correct_assignments / len(set_true) if len(set_true) > 0 else 0.0
            
            # Append to final evaluation
            eval_record = row.copy()
            eval_record['slice_match'] = slice_match
            eval_record['slice_accuracy'] = slice_accuracy
            eval_record['positional_accuracy'] = positional_accuracy
            eval_record['n_true_cells'] = len(set_true)
            eval_records.append(eval_record)
            
        return pd.DataFrame(eval_records)


class InferenceRunner:
    """
    Strictly for production use. Runs the alignment engine on NEW, unannotated 
    embryo data and saves the predicted coordinates and labels. 
    Does zero evaluation/ground-truth checking.
    """
    def __init__(self, engine, oracle=None):
        self.engine = engine
        self.oracle = oracle
        
    def annotate_dataset(self, unannotated_frames: list, output_csv: str):
        """Processes a list of EmbryoFrames and exports the predictions."""
        results = []
        
        for frame in tqdm(unannotated_frames, desc="Annotating Embryos"):
            try:
                # 1. Align
                res = self.engine.align_frame(frame, return_diagnostics=True)
                if res is None:
                    continue
                    
                # 2. Predict Confidence (if oracle provided)
                if self.oracle:
                    res = self.oracle.predict_confidence(res)
                    
                # 3. Format Output
                coords = res['coords']
                labels = res['labels']
                
                for i in range(len(labels)):
                    results.append({
                        'embryo_id': getattr(frame, 'embryo_id', 'unknown'),
                        'time_idx': getattr(frame, 'time_idx', -1),
                        'predicted_label': labels[i],
                        'x_aligned': coords[i, 0],
                        'y_aligned': coords[i, 1],
                        'z_aligned': coords[i, 2],
                        'confidence': res['diagnostics']['confidence_score'].iloc[i] if self.oracle else np.nan
                    })
            except Exception as e:
                print(f"Failed to process frame {getattr(frame, 'time_idx', 'unknown')}: {e}")
                
        # Export to CSV for downstream biological analysis
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")