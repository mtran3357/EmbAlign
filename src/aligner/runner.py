import pandas as pd
import numpy as np
from tqdm import tqdm

class PipelineEvaluator:
    @staticmethod
    def evaluate_benchmark(frame_df, cell_df, full_ground_truth_df):
        eval_records = []
        
        # 1. Type Standardization
        for df in [frame_df, cell_df, full_ground_truth_df]:
            if not df.empty:
                if 'embryo_id' in df.columns:
                    df.loc[:, 'embryo_id'] = df['embryo_id'].astype(str)
                if 'time_idx' in df.columns:
                    df.loc[:, 'time_idx'] = df['time_idx'].astype(int)

        # 2. Build Ground Truth Lookup
        gt_lookup = full_ground_truth_df.groupby(['embryo_id', 'time_idx'])['cell_name'].apply(
            lambda x: set(str(c).strip() for c in x if pd.notna(c))
        ).to_dict()

        for _, row in frame_df.iterrows():
            eid, t_idx = str(row['embryo_id']), int(row['time_idx'])
            set_true = gt_lookup.get((eid, t_idx), set())
            
            frame_id = row.get('frame_id', f"{eid}_t{t_idx}")
            
            # 3. Extract Predicted Labels dynamically from cell_df 
            set_pred = set()
            positional_accuracy = 0.0
            
            if not cell_df.empty and 'frame_id' in cell_df.columns:
                frame_cells = cell_df[
                    (cell_df['frame_id'] == frame_id) & 
                    (cell_df['config_name'] == row['config_name'])
                ]
                if not frame_cells.empty:
                    set_pred = set(
                        str(c).strip() for c in frame_cells['cell_name'] 
                        if pd.notna(c) and str(c).lower() != 'unassigned'
                    )
                    if 'is_correct' in frame_cells.columns:
                        positional_accuracy = frame_cells['is_correct'].mean()

            # 4. Calculate Overlap Metrics (Set Terminology)
            intersection = set_true.intersection(set_pred)
            set_accuracy = len(intersection) / len(set_true) if len(set_true) > 0 else 0.0
            set_match = 1.0 if set_accuracy == 1.0 and len(set_true) == len(set_pred) else 0.0
            
            eval_record = row.to_dict()
            eval_record.update({
                'set_match': set_match,
                'set_accuracy': set_accuracy,
                'positional_accuracy': positional_accuracy,
                'num_gt_cells': len(set_true),
                'num_pred_cells': len(set_pred)
            })
            eval_records.append(eval_record)
            
        return pd.DataFrame(eval_records)
    

class InferenceRunner:
    """
    Strictly for production use. Runs the alignment engine on NEW, unannotated 
    embryo data and saves the predicted coordinates and labels. 
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
                    
                # 2. Predict Confidence
                if self.oracle:
                    res = self.oracle.predict_confidence(res)
                    
                # 3. Format Output
                aligned_coords = res['coords']
                labels = res['labels']
                
                # Fetch raw data from the frame's metadata (if it exists)
                raw_df = getattr(frame, 'valid_df', None)
                
                for i in range(len(labels)):
                    record = {
                        'embryo_id': getattr(frame, 'embryo_id', 'unknown'),
                        'time_idx': getattr(frame, 'time_idx', -1),
                        'predicted_label': labels[i],
                        # Original Unscaled Coordinates (for biological mapping)
                        'raw_x': raw_df['X'].iloc[i] if raw_df is not None else np.nan,
                        'raw_y': raw_df['Y'].iloc[i] if raw_df is not None else np.nan,
                        'raw_z': raw_df['Z'].iloc[i] if raw_df is not None else np.nan,
                        'raw_d': raw_df['D'].iloc[i] if raw_df is not None else np.nan,
                        # Aligned Coordinates (for algorithm debugging/3D plotting)
                        'x_aligned': aligned_coords[i, 0],
                        'y_aligned': aligned_coords[i, 1],
                        'z_aligned': aligned_coords[i, 2],
                        'confidence': res['diagnostics']['confidence_score'].iloc[i] if self.oracle else np.nan
                    }
                    results.append(record)
                    
            except Exception as e:
                print(f"Failed to process frame {getattr(frame, 'time_idx', 'unknown')}: {e}")
                
        # Export to CSV
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")