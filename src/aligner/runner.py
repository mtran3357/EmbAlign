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
        # We strip whitespace to ensure 'ABpl ' matches 'ABpl'
        gt_lookup = full_ground_truth_df.groupby(['embryo_id', 'time_idx'])['cell_name'].apply(
            lambda x: set(str(c).strip() for c in x if pd.notna(c))
        ).to_dict()

        for _, row in frame_df.iterrows():
            eid, t_idx = str(row['embryo_id']), int(row['time_idx'])
            set_true = gt_lookup.get((eid, t_idx), set())
            
            # 3. Extract and Clean Predicted Labels
            raw_pred = []
            if 'labels' in row and isinstance(row['labels'], (list, tuple, np.ndarray)):
                raw_pred = row['labels']
            elif 'cell_names' in row: 
                raw_pred = str(row['cell_names']).split(';')
            
            # Clean: Remove "unassigned", strip spaces, remove NaNs
            set_pred = set(
                str(c).strip() for c in raw_pred 
                if pd.notna(c) and str(c).lower() != 'unassigned'
            )

            # 4. Calculate Overlap Metrics
            intersection = set_true.intersection(set_pred)
            
            # This is the "Proportion of total observed cells" you requested
            # If the engine found 19/20 correct labels, this will be 0.95
            slice_accuracy = len(intersection) / len(set_true) if len(set_true) > 0 else 0.0
            
            # Slice Match is TRUE (1.0) only if every single label matches perfectly
            slice_match = 1.0 if slice_accuracy == 1.0 and len(set_true) == len(set_pred) else 0.0

            # 5. Positional Accuracy (Cell-to-Cell)
            positional_accuracy = 0.0
            if not cell_df.empty and 'embryo_id' in cell_df.columns:
                frame_cells = cell_df[
                    (cell_df['embryo_id'] == eid) & 
                    (cell_df['time_idx'] == t_idx) & 
                    (cell_df['config_name'] == row['config_name'])
                ]
                if not frame_cells.empty and 'is_correct' in frame_cells.columns:
                    # 'is_correct' should be pre-calculated in engine.py
                    positional_accuracy = frame_cells['is_correct'].mean()
            
            eval_record = row.to_dict()
            eval_record.update({
                'slice_match': slice_match,
                'slice_accuracy': slice_accuracy,
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