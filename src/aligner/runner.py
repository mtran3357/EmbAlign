import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional
from .models import EmbryoFrame

class BatchReporter:
    """Aggregates cell and frame metrics for final CSV output."""
    def __init__(self):
        self.cell_records = []
        self.frame_records = []
        
    def add_result(self, frame: EmbryoFrame, result: dict):
        """Parses engine output and frame metadata into flat records."""
        
        # Metadata extraction
        eid = frame.embryo_id
        tid = frame.time_idx
        n_valid = len(frame)
        meta = frame.valid_df.iloc[0]
        
        # Results extraction
        inferred_labels = result.get('labels', [])
        aligned_coords = result.get('coords', np.full((n_valid, 3), np.nan))
        total_cost= result.get('cost', np.nan)
        
        # Frame level results
        true_labels = frame.valid_df['cell_name'].astype(str).tolist()
        correct = [i == t for i, t in zip(inferred_labels, true_labels)]
        accuracy = np.mean(correct) if n_valid > 0 else np.nan
        
        self.frame_records.append({
            "embryo_id": eid,
            "time_idx": tid,
            "canonical_time": meta.get('canonical_time', np.nan),
            "source_file": meta.get('source_file', "unknown"),
            "N_valid": n_valid,
            "frame_accuracy": accuracy,
            "total_mahalanobis_cost": total_cost,
            "mean_mahalanobis_sq": total_cost / n_valid if n_valid > 0 else np.nan,
            "best_slice_id": result.get('slice_id'),
            "scale_factor": result.get('scale_factor', 1.0)
        })
        
        # Cell-level records
        for i in range(n_valid):
            self.cell_records.append({
                "embryo_id": eid,
                "time_idx": tid,
                "cell_name": true_labels[i],
                "inferred_label": inferred_labels[i],
                "x_atlas_infer": aligned_coords[i, 0],
                "y_atlas_infer": aligned_coords[i, 1],
                "z_atlas_infer": aligned_coords[i, 2],
                "is_correct": correct[i]
            })
    
    def save(self, cell_out: str, frame_out:str):
        """Exports results to CSV."""
        pd.DataFrame(self.cell_records).to_csv(cell_out, index=False)
        pd.DataFrame(self.frame_records).to_csv(frame_out, index=False)
        
class BatchRunner:
    """Orchestrates the alignment engine over the full batched dataset."""
    def __init__(self, engine, reporter: BatchReporter):
        self.engine = engine
        self.reporter = reporter 
    
    def run(self, df: pd.DataFrame, max_N: Optional[int] = None):
        """Groups the dataframe and processes each frame sequentially."""
        df = df.sort_values(["embryo_id", "time_idx"])
        grouped = df.groupby(["embryo_id", "time_idx"], sort=False)
        
        print(f"Starting batch run for {len(grouped)} frames . . .")
        
        for (eid, tid), frame_df in tqdm(grouped, desc="Aligning Frames"):
            # Check cell count
            n_valid = len(frame_df[frame_df['valid'] == 1])
            
            if max_N and n_valid > max_N:
                continue
            
            try:
                # Initialize
                frame = EmbryoFrame.from_dataframe(df, eid, tid)
                # Align
                result = self.engine.align_frame(frame)
                # Log 
                if result:
                    self.reporter.add_result(frame, result)
            
            except Exception as e:
                print(f"Skipping Embryo {eid} T={tid} due to error: {e}")
                
        