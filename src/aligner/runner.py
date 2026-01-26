import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from typing import Optional
from .models import EmbryoFrame

class BatchReporter:
    """Aggregates cell and frame metrics for final CSV output."""
    def __init__(self):
        self.cell_records = []
        self.frame_records = []
        self.trace_records = []
        
    def add_result(self, frame: EmbryoFrame, result: dict, traces: dict = None):
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
        
        # Optional traces
        if traces:
            for s_id, data in traces.items():
                for step in data['coarse']:
                    self.trace_records.append({
                        "embryo_id": eid,
                        "time_idx": tid,
                        "slice_id": s_id,
                        "phase": "coarse",
                        "sign": step['sign'],
                        "angle_deg": step['angle_deg'],
                        "cost": step['cost'],
                        "is_winner": s_id == result.get('slice_id')
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
    
    def run(self, df: pd.DataFrame, max_N: Optional[int] = None, trace: bool = False):
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
                if trace:
                    result, landscape = self.engine.align_frame(frame, trace=True)
                else:
                    result = self.engine.align_frame(frame, trace=False)
                    landscape = None
                # Log 
                if result:
                    self.reporter.add_result(frame, result, traces = landscape)
            
            except Exception as e:
                print(f"Skipping Embryo {eid} T={tid} due to error: {e}")
                
class InferenceRunner:
    def __init__(self, engine, px_xy=0.0183, px_z=0.75):
        """Initializes the runner with a test embryo."""
        self.engine = engine
        self.px_xy = px_xy
        self.px_z = px_z
    
    def run_single(self, csv_path: str, embryo_id: str = None):
        """Run full inference pipeline on a single test embroy CSV."""
        if embryo_id is None:
            embryo_id = Path(csv_path.stem)
        # Raw matrix load
        raw_df = pd.read_csv(csv_path)
        matrix = raw_df[['X', 'Y', 'Z']]
        # Initialize Frame
        frame = EmbryoFrame.from_matrix(
            matrix = matrix,
            embryo_id=embryo_id,
            px_xy=self.px_xy,
            px_z=self.px_z
        )
        frame.prepare()
        
        result, *_ = self.engine.align_frame(frame)
        
        return result
    
    def batch_run(self, directory_path: str):
        """Automatically finds all CSVs in a folder and runs inference."""
        all_results = {}
        csv_files = Path(directory_path).glob("*.csv")
        
        for file in csv_files:
            print(f"Processing inference for: {file.name}")
            all_results[file.stem] = self.run_single(str(file))
            
        return all_results
    
class InferenceReporter:
    """Mirror of BatchReporter for unlabeled test data."""
    def __init__(self, atlas):
        self.atlas = atlas
        self.cell_records = []
        self.frame_records = []
        self.trace_records =[]
        
    def add_result(self, frame: 'EmbryoFrame', result: dict, traces: dict = None):
        """Parses pipeline output for unlabeled inference data."""
        eid = frame.embryo_id
        tid = frame.time_idx
        n_cells = len(frame)
        
        # 1. Extraction
        inferred_labels = result.get('labels', [])
        aligned_coords = result.get('coords', np.full((n_cells, 3), np.nan))
        total_cost = result.get('cost', np.nan)
        best_slice = result.get('slice_id', 'unknown')
        
        # 2. Get Atlas Goalposts for inferred labels
        # This provides the X_atlas, Y_atlas, Z_atlas benchmarks
        atlas_means, _, _ = self.atlas.get_params(inferred_labels)
        
        # 3. Frame-level records
        self.frame_records.append({
            "embryo_id": eid,
            "time_idx": tid,
            "N_cells": n_cells,
            "total_cost_slice": total_cost,
            "mean_mahalanobis_sq": total_cost / n_cells if n_cells > 0 else np.nan,
            "best_slice_key": best_slice,
            "scale_factor": frame.scale_factor # Median dist used for normalization
        })
        
        # 4. Cell-level records (Flattened for the requested CSV format)
        for i in range(n_cells):
            self.cell_records.append({
                "embryo_id": eid,
                "time_idx": tid,
                # Raw/Physical Coordinates
                "X_phys": frame.coords[i, 0],
                "Y_phys": frame.coords[i, 1],
                "Z_phys": frame.coords[i, 2],
                # Normalized (Centered & Scaled)
                "X_norm": frame.normalized_coords[i, 0],
                "Y_norm": frame.normalized_coords[i, 1],
                "Z_norm": frame.normalized_coords[i, 2],
                # Atlas-Aligned (The result of the engine's ICP)
                "X_atlas_infer": aligned_coords[i, 0],
                "y_atlas_infer": aligned_coords[i, 1],
                "z_atlas_infer": aligned_coords[i, 2],
                # Reference Atlas Means
                "X_atlas_ref": atlas_means[i, 0],
                "Y_atlas_ref": atlas_means[i, 1],
                "Z_atlas_ref": atlas_means[i, 2],
                # Identification
                "assigned_label": inferred_labels[i],
                "mahalanobis_sq": result.get('per_cell_costs', [np.nan]*n_cells)[i]
            })

        # 5. Optional traces (Mirroring BatchReporter style)
        if traces:
            for s_id, data in traces.items():
                for step in data.get('coarse', []):
                    self.trace_records.append({
                        "embryo_id": eid,
                        "time_idx": tid,
                        "slice_id": s_id,
                        "phase": "coarse",
                        "angle_deg": step.get('angle_deg'),
                        "cost": step.get('cost'),
                        "is_winner": s_id == best_slice
                    })
                
    def save(self, cell_out: str, frame_out: str, trace_out: str = None):
        """Exports results to CSV."""
        pd.DataFrame(self.cell_records).to_csv(cell_out, index=False)
        pd.DataFrame(self.frame_records).to_csv(frame_out, index=False)
        if trace_out and self.trace_records:
            pd.DataFrame(self.trace_records).to_csv(trace_out, index=False)