import pandas as pd
import numpy as np
# import os
# import json
import time
from tqdm import tqdm
from typing import Optional
from aligner.models import EmbryoFrame
from aligner.atlas import SliceAtlas
from sklearn.metrics import f1_score

class BatchReporter:
    """Aggregates cell and frame metrics for final CSV output."""
    def __init__(self, full_df: pd.DataFrame = None):
        self.full_df = full_df
        self.cell_records = []
        self.frame_records = []
        self.diag_records = []
        self.trace_records = []
        self.skipped_records = []
        
    def log_skip(self, frame, reason: str):
        """NEW: Explicitly records frames the engine could not process."""
        self.skipped_records.append({
            "embryo_id": frame.embryo_id,
            "time_idx": frame.time_idx,
            "n_cells": len(frame),
            "reason": reason
        })
        
    def get_diagnostic_report(self):
        """Returns the concatenated diagnostic dataframe only."""
        if not self.diag_records:
            return pd.DataFrame()
        return pd.concat(self.diag_records, ignore_index=True)
    
    def add_result(self, frame, result, elapsed_time, atlas: "SliceAtlas", traces=None):
        """Parses engine output and frame metadata into flat records."""
        if isinstance(result, dict) and 'diagnostics' in result:
            diag_df = result['diagnostics'].copy()
            diag_df['embryo_id'] = frame.embryo_id
            diag_df['time_idx'] = frame.time_idx
            self.diag_records.append(diag_df)
        # Metadata extraction
        eid = frame.embryo_id
        tid = frame.time_idx
        n_valid = len(frame)
        meta = frame.valid_df.iloc[0]
        
        # Results extraction
        inferred_labels = result.get('labels', [])
        aligned_coords = result.get('coords', np.full((n_valid, 3), np.nan))
        #per_cell_mah = result.get('per_cell_mah', np.full(n_valid, np.nan))
        total_cost= result.get('cost', np.nan)
        
        # Find ground truth slice
        true_slice = self.get_true_slice_id(frame, atlas)
        picked_slice = result.get('slice_id')
        
        # Count how many cells the matcher 'rejected' into the slack bin
        unassigned_mask = [label == "unassigned" for label in inferred_labels]
        n_unassigned = sum(unassigned_mask)
        
        # Frame level results
        true_labels = frame.valid_df['cell_name'].astype(str).tolist()
        for i in range(n_valid):
            gt = true_labels[i]
            pred = inferred_labels[i]
            
            # Biological Goodness: Sulston Prefix Ratio
            lcp_len = self._get_sulston_lcp(gt, pred)
            lineage_score = lcp_len / max(len(gt), len(pred)) if gt != "unassigned" else 0
            
            self.cell_records.append({
                "embryo_id": eid,
                "time_idx": tid,
                "cell_name": gt,
                "inferred_label": pred,
                "is_correct": gt == pred,
                "sulston_lcp": lcp_len,
                "lineage_score": lineage_score,
                "x_atlas_infer": aligned_coords[i, 0],
                "y_atlas_infer": aligned_coords[i, 1],
                "z_atlas_infer": aligned_coords[i, 2]
            })
            
        self.frame_records.append({
            "embryo_id": eid,
            "time_idx": tid,
            "runtime_sec": elapsed_time, # Timing metric
            "N_valid": n_valid,
            "n_unassigned": n_unassigned,
            "completeness": (n_valid - n_unassigned) / n_valid if n_valid > 0 else 0,
            "frame_accuracy": np.mean([gt == pred for gt, pred in zip(true_labels, inferred_labels)]),
            "total_mahalanobis_cost": total_cost,
            "mean_mahalanobis_sq": total_cost / n_valid if n_valid > 0 else np.nan,
            "slice_match": true_slice == picked_slice if true_slice is not None else np.nan
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
                
    def summarize_embryo_performance(self):
        """Generates a diagnostic summary of alignment run."""
        if not self.cell_records or not self.frame_records:
            return pd.DataFrame()
        
        if isinstance(self.cell_records[0], pd.DataFrame):
            return pd.concat(self.cell_records, ignore_index=True)
        
        cells = pd.DataFrame(self.cell_records)
        frames = pd.DataFrame(self.frame_records)
        
        frames['is_perfect'] = frames['frame_accuracy'] == 1.0
        # Aggregate frame-level metrics
        frame_summary = frames.groupby('embryo_id').agg(
            n_frames = ('time_idx', 'nunique'),
            avg_completeness = ('completeness', 'mean'), 
            avg_accuracy = ('frame_accuracy', 'mean'),
            total_unassigned = ('n_unassigned', 'sum'),
            max_N=('N_valid', 'max'),
            avg_mahalanobis_sq=('mean_mahalanobis_sq', 'mean'),
            mahalanobis_sq_sd =('mean_mahalanobis_sq', 'std'),
            total_alignment_cost=('total_mahalanobis_cost', 'sum'),
            prop_perfect_frames=('is_perfect', 'mean')
        )

        # Calculate F1 metrics
        def get_macro_f1(group):
            try:
                return f1_score(group['cell_name'], group['inferred_label'], average='macro')
            except:
                return np.nan
        
        cell_summary = cells.groupby('embryo_id').agg(
            total_cell_obs=('is_correct', 'count'),
            cell_accuracy=('is_correct', 'mean')
        )
        cell_summary['f1_macro'] = cells.groupby('embryo_id').apply(get_macro_f1)
        
        summary = frame_summary.join(cell_summary).reset_index()
        
        return summary.sort_values('cell_accuracy', ascending=False)
    
    def summarize_embryo_performance(self):
        """
        Generates two distinct reports: 
        1. Benchmark summary (embryo-level aggregates)
        2. Diagnostic summary (cell-level feature DataFrame)
        """
        if not self.cell_records:
            return pd.DataFrame(), pd.DataFrame()

        # 1. Split mixed records by type
        cell_dicts = [r for r in self.cell_records if isinstance(r, dict)]
        cell_dfs = [r for r in self.cell_records if isinstance(r, pd.DataFrame)]

        # 2. Process Benchmark Summary (if standard dict records exist)
        if cell_dicts and self.frame_records:
            cells = pd.DataFrame(cell_dicts)
            frames = pd.DataFrame(self.frame_records)
            
            frames['is_perfect'] = frames['frame_accuracy'] == 1.0
            
            # Aggregate frame-level metrics
            frame_summary = frames.groupby('embryo_id').agg(
                n_frames=('time_idx', 'nunique'),
                avg_completeness=('completeness', 'mean'),
                avg_accuracy=('frame_accuracy', 'mean'),
                total_unassigned=('n_unassigned', 'sum'),
                max_N=('N_valid', 'max'),
                avg_mahalanobis_sq=('mean_mahalanobis_sq', 'mean'),
                mahalanobis_sq_sd=('mean_mahalanobis_sq', 'std'),
                total_alignment_cost=('total_mahalanobis_cost', 'sum'),
                prop_perfect_frames=('is_perfect', 'mean')
            )

            def get_macro_f1(group):
                try: return f1_score(group['cell_name'], group['inferred_label'], average='macro')
                except: return np.nan
            
            cell_summary = cells.groupby('embryo_id').agg(
                total_cell_obs=('is_correct', 'count'),
                cell_accuracy=('is_correct', 'mean')
            )
            cell_summary['f1_macro'] = cells.groupby('embryo_id').apply(get_macro_f1)
            
            benchmark_summary = frame_summary.join(cell_summary).reset_index()
            benchmark_summary = benchmark_summary.sort_values('cell_accuracy', ascending=False)
        else:
            benchmark_summary = pd.DataFrame()

        # 3. Process Diagnostic Summary (concatenate cell-level DataFrames)
        if cell_dfs:
            diagnostic_summary = pd.concat(cell_dfs, ignore_index=True)
        else:
            diagnostic_summary = pd.DataFrame()

        return benchmark_summary, diagnostic_summary
    
    def summarize_frame_diagnostics(self):
        """
        Generates a per-frame diagnostic report strictly for benchmarking standard metrics.
        """
        # 1. Isolate only the dictionaries (Standard Benchmark Records)
        # This prevents diagnostic DataFrames from causing a TypeError in pd.DataFrame()
        standard_records = [r for r in self.cell_records if isinstance(r, dict)]
        
        if not standard_records or not self.frame_records:
            return pd.DataFrame()
        
        df_cells = pd.DataFrame(standard_records)
        df_frames = pd.DataFrame(self.frame_records)
        
        # 2. Aggregate cell metrics to frame level
        frame_cell_stats = df_cells.groupby(['embryo_id', 'time_idx']).agg(
            avg_lineage_score=('lineage_score', 'mean'),
        ).reset_index()
        
        # 3. Merge frames and cell stats
        diagnostics = pd.merge(
            df_frames,
            frame_cell_stats,
            on=['embryo_id', 'time_idx'],
            how='inner'
        )

        # 4. Dataset Coverage Summary
        total_attempted = len(self.frame_records) + len(self.skipped_records)
        if total_attempted > 0:
            coverage = len(self.frame_records) / total_attempted
            print(f"Dataset Coverage: {coverage:.1%} ({len(self.frame_records)} aligned, {len(self.skipped_records)} skipped)")
            
        # 5. Optional metadata merge
        if self.full_df is not None:
            # List of cell-level diagnostics that should not be in a frame summary
            cell_level_cols = [
                'cell_name', 'x_aligned', 'y_aligned', 'z_aligned', 
                'x_um', 'y_um', 'z_um', 'valid', 'source_file'
            ]
            
            # Drop the blacklisted columns and then drop duplicates
            frame_meta = self.full_df.drop(columns=cell_level_cols, errors='ignore')
            frame_meta = frame_meta.drop_duplicates(['embryo_id', 'time_idx'])
            
            diagnostics = pd.merge(
                diagnostics, 
                frame_meta, 
                on=['embryo_id', 'time_idx'], 
                how='left'
            )
            
        return diagnostics.sort_values(by=['embryo_id', 'time_idx'])
    def get_true_slice_id(self, frame, atlas: SliceAtlas):
        """
        Finds the slice_id in the atlas that exactly matches the frame's ground truth labels.
        """
        gt_labels = tuple(sorted(frame.valid_df['cell_name'].astype(str).tolist()))
        n_cells = len(gt_labels)
        # Get candidate slice IDs for this cell count
        candidates = atlas.get_candidates(n_cells)
        # Check each candidate for an exact label match
        for s_id in candidates:
            if atlas.get_labels(s_id) == gt_labels:
                return s_id
                
        return None
            
    
    def _get_sulston_lcp(self, label1, label2):
        """Calculates the length of of the longos common prefix for sulston names."""
        if not isinstance(label1, str) or not isinstance(label2, str):
            return 0
        lcp = 0
        for c1, c2 in zip(label1, label2):
            if c1==c2:
                lcp += 1
            else:
                break
        return lcp
                        
    def save(self, cell_out: str, frame_out:str):
        """Exports results to CSV."""
        pd.DataFrame(self.cell_records).to_csv(cell_out, index=False)
        pd.DataFrame(self.frame_records).to_csv(frame_out, index=False)
        
class BatchRunner:
    """Orchestrates the alignment engine over the full batched dataset."""
    def __init__(self, engine, reporter: BatchReporter):
        self.engine = engine
        self.reporter = reporter 
    
    def run(self, df: pd.DataFrame, life_history_df: pd.DataFrame = None, max_N: Optional[int] = None, trace: bool = False, return_diagnostics=False):
        """Groups the dataframe and processes each frame sequentially."""
        # 1. Pipeline Manifest
        print("="*40)
        print(" PIPELINE RUN MANIFEST")
        print("="*40)
        print(f"Engine:    {self.engine.__class__.__name__}")
        print(f"Matcher:   {self.engine.matcher.__class__.__name__}")
        print(f"Transform: {self.engine.transformer.__class__.__name__}")
        
        # 2. Parameter Inspection
        print("-" * 40)
        print("Active Parameters:")
        # Pull live tau from engine settings
        print(f"  - Tau (Slack):  {self.engine.settings.get('tau', 'Not Set')}")
        
        # Pull internal matcher params
        if hasattr(self.engine.matcher, 'epsilon'):
            print(f"  - Epsilon:      {self.engine.matcher.epsilon}")
        if hasattr(self.engine.matcher, 'max_iters'):
            print(f"  - ICP Max Iters: {self.engine.settings.get('icp_iters')}")
        
        print(f"  - Angle Step:   {self.engine.settings.get('angle_step_deg')}°")
        print("="*40 + "\n")
        df = df.sort_values(["embryo_id", "time_idx"])
        grouped = df.groupby(["embryo_id", "time_idx"], sort=False)
        
        print(f"Starting batch run for {len(grouped)} frames . . .")
        # Normal execution
        for (eid, tid), frame_df in tqdm(grouped, desc="Aligning Frames"):
            # Check cell count
            n_valid = len(frame_df[frame_df['valid'] == 1])
            
            if max_N and n_valid > max_N:
                continue
            
            try:
                # Initialize
                frame = EmbryoFrame.from_dataframe(df, eid, tid)
                # Align
                start_time = time.time()
                if trace:
                    output = self.engine.align_frame(frame, trace=True, return_diagnostics=return_diagnostics, life_history_df = life_history_df)
                    result, landscape = output if output else (None, None)
                else:
                    result = self.engine.align_frame(frame, trace=False, return_diagnostics=return_diagnostics, life_history_df = life_history_df)
                    landscape = None
                elapsed = time.time() - start_time
                # Log 
                if result:
                    self.reporter.add_result(
                        frame, 
                        result, 
                        elapsed, 
                        atlas=self.engine.slice_db, 
                        traces=landscape
                    )
                else:
                    self.reporter.log_skip(frame, reason="Missing Slice Template")
            
            except Exception as e:
                self.reporter.log_skip(frame, reason=f"Runtime Error: {e}")
        # print run summary
        print("\n" + "="*40)
        print(" PIPELINE RESULTS SUMMARY")
        print("="*40)
        
        if self.reporter.frame_records:
            # Create a temporary DF from the reporter's records
            results_df = pd.DataFrame(self.reporter.frame_records)
            
            avg_acc = results_df['frame_accuracy'].mean()
            total_time = results_df['runtime_sec'].sum()
            avg_time = results_df['runtime_sec'].mean()
            n_frames = len(results_df)

            print(f"Frames Processed: {n_frames}")
            print(f"Total Runtime:    {total_time:.2f}s")
            print(f"Avg Time/Frame:   {avg_time:.2f}s")
            print(f"Mean Accuracy:    {avg_acc:.2%}")
        else:
            print("No frames were successfully processed.")
        
        print("="*40 + "\n")
                
class InferenceRunner:
    def __init__(self, engine, px_xy=0.1083, px_z=0.75):
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