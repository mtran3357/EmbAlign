import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import Dict, List
from aligner.engine import LegacyEngine
from aligner.matcher import HungarianMatcher, SinkhornMatcher
from aligner.plot_utils import SpatialVisualizer
from aligner.runner import BatchReporter
from aligner.models import EmbryoFrame

class BenchmarkingSuite:
    """
    Orchestrates comparisonsbetween different alignment ppieline configurations.
    """
    def __init__(self, atlas, slice_db, transformer):
        self.atlas = atlas
        self.slice_db = slice_db
        self.transformer = transformer
        self.configs: Dict[str, LegacyEngine] = {}
        self.viz = SpatialVisualizer(atlas)
        
    def add_config(self, name: str, matcher_type: str = "hungarian", settings: dict = None):
        """Modularly registers a new pipeline configuration."""
        if matcher_type == "sinkhorn":
            matcher = SinkhornMatcher(
                epsilon.settings.get('epsilon', 0.05) if settings else 0.05
            )
        else: 
            matcher = HungarianMatcher()
        
        engine = LegacyEngine(
            self.atlas, self.slice_db, matcher, self.transformer, settings
        )
        self.configs[name] = engine
    
    def compare_frame(self, frame, title_prefix = ""):
        """Runs all configs on one frame and shows side by side 3D plots."""
        n_configs = len(self.configs)
        fig = plt.figure(figsize=(5 * n_configs, 5))
        
        comparison_data = []
        
        for i, (name, engine) in enumerate(self.configs.items()):
            start = time.time()
            frame_copy = frame
            result = engine.align_frame(frame_copy)
            elapsed = time.time() - start 
    
            true_labels = frame.valid_df['cell_name'].astype(str).tolist()
            acc = np.mean([inf == tru for inf, tru in zip(result['labels'], true_labels)])
            
            comparison_data.append({
                "config": name, 
                "accuracy": acc,
                "cost": result['cost'],
                "runtime": elapsed
            })
    
            ax = fig.add_subplot(1, n_configs, i + 1, projection='3d')
            full_title = f"{title_prefix}\nConfig: {name}\nAcc: {acc:.1%} | Cost: {result['cost']:.2f}"
            self.viz.plot_alignment(frame, result, ax=ax, title=full_title)
            
        plt.tight_layout()
        plt.show()
        return pd.DataFrame(comparison_data)
    
    def run_sweep(self, full_df, samples_df):
        """
        Fixed implementation to ensure results are captured and returned correctly.
        """
        all_reports = []
        
        for name, engine in self.configs.items():
            print(f"\n>>> PROCESSING CONFIG: {name}")
            # Initialize a fresh reporter for this configuration
            reporter = BatchReporter()
            success_count = 0
            
            for _, row in samples_df.iterrows():
                eid = row['embryo_id']
                tid = row['time_idx']
                
                try:
                    # Load the frame
                    frame = EmbryoFrame.from_dataframe(full_df, eid, tid)
                    
                    # Align and time it
                    start_time = time.time()
                    result = engine.align_frame(frame)
                    elapsed = time.time() - start_time
                    
                    # Add to the local reporter instance
                    if result:
                        reporter.add_result(frame, result, elapsed, atlas=self.slice_db)
                        success_count += 1
                        
                except Exception as e:
                    continue
            
            print(f" -> Successfully aligned {success_count} / {len(samples_df)} frames.")
            
            # Extract the diagnostic summary
            summary = reporter.summarize_frame_diagnostics()
            if not summary.empty:
                summary['config_name'] = name
                all_reports.append(summary)
                
        # Concatenate all config reports into the final result
        if not all_reports:
            print("CRITICAL: No reports were generated. Check the 'valid' column in full_df.")
            return pd.DataFrame()
            
        return pd.concat(all_reports, ignore_index=True)

        