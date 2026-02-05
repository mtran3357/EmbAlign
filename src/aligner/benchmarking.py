import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import Dict, List
from aligner.engine import LegacyEngine, EngineV1
from aligner.matcher import HungarianMatcher, SinkhornMatcher
from aligner.plot_utils import SpatialVisualizer
from aligner.runner import BatchReporter, BatchRunner
from aligner.models import EmbryoFrame

class BenchmarkingSuite:
    """
    Orchestrates comparisonsbetween different alignment ppieline configurations.
    """
    def __init__(self, atlas, slice_db, transformer):
        self.atlas = atlas
        self.slice_db = slice_db
        self.transformer = transformer
        self.configs = {}
        self.viz = SpatialVisualizer(atlas)
        
    def add_config(self, name: str, engine_type: str = "legacy", matcher_type: str = "hungarian", settings: dict = None):
        """
        Updated to support both Engine types.
        engine_type: "legacy" or "v1"
        """
        settings = settings if settings is not None else {}
        
        # 1. Select Matcher
        if matcher_type == "sinkhorn":
            matcher = SinkhornMatcher(
                epsilon=settings.get('epsilon_coarse', 0.05),
                max_iters=settings.get('max_iters', 100)
            )
        else: 
            matcher = HungarianMatcher(tau=settings.get('tau_strict', 1e6))
        
        # 2. Select Engine Architecture
        if engine_type == "v1":
            engine = EngineV1(self.atlas, self.slice_db, matcher, self.transformer, settings)
        else:
            engine = LegacyEngine(self.atlas, self.slice_db, matcher, self.transformer, settings)
            
        self.configs[name] = engine
    
    def compare_frame(self, frame, title_prefix=""):
        n_configs = len(self.configs)
        fig = plt.figure(figsize=(5 * n_configs, 5))
        comparison_data = []
        
        # Ensure frame is prepared once for all configs
        frame.prepare() 
        
        for i, (name, engine) in enumerate(self.configs.items()):
            start = time.time()
            
            # CRITICAL: We call WITHOUT trace=True so it returns only the result dict
            result = engine.align_frame(frame) 
            elapsed = time.time() - start 
    
            # Standard Accuracy Check
            true_labels = [str(l).upper() for l in frame.valid_df['cell_name']]
            pred_labels = [str(l).upper() for l in result['labels']]
            
            # Handle list length mismatches if they occur
            acc = np.mean([p == t for p, t in zip(pred_labels, true_labels)])
            
            comparison_data.append({
                "config": name, 
                "accuracy": acc,
                "cost": result['cost'],
                "runtime": elapsed,
                "engine_type": engine.__class__.__name__
            })
    
            ax = fig.add_subplot(1, n_configs, i + 1, projection='3d')
            full_title = f"{name}\nAcc: {acc:.1%} | Cost: {result['cost']:.2f}"
            self.viz.plot_alignment(frame, result, ax=ax, title=full_title)
            
        plt.tight_layout()
        plt.show()
        return pd.DataFrame(comparison_data)
    
    def run_sweep(self, df: pd.DataFrame, metadata_ref: pd.DataFrame = None):
        """
        Runs all registered configs on the provided dataframe.
        To run a subset, pre-filter the 'df' before passing it in.
        """
        all_reports = []
        
        # If metadata_ref isn't provided, we assume 'df' contains 
        # all necessary columns (like canonical_time)
        ref_df = metadata_ref if metadata_ref is not None else df
        
        for name, engine in self.configs.items():
            # 1. Initialize Reporter with the reference metadata
            reporter = BatchReporter(full_df=ref_df)
            
            # 2. Instantiate the BatchRunner with the specific configuration
            runner = BatchRunner(engine, reporter)
            
            # 3. Trigger the professional run with progress tracking
            # This handles grouping by embryo_id and time_idx internally
            runner.run(df)
            
            # 4. Extract diagnostic summary
            summary = reporter.summarize_frame_diagnostics()
            if not summary.empty:
                summary['config_name'] = name
                all_reports.append(summary)
                
        if not all_reports:
            print(">>> WARNING: No frames were successfully aligned. Check dataframe filtering.")
            return pd.DataFrame()
            
        return pd.concat(all_reports, ignore_index=True)