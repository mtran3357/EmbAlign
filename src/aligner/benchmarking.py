import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import Dict, List
from aligner.engine import LegacyEngine, EngineV1, EngineV2, EngineV3
from aligner.matcher import HungarianMatcher, SinkhornMatcher
from aligner.plot_utils import SpatialVisualizer
from aligner.runner import BatchReporter, BatchRunner
from aligner.models import EmbryoFrame
from aligner.atlas import GPTimeAtlas, GPToStaticAdapter, AtlasBuilder

class BenchmarkingSuite:
    """
    Orchestrates comparisonsbetween different alignment ppieline configurations.
    """
    ENGINE_REGISTRY = {
        "legacy": LegacyEngine,
        "v1": EngineV1,
        "v2": EngineV2,
        "v3": EngineV3
    }
    DYNAMIC_ENGINES = {"v2", "v3"}
    
    def __init__(self, atlas, slice_db, transformer):
        self.atlas = atlas
        self.slice_db = slice_db
        self.transformer = transformer
        self.configs = {}
        self.viz = SpatialVisualizer(atlas)
        
    def add_config(self, name: str, engine_type: str = "legacy", 
                   matcher_type: str = "hungarian", settings: dict = None,
                   override_atlas=None, override_slice_db=None):
        """
        Registers a specific engine configuration for benchmarking.
        Automatically adapts dynamic GP atlases for static engines (Legacy/V1).
        """
        settings = settings or {}
        base_atlas = override_atlas or self.atlas
        active_slice_db = override_slice_db or self.slice_db
        
        # Select Matcher
        if matcher_type == "sinkhorn":
            matcher = SinkhornMatcher(
                epsilon=settings.get('epsilon_coarse', 0.05),
                max_iters=settings.get('max_iters', 100)
            )
        else: 
            matcher = HungarianMatcher(tau=settings.get('tau_strict', 1e6))
        
        #Atlas Adaptation Logic
        active_atlas = base_atlas
        if engine_type not in self.DYNAMIC_ENGINES and isinstance(base_atlas, GPTimeAtlas):
            static_state = base_atlas.get_state(100.0)
            active_atlas = GPToStaticAdapter(static_state['labels'], static_state['means'], static_state['variances'])
        
        # Instantiate via Registry
        engine_class = self.ENGINE_REGISTRY.get(engine_type, LegacyEngine)
        engine = engine_class(active_atlas, active_slice_db, matcher, self.transformer, settings)
            
        self.configs[name] = engine
        print(f"Added {engine_type.upper()} config: '{name}'")
        
    # def compare_frame(self, frame, title_prefix=""):
    #     """Visual comparison of all registered configs on a single frame."""
    #     n_configs = len(self.configs)
    #     fig = plt.figure(figsize=(5 * n_configs, 5))
    #     comparison_data = []
        
    #     frame.prepare() 
        
    #     for i, (name, engine) in enumerate(self.configs.items()):
    #         start = time.time()
    #         result = engine.align_frame(frame) 
    #         elapsed = time.time() - start 
    
    #         # Standard Accuracy Check
    #         true_labels = [str(l).upper() for l in frame.valid_df['cell_name']]
    #         pred_labels = [str(l).upper() for l in result['labels']]
    #         acc = np.mean([p == t for p, t in zip(pred_labels, true_labels)])
            
    #         comparison_data.append({
    #             "config": name, 
    #             "accuracy": acc,
    #             "cost": result['cost'],
    #             "runtime": elapsed,
    #             "engine_type": engine.__class__.__name__
    #         })
    
    #         ax = fig.add_subplot(1, n_configs, i + 1, projection='3d')
    #         full_title = f"{name}\nAcc: {acc:.1%} | Cost: {result['cost']:.2f}"
            
    #         # Temporary re-assignment of visualizer atlas to match engine's data source
    #         old_viz_atlas = self.viz.atlas
    #         viz_atlas = engine.atlas
            
    #         # If the engine is using a GP Atlas, get a specific snapshot for this frame's time
    #         if isinstance(viz_atlas, GPTimeAtlas):
    #             t = frame.valid_df['canonical_time'].iloc[0] if (frame.valid_df is not None and 'canonical_time' in frame.valid_df.columns) else 100.0
    #             state = viz_atlas.get_state(t)
    #             viz_atlas = GPToStaticAdapter(state['labels'], state['means'], state['variances'])
            
    #         self.viz.atlas = viz_atlas
    #         self.viz.plot_alignment(frame, result, ax=ax, title=full_title)
    #         self.viz.atlas = old_viz_atlas # Restore original
            
    #     plt.tight_layout()
    #     plt.show()
    #     return pd.DataFrame(comparison_data)
    
    def compare_frame(self, frame, title_prefix=""):
        n_configs = len(self.configs)
        fig, comparison_data = plt.figure(figsize=(5 * n_configs, 5)), []
        frame.prepare() 
        
        for i, (name, engine) in enumerate(self.configs.items()):
            start = time.time()
            result = engine.align_frame(frame) 
            elapsed = time.time() - start 
    
            # Define accuracy to avoid NameError
            true_labels = [str(l).upper() for l in frame.valid_df['cell_name']] if frame.valid_df is not None else []
            pred_labels = [str(l).upper() for l in result.get('labels', [])]
            acc = np.mean([p == t for p, t in zip(pred_labels, true_labels)]) if true_labels else 0.0
            
            comparison_data.append({"config": name, "accuracy": acc, "cost": result['cost'], "runtime": elapsed})
    
            ax = fig.add_subplot(1, n_configs, i + 1, projection='3d')
            full_title = f"{title_prefix}\n{name}\nAcc: {acc:.1%} | Cost: {result['cost']:.2f}"
            
            # Visualizer fix for GPTimeAtlas
            old_viz_atlas = self.viz.atlas
            viz_atlas = engine.atlas
            if isinstance(viz_atlas, GPTimeAtlas):
                t = frame.valid_df['canonical_time'].iloc[0] if frame.valid_df is not None else 100.0
                state = viz_atlas.get_state(t)
                viz_atlas = GPToStaticAdapter(state['labels'], state['means'], state['variances'])
            
            self.viz.atlas = viz_atlas
            self.viz.plot_alignment(frame, result, ax=ax, title=full_title)
            self.viz.atlas = old_viz_atlas
            
        plt.tight_layout()
        plt.show()
        return pd.DataFrame(comparison_data)
    
    # def run_sweep(self, df: pd.DataFrame, metadata_ref: pd.DataFrame = None, return_diagnostics: bool = False):
    #     """Runs all registered configs across a dataset (e.g., full embryos)."""
    #     all_reports = []
    #     ref_df = metadata_ref if metadata_ref is not None else df
        
    #     for name, engine in self.configs.items():
    #         reporter = BatchReporter(full_df=ref_df)
    #         runner = BatchRunner(engine, reporter)
    #         runner.run(df, return_diagnostics=return_diagnostics)
            
    #         summary = reporter.summarize_frame_diagnostics()
    
    #         diag_report = reporter.get_diagnostic_report()
            
    #         return summary, diag_report
        
    def run_sweep(self, df: pd.DataFrame, metadata_ref: pd.DataFrame = None, life_history_df: pd.DataFrame = None, return_diagnostics: bool = False):
        """Runs all registered configs and aggregates results with labels."""
        all_summaries = []
        all_diag_reports = []
        ref_df = metadata_ref if metadata_ref is not None else df
        
        for name, engine in self.configs.items():
            print(f"Running engine: {name}")
            reporter = BatchReporter(full_df=ref_df)
            runner = BatchRunner(engine, reporter)
            runner.run(df, return_diagnostics=return_diagnostics, life_history_df=life_history_df)
            
            # 1. Get reports
            summary = reporter.summarize_frame_diagnostics()
            diag_report = reporter.get_diagnostic_report()
            
            # 2. Inject the configuration name so results are identifiable
            if summary is not None:
                summary['config_name'] = name
                all_summaries.append(summary)
                
            if return_diagnostics and diag_report is not None:
                diag_report['config_name'] = name
                all_diag_reports.append(diag_report)
        
        # 3. Concatenate all configs into a unified report
        final_summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
        final_diag = pd.concat(all_diag_reports, ignore_index=True) if all_diag_reports else pd.DataFrame()
        
        return final_summary, final_diag

    def run_cv(self, full_df: pd.DataFrame, builder: 'AtlasBuilder', withheld_ids: List[str] = None):
        """
        Executes a Leave-One-Embryo-Out (LOEO) CV.
        """
        all_embryos = full_df['embryo_id'].unique()
        all_results = []
        all_diagnostics = []
        
        for test_id in tqdm(all_embryos, desc="CV Fold"):
            # 1. Setup Train/Test Split
            train_ids = [eid for eid in all_embryos if eid != test_id]
            
            # 2. Fit fold-specific Atlas
            print(f"Fitting atlas (fold: {test_id})")
            gp_atlas, slice_db = builder.fit(train_ids)
            
            # 3. Update the suite's state for this fold
            self.gp_atlas = gp_atlas
            self.slice_db = slice_db
            
            # 4. Sweep the test embryo
            test_df = full_df[full_df['embryo_id'] == test_id].copy()
            res, diag = self.run_sweep(test_df, return_diagnostics=True)
            
            # 5. Tag and Accumulate
            res['test_embryo'] = test_id
            diag['test_embryo'] = test_id
            all_results.append(res)
            all_diagnostics.append(diag)
            
        return pd.concat(all_results), pd.concat(all_diagnostics)
    
    def run_cv(self, full_df: pd.DataFrame, builder: 'AtlasBuilder'):
        all_embryos = full_df['embryo_id'].unique()
        all_results = []
        all_diagnostics = []
        
        for test_id in tqdm(all_embryos, desc="CV Progress"):
            try:
                # 1. Fit Fold
                train_ids = [eid for eid in all_embryos if eid != test_id]
                gp_atlas, slice_db = builder.fit(train_ids)
                lh = builder.life_history  # <--- Capture fold-specific LH
                
                # 2. Update Suite References
                for name in self.configs:
                    # Point engine to the new atlas/db
                    self.configs[name].atlas = gp_atlas
                    self.configs[name].slice_db = slice_db
                
                # 3. Prepare Test Data
                test_df = full_df[full_df['embryo_id'] == test_id].copy()
                # Ensure aggregate columns are prepared
                test_df = prepare_sweep_df(test_df) 
                
                # 4. Sweep with LH injected
                res, diag = self.run_sweep(
                    test_df, 
                    life_history_df=lh,  # <--- Pass LH here
                    return_diagnostics=True
                )
                
                # 5. Tag and Accumulate
                if res is not None:
                    res['test_embryo'] = test_id
                    all_results.append(res)
                if diag is not None:
                    diag['test_embryo'] = test_id
                    all_diagnostics.append(diag)
                    
            except Exception as e:
                print(f"Failed fold {test_id}: {e}")
                continue
            
        return pd.concat(all_results), pd.concat(all_diagnostics)

            
