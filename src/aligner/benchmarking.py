import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import Dict, List, Tuple

from aligner.config import PipelineConfig, MatcherType
from aligner.atlas import AtlasFactory
from aligner.matcher import HungarianMatcher, SinkhornMatcher
from aligner.engine import ModularAlignmentEngine
from aligner.oracle import DiagnosticLayer
from aligner.models import EmbryoFrame
from aligner.runner import PipelineEvaluator

class BenchmarkingSuite:
    """
    Orchestrates Leave-One-Out Cross-Validation (LOOCV) sweeps across 
    different modular alignment pipeline configurations.
    """
    def __init__(self, full_df: pd.DataFrame, transformer):
        self.full_df = full_df
        self.transformer = transformer
        self.configs: Dict[str, PipelineConfig] = {}
        
    def add_config(self, name: str, config: PipelineConfig):
        """Registers a specific pipeline configuration for the benchmark sweep."""
        self.configs[name] = config
        print(f"Registered configuration: {name}")

    def _get_matcher(self, matcher_type: MatcherType, config: PipelineConfig):
        """Dynamically instantiates the requested matcher with config parameters."""
        if matcher_type == MatcherType.HUNGARIAN:
            return HungarianMatcher(tau=config.tau)
        elif matcher_type == MatcherType.SINKHORN:
            # Coarse vs Refine epsilon is handled by the engine passing kwargs,
            # but we set the default here.
            return SinkhornMatcher(epsilon=config.epsilon_refine, 
                                   stop_thr=config.sinkhorn_stop_thr,
                                   max_iters=config.sinkhorn_max_iters
                                   )
        else:
            raise ValueError(f"Unknown matcher type: {matcher_type}")

    def run_sweep(self, verbose: bool = True, limit_folds: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the Two-Pass LOOCV benchmarking sweep for all registered configs.
        Returns evaluated DataFrames containing positional and slice accuracies.
        """
        all_frame_results = []
        all_diagnostics = []
        
        # 1. Define the complete universe of embryos for training
        all_embryos = self.full_df['embryo_id'].unique()
        
        # 2. Define the restricted list of embryos for testing
        if limit_folds is not None:
            test_embryos = all_embryos[:limit_folds]
            if verbose: print(f"Testing restricted to {limit_folds} fold(s): {test_embryos}")
        else:
            test_embryos = all_embryos
        
        for config_name, config in self.configs.items():
            if verbose:
                print(f"\n{'='*50}\nRunning Sweep: {config_name}\n{'='*50}")
            
            pass1_results = {eid: [] for eid in test_embryos}
            
            # ==========================================
            # PASS 1: UNBIASED ALIGNMENT
            # ==========================================
            if verbose: print("Pass 1: Leave-One-Out Alignment...")
            for test_embryo in tqdm(test_embryos, desc="LOOCV Folds", disable=not verbose):
                
                # FIX: Train on the full dataset minus the test embryo
                train_embryos = [e for e in all_embryos if e != test_embryo]
                
                # Build the fresh atlas/slice DB for this fold
                factory = AtlasFactory(self.full_df, config)
                spatial_atlas, slice_db = factory.build(train_embryos)
                
                engine = ModularAlignmentEngine(
                    config=config, 
                    atlas=spatial_atlas, 
                    slice_db=slice_db,
                    coarse_matcher=self._get_matcher(config.coarse_matcher, config),
                    icp_matcher=self._get_matcher(config.icp_matcher, config),
                    transformer=self.transformer
                )
                
                test_df = self.full_df[self.full_df['embryo_id'] == test_embryo]
                for t_idx in test_df['time_idx'].unique():
                    try:
                        frame = EmbryoFrame.from_dataframe(self.full_df, test_embryo, t_idx)
                        result = engine.align_frame(
                            frame, 
                            life_history_df=factory.life_history, 
                            return_diagnostics=config.enable_diagnostics
                        )
                        
                        if result is not None:
                            # Standardize metadata types immediately
                            result.update({
                                'config_name': config_name, 
                                'embryo_id': str(test_embryo), 
                                'time_idx': int(t_idx)
                            })
                            # Ensure diagnostics carry metadata for the evaluator
                            if 'diagnostics' in result and result['diagnostics'] is not None:
                                result['diagnostics']['config_name'] = config_name
                                result['diagnostics']['embryo_id'] = str(test_embryo)
                                result['diagnostics']['time_idx'] = int(t_idx)
                            
                            pass1_results[test_embryo].append(result)
                            
                    except Exception as e:
                        if verbose: print(f"Skipped {test_embryo} T={t_idx}: {e}")

            # ==========================================
            # PASS 2: LEAVE-ONE-OUT ORACLE
            # ==========================================
            if config.enable_diagnostics:
                if verbose: print("Pass 2: Leave-One-Out Diagnostic Training...")
                for test_embryo in tqdm(test_embryos, desc="Training Oracles", disable=not verbose):
                    train_diags = [
                        res['diagnostics'] for eid, results in pass1_results.items() 
                        if str(eid) != str(test_embryo) 
                        for res in results if 'diagnostics' in res
                    ]
                    
                    if train_diags:
                        oracle = DiagnosticLayer(training_data=pd.concat(train_diags, ignore_index=True))
                        for res in pass1_results[test_embryo]:
                            res.update(oracle.predict_confidence(res))

            # ==========================================
            # CONFIG AGGREGATION
            # ==========================================
            cfg_frame_records = []
            cfg_diag_records = []
            
            for eid, results in pass1_results.items():
                for res in results:
                    if 'diagnostics' in res and res['diagnostics'] is not None:
                        d_df = res.pop('diagnostics')
                        cfg_diag_records.append(d_df)
                        all_diagnostics.append(d_df)
                    
                    # Flatten result for the frame-level DataFrame
                    flat_res = {k: v for k, v in res.items() if k not in ['coords', 'ref_frame']}
                    flat_res['num_cells_assigned'] = len(res.get('labels', []))
                    all_frame_results.append(flat_res)
                    cfg_frame_records.append(flat_res)

            # --- Live Reporting Block (Optimized) ---
            if verbose and cfg_frame_records:
                temp_frames = pd.DataFrame(cfg_frame_records)
                temp_diags_df = pd.concat(cfg_diag_records, ignore_index=True) if cfg_diag_records else pd.DataFrame()
                
                # Evaluate this config specifically
                eval_df = PipelineEvaluator.evaluate_benchmark(temp_frames, temp_diags_df, self.full_df)
                
                print(f"\n" + "-"*40)
                print(f"[{config_name}] SWEEP COMPLETE")
                print(f"Positional Accuracy:  {eval_df['positional_accuracy'].mean()*100:.1f}%")
                print(f"Slice Overlap (Acc):  {eval_df['slice_accuracy'].mean()*100:.1f}%")
                print(f"Perfect Matches:      {int(eval_df['slice_match'].sum())}/{len(eval_df)}")
                if 'mean_confidence' in temp_frames.columns:
                    print(f"Mean Oracle Confidence: {temp_frames['mean_confidence'].mean()*100:.1f}%")
                print("-" * 40 + "\n")

        # ==========================================
        # FINAL GLOBAL EVALUATION
        # ==========================================
        final_frame_df = pd.DataFrame(all_frame_results)
        final_diag_df = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
        
        if not final_frame_df.empty:
            final_frame_df = PipelineEvaluator.evaluate_benchmark(
                final_frame_df, 
                final_diag_df, 
                self.full_df
            )
        
        return final_frame_df, final_diag_df