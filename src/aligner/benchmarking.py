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

    def run_sweep(self, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the Two-Pass LOOCV benchmarking sweep for all registered configs.
        
        Args:
            verbose (bool): If True, prints progress bars and live evaluation summaries. 
                            If False, runs completely silently.
        """
        all_frame_results = []
        all_diagnostics = []
        
        embryos = self.full_df['embryo_id'].unique()
        
        for config_name, config in self.configs.items():
            if verbose:
                print(f"\n{'='*50}\nRunning Sweep: {config_name}\n{'='*50}")
            
            pass1_results = {eid: [] for eid in embryos}
            
            # ==========================================
            # PASS 1: UNBIASED ALIGNMENT
            # ==========================================
            if verbose: print("Pass 1: Leave-One-Out Alignment...")
            for test_embryo in tqdm(embryos, desc="LOOCV Folds", disable=not verbose):
                train_embryos = [e for e in embryos if e != test_embryo]
                
                factory = AtlasFactory(self.full_df, config)
                spatial_atlas, slice_db = factory.build(train_embryos)
                
                coarse_m = self._get_matcher(config.coarse_matcher, config)
                icp_m = self._get_matcher(config.icp_matcher, config)
                
                engine = ModularAlignmentEngine(
                    config=config, atlas=spatial_atlas, slice_db=slice_db,
                    coarse_matcher=coarse_m, icp_matcher=icp_m, transformer=self.transformer
                )
                
                test_df = self.full_df[self.full_df['embryo_id'] == test_embryo]
                for t_idx in test_df['time_idx'].unique():
                    try:
                        frame = EmbryoFrame.from_dataframe(self.full_df, test_embryo, t_idx)
                        start_time = time.time()
                        result = engine.align_frame(
                            frame, life_history_df=factory.life_history, 
                            return_diagnostics=config.enable_diagnostics
                        )
                        elapsed = time.time() - start_time
                        
                        if result is not None:
                            result.update({'config_name': config_name, 'embryo_id': test_embryo, 
                                           'time_idx': t_idx, 'elapsed_time': elapsed})
                            if 'diagnostics' in result:
                                result['diagnostics']['config_name'] = config_name
                            pass1_results[test_embryo].append(result)
                            
                    except Exception as e:
                        if verbose: print(f"Skipped {test_embryo} T={t_idx}: {e}")

            # ==========================================
            # PASS 2: LEAVE-ONE-OUT ORACLE
            # ==========================================
            if config.enable_diagnostics:
                if verbose: print("Pass 2: Leave-One-Out Diagnostic Training...")
                for test_embryo in tqdm(embryos, desc="Training Oracles", disable=not verbose):
                    train_diags = [res['diagnostics'] for eid, results in pass1_results.items() 
                                   if eid != test_embryo for res in results if 'diagnostics' in res]
                    
                    if train_diags:
                        oracle = DiagnosticLayer(training_data=pd.concat(train_diags, ignore_index=True))
                        for res in pass1_results[test_embryo]:
                            res.update(oracle.predict_confidence(res))

            # ==========================================
            # AGGREGATION & LIVE REPORTING
            # ==========================================
            cfg_frame_records = []
            cfg_diag_records = []
            
            for eid, results in pass1_results.items():
                for res in results:
                    if 'diagnostics' in res:
                        diag_df = res.pop('diagnostics')
                        cfg_diag_records.append(diag_df)
                        all_diagnostics.append(diag_df)
                    
                    flat_res = {k: v for k, v in res.items() if k not in ['coords', 'labels']}
                    flat_res['num_cells_assigned'] = len(res.get('labels', []))
                    flat_res['total_cost'] = res.get('cost', np.nan)
                    
                    cfg_frame_records.append(flat_res)
                    all_frame_results.append(flat_res)
            
            # --- The Verbose Live Evaluation Toggle ---
            if verbose and cfg_frame_records:
                cfg_frame_df = pd.DataFrame(cfg_frame_records)
                cfg_diag_df = pd.concat(cfg_diag_records, ignore_index=True) if cfg_diag_records else pd.DataFrame()
                
                eval_df = PipelineEvaluator.evaluate_benchmark(cfg_frame_df, cfg_diag_df, self.full_df)
                
                pos_acc = eval_df['positional_accuracy'].mean() * 100
                slice_hit = eval_df['slice_match'].mean() * 100
                conf = cfg_frame_df['mean_confidence'].mean() * 100 if 'mean_confidence' in cfg_frame_df.columns else float('nan')
                
                print(f"\n" + "-"*40)
                print(f"[{config_name}] SWEEP COMPLETE")
                print(f"Positional Accuracy:  {pos_acc:.1f}%")
                print(f"Perfect Slice Match:  {slice_hit:.1f}%")
                if not np.isnan(conf):
                    print(f"Mean Oracle Confidence: {conf:.1f}%")
                print("-" * 40 + "\n")

        final_frame_df = pd.DataFrame(all_frame_results) if all_frame_results else pd.DataFrame()
        final_diag_df = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
        
        return final_frame_df, final_diag_df