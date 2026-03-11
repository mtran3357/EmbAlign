import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import Dict, List, Tuple

# Import our newly refactored modular components
from config import PipelineConfig, MatcherType
from atlas import AtlasFactory
from matcher import HungarianMatcher, SinkhornMatcher
from engine import ModularAlignmentEngine
from oracle import DiagnosticLayer
from models import EmbryoFrame

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
            return SinkhornMatcher(epsilon=config.epsilon_refine, stop_thr=1e-3)
        else:
            raise ValueError(f"Unknown matcher type: {matcher_type}")

    def run_sweep(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the Two-Pass LOOCV benchmarking sweep for all registered configs.
        Returns flattened DataFrames of all frame-level and cell-level results.
        """
        all_frame_results = []
        all_diagnostics = []
        
        # Get unique embryos for LOOCV
        embryos = self.full_df['embryo_id'].unique()
        
        for config_name, config in self.configs.items():
            print(f"\n{'='*50}\nRunning Sweep: {config_name}\n{'='*50}")
            
            # Temporary storage for Pass 1 (Unbiased Alignment)
            pass1_results = {eid: [] for eid in embryos}
            
            # ==========================================
            # PASS 1: UNBIASED ALIGNMENT (Generate Features)
            # ==========================================
            print("Pass 1: Leave-One-Out Alignment...")
            for test_embryo in tqdm(embryos, desc="LOOCV Folds"):
                train_embryos = [e for e in embryos if e != test_embryo]
                
                # 1. Build Atlas on Training Data ONLY
                factory = AtlasFactory(self.full_df, config)
                spatial_atlas, slice_db = factory.build(train_embryos)
                
                # 2. Instantiate Matchers and Engine
                coarse_m = self._get_matcher(config.coarse_matcher, config)
                icp_m = self._get_matcher(config.icp_matcher, config)
                
                engine = ModularAlignmentEngine(
                    config=config,
                    atlas=spatial_atlas,
                    slice_db=slice_db,
                    coarse_matcher=coarse_m,
                    icp_matcher=icp_m,
                    transformer=self.transformer
                )
                
                # 3. Align the Test Embryo
                test_df = self.full_df[self.full_df['embryo_id'] == test_embryo]
                time_indices = test_df['time_idx'].unique()
                
                for t_idx in time_indices:
                    try:
                        # Build the observation frame
                        frame = EmbryoFrame.from_dataframe(self.full_df, test_embryo, t_idx)
                        
                        start_time = time.time()
                        # Pass life_history so engine can calculate div_delta
                        result = engine.align_frame(
                            frame, 
                            life_history_df=factory.life_history, 
                            return_diagnostics=config.enable_diagnostics
                        )
                        elapsed = time.time() - start_time
                        
                        if result is not None:
                            # Tag metadata
                            result['config_name'] = config_name
                            result['embryo_id'] = test_embryo
                            result['time_idx'] = t_idx
                            result['elapsed_time'] = elapsed
                            if 'diagnostics' in result:
                                result['diagnostics']['config_name'] = config_name
                                
                            pass1_results[test_embryo].append(result)
                            
                    except Exception as e:
                        print(f"Skipped {test_embryo} T={t_idx}: {e}")

            # ==========================================
            # PASS 2: LEAVE-ONE-OUT ORACLE (Train Diagnostic Layer)
            # ==========================================
            if config.enable_diagnostics:
                print("Pass 2: Leave-One-Out Diagnostic Training...")
                for test_embryo in tqdm(embryos, desc="Training Oracles"):
                    # 1. Gather OOF (Out-Of-Fold) Diagnostics for Training
                    train_diags = []
                    for train_embryo in embryos:
                        if train_embryo != test_embryo:
                            for res in pass1_results[train_embryo]:
                                if 'diagnostics' in res:
                                    train_diags.append(res['diagnostics'])
                    
                    if not train_diags:
                        continue
                        
                    train_df = pd.concat(train_diags, ignore_index=True)
                    
                    # 2. Train the Oracle on OOF data
                    oracle = DiagnosticLayer(training_data=train_df)
                    
                    # 3. Score the Test Embryo
                    for res in pass1_results[test_embryo]:
                        scored_res = oracle.predict_confidence(res)
                        # Replace the unscored result with the scored one
                        res.update(scored_res)

            # ==========================================
            # AGGREGATION: Flatten the dictionaries
            # ==========================================
            for eid, results in pass1_results.items():
                for res in results:
                    # Separate the diagnostics DataFrame from the main dictionary
                    if 'diagnostics' in res:
                        diag_df = res.pop('diagnostics')
                        all_diagnostics.append(diag_df)
                    
                    # Flatten the rest for frame-level results
                    # (Removing heavy arrays like 'coords' to keep the dataframe clean)
                    flat_res = {k: v for k, v in res.items() if k not in ['coords', 'labels']}
                    flat_res['num_cells_assigned'] = len(res.get('labels', []))
                    flat_res['total_cost'] = res.get('cost', np.nan)
                    all_frame_results.append(flat_res)

        # Concatenate everything into two master DataFrames
        final_frame_df = pd.DataFrame(all_frame_results) if all_frame_results else pd.DataFrame()
        final_diag_df = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
        
        return final_frame_df, final_diag_df