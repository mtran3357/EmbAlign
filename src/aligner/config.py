from enum import Enum, auto
from dataclasses import dataclass

class AtlasStrategy(Enum):
    STATIC = auto()         
    TIME_RESOLVED = auto()  

class SliceStrategy(Enum):
    OBSERVED = auto()       
    AUGMENTED = auto()      

class MatcherType(Enum):
    HUNGARIAN = auto()      
    SINKHORN = auto()       

class InitStrategy(Enum):
    SINGLE = auto()         
    TOURNAMENT = auto()    

@dataclass
class PipelineConfig:
    """Master configuration for the Modular Alignment Engine."""
    
    # --- Architectural Switches ---
    atlas_strategy: AtlasStrategy = AtlasStrategy.TIME_RESOLVED
    slice_strategy: SliceStrategy = SliceStrategy.AUGMENTED
    coarse_matcher: MatcherType = MatcherType.HUNGARIAN  
    icp_matcher: MatcherType = MatcherType.SINKHORN      
    init_strategy: InitStrategy = InitStrategy.TOURNAMENT
    use_slack: bool = False          
    enable_diagnostics: bool = True
    
    # --- Alignment Hyperparameters ---
    angle_step_deg: float = 4.0
    icp_iters: int = 10            
    tau: float = 1e6               
    epsilon_coarse: float = 0.1    
    epsilon_refine: float = 0.01   
    k_tournament: int = 3          
    
    # --- Sinkhorn Internals ---
    sinkhorn_max_iters: int = 100
    sinkhorn_stop_thr: float = 1e-3

    # --- Atlas & Data Quality Thresholds ---
    min_samples_static: int = 5
    min_points_gp: int = 4
    min_count_var: int = 3
    map_t_max: float = 200.0
    
    @classmethod
    def v0_legacy(cls):
        """v0.0: Static Atlas, Observed Slices, Hard Hungarian Matching."""
        return cls(
            atlas_strategy=AtlasStrategy.STATIC,
            slice_strategy=SliceStrategy.OBSERVED,
            coarse_matcher=MatcherType.HUNGARIAN,  
            icp_matcher=MatcherType.HUNGARIAN,
            init_strategy=InitStrategy.SINGLE,
            icp_iters=5,
            use_slack=False 
        )

    @classmethod
    def v1_0_fuzzy(cls):
        """v1.0: Introduces Sinkhorn ICP and soft matching."""
        return cls(
            atlas_strategy=AtlasStrategy.STATIC,
            slice_strategy=SliceStrategy.OBSERVED,
            coarse_matcher=MatcherType.HUNGARIAN,  
            icp_matcher=MatcherType.SINKHORN,
            init_strategy=InitStrategy.SINGLE,
        )

    @classmethod
    def v1_1_tournament(cls):
        """v1.1: Introduces K-valley tournament initialization."""
        config = cls.v1_0_fuzzy()
        config.init_strategy = InitStrategy.TOURNAMENT
        config.k_tournament = 3
        return config

    @classmethod
    def v2_0_dynamic(cls):
        """v2.0: Introduces GP Time-Resolved Atlas."""
        config = cls.v1_1_tournament()
        config.atlas_strategy = AtlasStrategy.TIME_RESOLVED
        config.min_points_gp = 4 
        return config
        
    @classmethod
    def v2_1_augmented(cls):
        """v2.1: Introduces MAP Augmented Slices and Diagnostics."""
        config = cls.v2_0_dynamic()
        config.slice_strategy = SliceStrategy.AUGMENTED
        config.enable_diagnostics = True
        return config
    
    @classmethod
    def v3_0_production(cls):
        """v3.0: Time-Resolved GP Atlas, strictly OBSERVED slices, Full Diagnostics."""
        config = cls.v2_0_dynamic() # Inherits Time-Resolved GP & Tournament Init
        config.slice_strategy = SliceStrategy.OBSERVED # Forces observed slices only
        config.enable_diagnostics = True
        return config