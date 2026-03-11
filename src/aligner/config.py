from enum import Enum, auto
from dataclasses import dataclass

class AtlasStrategy(Enum):
    STATIC = auto()         # v0.0, v1.0, v1.1
    TIME_RESOLVED = auto()  # v2.0, v2.1

class SliceStrategy(Enum):
    OBSERVED = auto()       # v0.0 through v2.0
    AUGMENTED = auto()      # v2.1 (MAP estimated)

class MatcherType(Enum):
    HUNGARIAN = auto()      # v0.0 (Hard 1-to-1)
    SINKHORN = auto()       # v1.0+ (Fuzzy/Soft)

class InitStrategy(Enum):
    SINGLE = auto()         # v0.0, v1.0 (Winner-take-all coarse scan)
    TOURNAMENT = auto()     # v1.1+ (Top-K valleys)

@dataclass
class PipelineConfig:
    """Master configuration for the Modular Alignment Engine."""
    
    # --- Architectural Switches ---
    atlas_strategy: AtlasStrategy = AtlasStrategy.TIME_RESOLVED
    slice_strategy: SliceStrategy = SliceStrategy.AUGMENTED
    coarse_matcher: MatcherType = MatcherType.HUNGARIAN  
    icp_matcher: MatcherType = MatcherType.SINKHORN
    init_strategy: InitStrategy = InitStrategy.TOURNAMENT
    use_slack: bool = True
    enable_diagnostics: bool = True
    
    # --- Hyperparameters ---
    angle_step_deg: float = 4.0
    icp_iters: int = 10            # Default 10 for soft ICP, 5 for legacy hard ICP
    tau: float = 1e6               # Slack cost for unassigned cells
    epsilon_coarse: float = 0.1    # Sinkhorn entropy regularization for scanning
    epsilon_refine: float = 0.01   # Sinkhorn entropy regularization for refinement
    k_tournament: int = 3          # Number of valleys to explore during initialization
    
    @classmethod
    def v0_legacy(cls):
        """v0.0: Static Atlas, Observed Slices, Hard Hungarian Matching."""
        return cls(
            atlas_strategy=AtlasStrategy.STATIC,
            slice_strategy=SliceStrategy.OBSERVED,
            matcher_type=MatcherType.HUNGARIAN,
            init_strategy=InitStrategy.SINGLE,
            enable_diagnostics=False,
            icp_iters=5 
        )

    @classmethod
    def v1_0_fuzzy(cls):
        """v1.0: Introduces Sinkhorn ICP and soft matching."""
        return cls(
            atlas_strategy=AtlasStrategy.STATIC,
            slice_strategy=SliceStrategy.OBSERVED,
            matcher_type=MatcherType.SINKHORN,
            init_strategy=InitStrategy.SINGLE,
            enable_diagnostics=False
        )

    @classmethod
    def v1_1_tournament(cls):
        """v1.1: Introduces K-valley tournament initialization."""
        config = cls.v1_0_fuzzy()
        config.init_strategy = InitStrategy.TOURNAMENT
        return config

    @classmethod
    def v2_0_dynamic(cls):
        """v2.0: Introduces GP Time-Resolved Atlas."""
        config = cls.v1_1_tournament()
        config.atlas_strategy = AtlasStrategy.TIME_RESOLVED
        return config
        
    @classmethod
    def v2_1_augmented(cls):
        """v2.1: Introduces MAP Augmented Slices and Diagnostics."""
        config = cls.v2_0_dynamic()
        config.slice_strategy = SliceStrategy.AUGMENTED
        config.enable_diagnostics = True
        return config