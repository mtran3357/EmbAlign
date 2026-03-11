# src/aligner/__init__.py

# 1. Models and Data Structures
from .models import EmbryoFrame, ReferenceFrame

# 2. Configuration and Enums
from .config import (
    PipelineConfig, 
    AtlasStrategy, 
    SliceStrategy, 
    MatcherType, 
    InitStrategy
)

# 3. Core Logic Components
from .atlas import StaticGaussianAtlas, AtlasFactory, SliceTimeAtlas
from .matcher import HungarianMatcher, SinkhornMatcher
from .engine import ModularAlignmentEngine
from .transformer import RigidTransformer

# 4. Benchmarking and Execution
from .benchmarking import BenchmarkingSuite
from .oracle import DiagnosticLayer
from .runner import PipelineEvaluator, InferenceRunner