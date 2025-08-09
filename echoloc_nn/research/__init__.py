"""
EchoLoc-NN Research Framework

Experimental and research components for advancing quantum-inspired spatial localization.
This module provides tools for comparative studies, algorithm development, and academic research.

Components:
- experimental: Novel algorithmic implementations
- benchmarks: Comprehensive evaluation frameworks  
- comparative_studies: A/B testing and statistical analysis
- publications: Publication-ready experiment tools
"""

from .experimental import (
    QuantumSpatialFusion,
    AdaptiveQuantumPlanner,
    NovelEchoAttention,
    HybridQuantumCNN
)

from .benchmarks import (
    ResearchBenchmarkSuite,
    AlgorithmComparator, 
    StatisticalValidator,
    PerformanceProfiler
)

from .comparative_studies import (
    BaselineComparator,
    ExperimentalDesign,
    ResultsAnalyzer,
    SignificanceTester
)

from .publications import (
    PublicationDataset,
    ReproducibilityFramework,
    AcademicVisualization,
    MethodologyDocumenter
)

__all__ = [
    # Experimental algorithms
    "QuantumSpatialFusion",
    "AdaptiveQuantumPlanner", 
    "NovelEchoAttention",
    "HybridQuantumCNN",
    
    # Benchmarking
    "ResearchBenchmarkSuite",
    "AlgorithmComparator",
    "StatisticalValidator", 
    "PerformanceProfiler",
    
    # Comparative studies
    "BaselineComparator",
    "ExperimentalDesign",
    "ResultsAnalyzer",
    "SignificanceTester",
    
    # Publication support
    "PublicationDataset",
    "ReproducibilityFramework",
    "AcademicVisualization",
    "MethodologyDocumenter"
]