"""
Sentiment Analysis Module for EchoLoc-NN

Integrates sentiment analysis capabilities with spatial localization data.
Enables context-aware sentiment analysis based on location, movement patterns,
and multi-modal sensor fusion.

Core components:
- models: Transformer-based sentiment models with spatial awareness
- spatial_fusion: Integration with EchoLoc positioning data
- multi_modal: Combined audio-spatial-text sentiment analysis
- real_time: Streaming sentiment analysis with location context
"""

from .models import SpatialSentimentAnalyzer, MultiModalSentimentModel
from .spatial_fusion import LocationContextSentiment, MovementPatternAnalyzer
from .real_time import StreamingSentimentAnalyzer, SentimentLocationTracker
from .multi_modal import AudioSpatialSentiment, TextLocationSentiment

__all__ = [
    "SpatialSentimentAnalyzer",
    "MultiModalSentimentModel", 
    "LocationContextSentiment",
    "MovementPatternAnalyzer",
    "StreamingSentimentAnalyzer",
    "SentimentLocationTracker",
    "AudioSpatialSentiment",
    "TextLocationSentiment"
]