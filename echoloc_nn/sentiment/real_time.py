"""
Real-time streaming sentiment analysis with spatial context.

Provides continuous sentiment monitoring integrated with EchoLoc 
ultrasonic positioning for location-aware emotional analytics.
"""

import torch
import numpy as np
import asyncio
import time
from typing import Dict, List, Optional, Callable, Union, AsyncIterator
from dataclasses import dataclass, field
from collections import deque
import threading
import json
from queue import Queue, Empty

from .models import SpatialSentimentAnalyzer, MultiModalSentimentModel
from .spatial_fusion import LocationContextSentiment, SentimentLocation, LocationContext
from ..hardware.ultrasonic_array import UltrasonicArray
from ..inference.locator import EchoLocator
from ..utils.error_handling import handle_errors, EchoLocError
from ..utils.monitoring import PerformanceMonitor

@dataclass
class StreamingConfig:
    \"\"\"Configuration for real-time sentiment analysis.\"\"\"
    update_frequency_hz: float = 10.0
    batch_size: int = 1
    max_buffer_size: int = 1000
    confidence_threshold: float = 0.7
    spatial_update_rate_hz: float = 20.0
    enable_audio: bool = False
    enable_movement_analysis: bool = True
    
    # Performance settings
    max_latency_ms: float = 100.0
    enable_gpu_acceleration: bool = True
    
    # Output settings
    save_to_file: bool = False
    output_file: Optional[str] = None
    enable_callbacks: bool = True

@dataclass
class StreamingMetrics:
    \"\"\"Performance metrics for streaming analysis.\"\"\"
    total_predictions: int = 0
    avg_latency_ms: float = 0.0
    predictions_per_second: float = 0.0
    spatial_updates_per_second: float = 0.0
    confidence_distribution: Dict[str, float] = field(default_factory=dict)
    sentiment_distribution: Dict[str, float] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
class StreamingSentimentAnalyzer:
    \"\"\"Real-time sentiment analysis with spatial context.\"\"\"
    
    def __init__(
        self,
        model: Union[SpatialSentimentAnalyzer, MultiModalSentimentModel],
        location_manager: LocationContextSentiment,
        config: StreamingConfig = StreamingConfig(),
        ultrasonic_array: Optional[UltrasonicArray] = None
    ):
        self.model = model
        self.location_manager = location_manager
        self.config = config
        self.array = ultrasonic_array
        
        # Threading and streaming state
        self.is_streaming = False
        self.text_queue = Queue(maxsize=config.max_buffer_size)
        self.result_queue = Queue(maxsize=config.max_buffer_size)
        self.spatial_queue = Queue(maxsize=config.max_buffer_size)
        
        # Performance monitoring
        self.metrics = StreamingMetrics()
        self.performance_monitor = PerformanceMonitor(\"StreamingSentiment\")
        
        # Callbacks
        self.callbacks = []
        
        # Buffer for batch processing
        self.text_buffer = []
        self.spatial_buffer = []
        
        # Current location context
        self.current_location = None
        
    def add_callback(self, callback: Callable[[SentimentLocation], None]):
        \"\"\"Add callback function for processing results.\"\"\"
        self.callbacks.append(callback)
        
    @handle_errors
    def start_streaming(self):
        \"\"\"Start real-time sentiment analysis streaming.\"\"\"
        
        if self.is_streaming:
            raise EchoLocError(\"Streaming already active\")
            
        self.is_streaming = True
        self.metrics = StreamingMetrics()
        
        # Start worker threads
        self.spatial_thread = threading.Thread(target=self._spatial_worker, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        
        self.spatial_thread.start()
        self.processing_thread.start()
        
        print(f\"✓ Streaming sentiment analysis started at {self.config.update_frequency_hz} Hz\")
        
    def stop_streaming(self):
        \"\"\"Stop real-time streaming.\"\"\"
        self.is_streaming = False
        
        # Wait for threads to finish
        if hasattr(self, 'spatial_thread'):
            self.spatial_thread.join(timeout=2.0)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
            
        print(\"✓ Streaming stopped\")
        
    def analyze_text(self, text: str, priority: bool = False) -> Optional[SentimentLocation]:
        \"\"\"Analyze single text with current spatial context.\"\"\"
        
        if not self.is_streaming:
            # Direct analysis mode
            return self._analyze_single(text)
            
        # Queue for streaming analysis
        try:
            if priority:
                # Insert at front of queue for priority processing
                temp_queue = Queue()
                temp_queue.put((text, time.time()))
                
                # Move existing items to temp queue
                while not self.text_queue.empty():
                    try:
                        temp_queue.put(self.text_queue.get_nowait())
                    except Empty:
                        break
                        
                # Refill original queue with priority item first
                self.text_queue = Queue(maxsize=self.config.max_buffer_size)
                while not temp_queue.empty():
                    try:
                        self.text_queue.put(temp_queue.get_nowait())
                    except Empty:
                        break
            else:
                self.text_queue.put((text, time.time()))
                
        except:
            # Queue full, analyze directly
            return self._analyze_single(text)
            
        return None
        
    def _analyze_single(self, text: str) -> SentimentLocation:
        \"\"\"Analyze single text with current location context.\"\"\"
        
        start_time = time.time()
        
        # Get current location
        if self.current_location is None:
            self.current_location = self.location_manager.get_current_location()
            
        # Extract spatial features
        spatial_features = self.location_manager.get_spatial_features(self.current_location)
        
        # Pad or truncate to expected size (9 base features)
        if len(spatial_features) > 9:
            spatial_input = spatial_features[:9]
        else:
            spatial_input = np.pad(spatial_features, (0, 9 - len(spatial_features)))
            
        # Run sentiment analysis
        with self.performance_monitor.measure(\"sentiment_inference\"):
            result = self.model.predict_sentiment(
                text=text,
                spatial_context=spatial_input
            )
            
        # Create result object
        sentiment_location = SentimentLocation(
            sentiment=result[\"sentiment\"],
            sentiment_probs=result[\"probabilities\"],
            confidence=float(result[\"confidence\"]),
            location=self.current_location,
            spatial_influence=float(result.get(\"spatial_influence\", 0.0)),
            text=text,
            timestamp=time.time()
        )
        
        # Update metrics
        latency_ms = (time.time() - start_time) * 1000
        self._update_metrics(sentiment_location, latency_ms)
        
        # Execute callbacks
        if self.config.enable_callbacks:
            for callback in self.callbacks:
                try:
                    callback(sentiment_location)
                except Exception as e:
                    print(f\"Callback error: {e}\")
                    
        return sentiment_location
        
    def _spatial_worker(self):
        \"\"\"Background worker for spatial context updates.\"\"\"
        
        update_interval = 1.0 / self.config.spatial_update_rate_hz
        
        while self.is_streaming:
            try:
                start_time = time.time()
                
                # Update location context
                if self.array is not None:
                    # Get fresh echo data
                    try:
                        echo_data = self.array.get_latest_echo(timeout=0.1)
                        self.current_location = self.location_manager.get_current_location(echo_data)
                    except:
                        # Fallback to last known location
                        self.current_location = self.location_manager.get_current_location()
                else:
                    self.current_location = self.location_manager.get_current_location()
                    
                # Sleep to maintain update rate
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f\"Spatial worker error: {e}\")
                time.sleep(0.1)
                
    def _processing_worker(self):
        \"\"\"Background worker for sentiment processing.\"\"\"
        
        update_interval = 1.0 / self.config.update_frequency_hz
        
        while self.is_streaming:
            try:
                start_time = time.time()
                
                # Collect batch of texts
                texts_to_process = []
                while len(texts_to_process) < self.config.batch_size:
                    try:
                        text_item = self.text_queue.get(timeout=0.1)
                        texts_to_process.append(text_item)
                    except Empty:
                        break
                        
                # Process batch if we have items
                if texts_to_process:
                    self._process_batch(texts_to_process)
                    
                # Sleep to maintain processing rate
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f\"Processing worker error: {e}\")
                time.sleep(0.1)
                
    def _process_batch(self, text_items: List[tuple]):
        \"\"\"Process batch of text items.\"\"\"
        
        for text, timestamp in text_items:
            try:
                result = self._analyze_single(text)
                
                # Add to result queue
                try:
                    self.result_queue.put(result, timeout=0.1)
                except:
                    pass  # Queue full, drop result
                    
            except Exception as e:
                print(f\"Batch processing error: {e}\")
                
    def _update_metrics(self, result: SentimentLocation, latency_ms: float):
        \"\"\"Update performance metrics.\"\"\"
        
        self.metrics.total_predictions += 1
        
        # Update average latency
        n = self.metrics.total_predictions
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * (n - 1) + latency_ms) / n
        )
        
        # Update predictions per second
        elapsed_time = time.time() - self.metrics.start_time
        if elapsed_time > 0:
            self.metrics.predictions_per_second = self.metrics.total_predictions / elapsed_time
            
        # Update distributions
        sentiment = result.sentiment
        if sentiment in self.metrics.sentiment_distribution:
            self.metrics.sentiment_distribution[sentiment] += 1
        else:
            self.metrics.sentiment_distribution[sentiment] = 1
            
        # Confidence bins
        conf_bin = f\"{int(result.confidence * 10) / 10:.1f}\"
        if conf_bin in self.metrics.confidence_distribution:
            self.metrics.confidence_distribution[conf_bin] += 1
        else:
            self.metrics.confidence_distribution[conf_bin] = 1
            
    def get_metrics(self) -> StreamingMetrics:
        \"\"\"Get current performance metrics.\"\"\"
        return self.metrics
        
    def get_results(self, max_results: int = 10) -> List[SentimentLocation]:
        \"\"\"Get recent analysis results.\"\"\"
        
        results = []
        for _ in range(min(max_results, self.result_queue.qsize())):
            try:
                results.append(self.result_queue.get_nowait())
            except Empty:
                break
                
        return results
        
class SentimentLocationTracker:
    \"\"\"Tracks sentiment patterns across spatial locations.\"\"\"
    
    def __init__(self, grid_resolution: float = 0.5):
        self.grid_resolution = grid_resolution  # meters
        self.sentiment_grid = {}  # (grid_x, grid_y) -> sentiment_data
        self.location_history = deque(maxlen=1000)
        
    def add_sentiment_location(self, sentiment_loc: SentimentLocation):
        \"\"\"Add sentiment-location data point.\"\"\"
        
        # Convert to grid coordinates
        pos = sentiment_loc.location.position
        grid_x = int(pos[0] / self.grid_resolution)
        grid_y = int(pos[1] / self.grid_resolution)
        grid_key = (grid_x, grid_y)
        
        # Initialize grid cell if needed
        if grid_key not in self.sentiment_grid:
            self.sentiment_grid[grid_key] = {
                \"sentiments\": [],
                \"confidences\": [],
                \"timestamps\": [],
                \"avg_sentiment\": 0.0,
                \"sentiment_variance\": 0.0
            }
            
        # Add data
        cell_data = self.sentiment_grid[grid_key]
        sentiment_score = self._sentiment_to_score(sentiment_loc.sentiment)
        
        cell_data[\"sentiments\"].append(sentiment_score)
        cell_data[\"confidences\"].append(sentiment_loc.confidence)
        cell_data[\"timestamps\"].append(sentiment_loc.timestamp)
        
        # Update statistics
        sentiments = cell_data[\"sentiments\"]
        cell_data[\"avg_sentiment\"] = np.mean(sentiments)
        cell_data[\"sentiment_variance\"] = np.var(sentiments) if len(sentiments) > 1 else 0.0
        
        # Add to history
        self.location_history.append(sentiment_loc)
        
    def _sentiment_to_score(self, sentiment: str) -> float:
        \"\"\"Convert sentiment label to numerical score.\"\"\"
        mapping = {
            \"very_negative\": -2.0,
            \"negative\": -1.0,
            \"neutral\": 0.0,
            \"positive\": 1.0,
            \"very_positive\": 2.0
        }
        return mapping.get(sentiment, 0.0)
        
    def get_location_sentiment(self, position: np.ndarray) -> Dict[str, float]:
        \"\"\"Get sentiment statistics for a specific location.\"\"\"
        
        grid_x = int(position[0] / self.grid_resolution)
        grid_y = int(position[1] / self.grid_resolution)
        grid_key = (grid_x, grid_y)
        
        if grid_key in self.sentiment_grid:
            return self.sentiment_grid[grid_key]
        else:
            return {
                \"avg_sentiment\": 0.0,
                \"sentiment_variance\": 0.0,
                \"data_points\": 0
            }
            
    def get_sentiment_heatmap(self) -> Dict[str, np.ndarray]:
        \"\"\"Generate sentiment heatmap data.\"\"\"
        
        if not self.sentiment_grid:
            return {\"positions\": np.array([]), \"sentiments\": np.array([])}
            
        positions = []
        sentiments = []
        
        for (grid_x, grid_y), data in self.sentiment_grid.items():
            # Convert back to world coordinates
            world_x = grid_x * self.grid_resolution
            world_y = grid_y * self.grid_resolution
            
            positions.append([world_x, world_y])
            sentiments.append(data[\"avg_sentiment\"])
            
        return {
            \"positions\": np.array(positions),
            \"sentiments\": np.array(sentiments),
            \"grid_resolution\": self.grid_resolution
        }"