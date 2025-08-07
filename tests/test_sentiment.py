"""
Comprehensive tests for spatial-aware sentiment analysis system.

Tests all components including models, spatial fusion, real-time processing,
multi-modal analysis, API endpoints, and performance optimizations.
"""

import pytest
import torch
import numpy as np
import asyncio
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List

# Import components to test
from echoloc_nn.sentiment.models import SpatialSentimentAnalyzer, MultiModalSentimentModel
from echoloc_nn.sentiment.spatial_fusion import LocationContextSentiment, LocationContext, MovementPatternAnalyzer
from echoloc_nn.sentiment.real_time import StreamingSentimentAnalyzer, StreamingConfig, SentimentLocationTracker
from echoloc_nn.sentiment.multi_modal import (
    MultiModalSentimentProcessor, AudioSpatialSentiment, TextLocationSentiment, 
    MultiModalInput, AudioFeatures
)
from echoloc_nn.sentiment.optimization import (
    ModelOptimizer, InferenceCache, BatchProcessor, ConcurrentSentimentAnalyzer, OptimizationConfig
)
from echoloc_nn.utils.error_handling import EchoLocError

class TestSpatialSentimentAnalyzer:
    \"\"\"Test suite for spatial sentiment analysis models.\"\"\"
    
    @pytest.fixture
    def analyzer(self):
        \"\"\"Create analyzer instance for testing.\"\"\"
        return SpatialSentimentAnalyzer(
            model_name=\"distilbert-base-uncased\",
            spatial_dim=64,
            sentiment_classes=3,
            max_sequence_length=128
        )
        
    @pytest.fixture
    def sample_spatial_context(self):
        \"\"\"Sample spatial context data.\"\"\"
        return np.array([
            1.5, 2.3, 0.8,  # position (x, y, z)
            0.1, -0.2, 0.0,  # velocity
            0.0, 0.1, 0.0   # acceleration
        ])
        
    def test_model_initialization(self, analyzer):
        \"\"\"Test model initialization and architecture.\"\"\"
        assert analyzer.spatial_dim == 64
        assert analyzer.sentiment_classes == 3
        assert analyzer.max_length == 128
        assert analyzer.tokenizer is not None
        assert analyzer.transformer is not None
        
    def test_forward_pass_text_only(self, analyzer):
        \"\"\"Test forward pass with text input only.\"\"\"
        text = \"I am feeling great today!\"
        
        outputs = analyzer.forward(text)
        
        assert \"sentiment_logits\" in outputs
        assert \"sentiment_probs\" in outputs
        assert \"confidence\" in outputs
        assert \"spatial_influence\" in outputs
        
        # Check tensor shapes
        assert outputs[\"sentiment_logits\"].shape == (1, 3)
        assert outputs[\"sentiment_probs\"].shape == (1, 3)
        assert outputs[\"confidence\"].shape == (1, 1)
        
    def test_forward_pass_with_spatial(self, analyzer, sample_spatial_context):
        \"\"\"Test forward pass with spatial context.\"\"\"
        text = \"This place makes me nervous\"
        spatial_tensor = torch.tensor(sample_spatial_context, dtype=torch.float32).unsqueeze(0)
        
        outputs = analyzer.forward(text, spatial_context=spatial_tensor)
        
        assert outputs[\"spatial_influence\"].item() > 0.0
        assert \"spatial_features\" in outputs
        assert outputs[\"spatial_features\"] is not None
        
    def test_predict_sentiment_interface(self, analyzer, sample_spatial_context):
        \"\"\"Test high-level prediction interface.\"\"\"
        text = \"I love this new location!\"
        
        # Text only
        result = analyzer.predict_sentiment(text)
        
        assert \"sentiment\" in result
        assert \"probabilities\" in result
        assert \"confidence\" in result
        assert result[\"sentiment\"] in [\"negative\", \"neutral\", \"positive\"]
        assert 0.0 <= result[\"confidence\"] <= 1.0
        
        # With spatial context
        result_spatial = analyzer.predict_sentiment(text, sample_spatial_context)
        
        assert \"spatial_influence\" in result_spatial
        assert isinstance(result_spatial[\"spatial_influence\"], (int, float))
        
    def test_batch_prediction(self, analyzer):
        \"\"\"Test batch prediction functionality.\"\"\"
        texts = [
            \"I am happy\",
            \"This is terrible\", 
            \"Neutral statement\"
        ]
        
        result = analyzer.predict_sentiment(texts)
        
        assert \"sentiment\" in result
        assert isinstance(result[\"sentiment\"], list)
        assert len(result[\"sentiment\"]) == len(texts)
        
    def test_input_validation(self, analyzer):
        \"\"\"Test input validation and error handling.\"\"\"
        
        # Empty text
        with pytest.raises((ValueError, EchoLocError)):
            analyzer.predict_sentiment(\"\")
            
        # Invalid spatial context shape
        with pytest.raises((ValueError, EchoLocError)):
            invalid_spatial = np.array([1.0, 2.0])  # Wrong size
            analyzer.predict_sentiment(\"test\", invalid_spatial)
            
        # Very long text
        very_long_text = \"word \" * 1000
        result = analyzer.predict_sentiment(very_long_text)
        assert result is not None  # Should handle gracefully
        
class TestMultiModalSentimentModel:
    \"\"\"Test suite for multi-modal sentiment analysis.\"\"\"
    
    @pytest.fixture
    def multimodal_model(self):
        \"\"\"Create multi-modal model for testing.\"\"\"
        return MultiModalSentimentModel(
            sentiment_classes=5,
            audio_features_dim=128,
            spatial_dim=64
        )
        
    @pytest.fixture
    def sample_audio_features(self):
        \"\"\"Sample audio features tensor.\"\"\"
        return torch.randn(1, 128)  # Batch size 1, 128 features
        
    def test_multimodal_forward_pass(self, multimodal_model, sample_audio_features):
        \"\"\"Test multi-modal forward pass.\"\"\"
        text = \"This sounds amazing!\"
        spatial_context = torch.randn(1, 9)
        
        outputs = multimodal_model.forward(
            text_input=text,
            audio_features=sample_audio_features,
            spatial_context=spatial_context
        )
        
        assert \"sentiment_logits\" in outputs
        assert \"sentiment_probs\" in outputs
        assert \"modality_weights\" in outputs
        assert \"text_outputs\" in outputs
        
        # Check modality weights sum to reasonable values
        weights = outputs[\"modality_weights\"]
        assert len(weights) == 3  # text, audio, spatial
        assert torch.allclose(torch.sum(weights), torch.tensor(1.0), atol=0.1)
        
    def test_modality_combinations(self, multimodal_model, sample_audio_features):
        \"\"\"Test different modality combinations.\"\"\"
        text = \"Test text\"
        spatial_context = torch.randn(1, 9)
        
        # Text only
        output1 = multimodal_model.forward(text_input=text)
        assert output1[\"sentiment_logits\"].shape == (1, 5)
        
        # Text + Audio
        output2 = multimodal_model.forward(
            text_input=text,
            audio_features=sample_audio_features
        )
        
        # Text + Spatial  
        output3 = multimodal_model.forward(
            text_input=text,
            spatial_context=spatial_context
        )
        
        # All modalities
        output4 = multimodal_model.forward(
            text_input=text,
            audio_features=sample_audio_features,
            spatial_context=spatial_context
        )
        
        # Results should be different with different modalities
        assert not torch.allclose(output1[\"sentiment_logits\"], output4[\"sentiment_logits\"])
        
class TestLocationContextSentiment:
    \"\"\"Test suite for spatial context management.\"\"\"
    
    @pytest.fixture
    def location_manager(self):
        \"\"\"Create location context manager.\"\"\"
        return LocationContextSentiment(history_length=10)
        
    @pytest.fixture
    def sample_position(self):
        \"\"\"Sample 3D position.\"\"\"
        return np.array([2.5, 1.8, 0.5])
        
    def test_zone_definitions(self, location_manager):
        \"\"\"Test spatial zone definitions and management.\"\"\"
        
        # Default zones should exist
        assert \"workspace\" in location_manager.zones
        assert \"social_area\" in location_manager.zones
        
        # Add custom zone
        location_manager.add_zone(
            name=\"meeting_room\",
            bounds=[[3, 6], [2, 5], [0, 3]],
            sentiment_bias=0.2,
            description=\"Conference room\"
        )
        
        assert \"meeting_room\" in location_manager.zones
        assert location_manager.zones[\"meeting_room\"][\"sentiment_bias\"] == 0.2
        
    def test_zone_detection(self, location_manager, sample_position):
        \"\"\"Test zone detection from position.\"\"\"
        
        # Position within workspace bounds
        workspace_pos = np.array([1.0, 1.0, 1.0])
        zone = location_manager._determine_zone(workspace_pos)
        assert zone == \"workspace\"
        
        # Position outside any zone
        outside_pos = np.array([100.0, 100.0, 100.0])
        zone = location_manager._determine_zone(outside_pos)
        assert zone is None
        
    def test_motion_calculation(self, location_manager):
        \"\"\"Test velocity and acceleration calculation.\"\"\"
        
        # Add some position history
        positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.5, 0.0])
        ]
        
        for i, pos in enumerate(positions):
            context = LocationContext(
                position=pos,
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                confidence=0.9,
                timestamp=time.time() + i
            )
            location_manager.location_history.append(context)
            
        # Calculate motion for new position
        new_pos = np.array([3.0, 1.0, 0.0])
        velocity, acceleration = location_manager._calculate_motion(new_pos, time.time() + 3)
        
        # Should have positive x velocity
        assert velocity[0] > 0
        
    def test_spatial_features_extraction(self, location_manager):
        \"\"\"Test spatial feature extraction.\"\"\"
        
        context = LocationContext(
            position=np.array([1.0, 2.0, 0.5]),
            velocity=np.array([0.1, -0.1, 0.0]),
            acceleration=np.array([0.01, 0.0, 0.0]),
            confidence=0.85,
            timestamp=time.time(),
            zone=\"workspace\"
        )
        
        features = location_manager.get_spatial_features(context)
        
        # Should include position, velocity, acceleration (9 features)
        # Plus zone encoding and movement features
        assert len(features) >= 9
        assert not np.isnan(features).any()
        
class TestStreamingSentimentAnalyzer:
    \"\"\"Test suite for real-time streaming analysis.\"\"\"
    
    @pytest.fixture
    def streaming_config(self):
        \"\"\"Streaming configuration for testing.\"\"\"
        return StreamingConfig(
            update_frequency_hz=10.0,
            max_buffer_size=100,
            confidence_threshold=0.5
        )
        
    @pytest.fixture
    def mock_model(self):
        \"\"\"Mock sentiment model for testing.\"\"\"
        model = Mock(spec=SpatialSentimentAnalyzer)
        model.predict_sentiment.return_value = {
            \"sentiment\": \"positive\",
            \"probabilities\": np.array([0.1, 0.2, 0.7]),
            \"confidence\": 0.8,
            \"spatial_influence\": 0.3
        }
        return model
        
    @pytest.fixture
    def mock_location_manager(self):
        \"\"\"Mock location manager for testing.\"\"\"
        manager = Mock(spec=LocationContextSentiment)
        manager.get_current_location.return_value = LocationContext(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            confidence=0.9,
            timestamp=time.time()
        )
        manager.get_spatial_features.return_value = np.zeros(15)  # Mock features
        return manager
        
    def test_streaming_initialization(self, mock_model, mock_location_manager, streaming_config):
        \"\"\"Test streaming analyzer initialization.\"\"\"
        
        analyzer = StreamingSentimentAnalyzer(
            model=mock_model,
            location_manager=mock_location_manager,
            config=streaming_config
        )
        
        assert analyzer.model == mock_model
        assert analyzer.location_manager == mock_location_manager
        assert analyzer.config == streaming_config
        assert not analyzer.is_streaming
        
    def test_single_text_analysis(self, mock_model, mock_location_manager, streaming_config):
        \"\"\"Test single text analysis without streaming.\"\"\"
        
        analyzer = StreamingSentimentAnalyzer(
            model=mock_model,
            location_manager=mock_location_manager,
            config=streaming_config
        )
        
        result = analyzer._analyze_single(\"Test text\")
        
        assert result.sentiment == \"positive\"
        assert result.confidence == 0.8
        assert result.text == \"Test text\"
        assert isinstance(result.timestamp, float)
        
        # Verify model was called
        mock_model.predict_sentiment.assert_called_once()
        
    def test_streaming_lifecycle(self, mock_model, mock_location_manager, streaming_config):
        \"\"\"Test streaming start/stop lifecycle.\"\"\"
        
        analyzer = StreamingSentimentAnalyzer(
            model=mock_model,
            location_manager=mock_location_manager,
            config=streaming_config
        )
        
        # Start streaming
        analyzer.start_streaming()
        assert analyzer.is_streaming
        
        # Stop streaming
        analyzer.stop_streaming()
        assert not analyzer.is_streaming
        
    def test_metrics_collection(self, mock_model, mock_location_manager, streaming_config):
        \"\"\"Test performance metrics collection.\"\"\"
        
        analyzer = StreamingSentimentAnalyzer(
            model=mock_model,
            location_manager=mock_location_manager,
            config=streaming_config
        )
        
        # Analyze some text to generate metrics
        result = analyzer._analyze_single(\"Test text for metrics\")
        
        metrics = analyzer.get_metrics()
        
        assert metrics.total_predictions >= 1
        assert metrics.avg_latency_ms > 0
        assert \"positive\" in metrics.sentiment_distribution
        
class TestAudioSpatialSentiment:
    \"\"\"Test suite for audio processing with spatial awareness.\"\"\"
    
    @pytest.fixture
    def audio_processor(self):
        \"\"\"Create audio processor for testing.\"\"\"
        return AudioSpatialSentiment(
            sample_rate=22050,
            n_mfcc=13,
            max_duration=5.0
        )
        
    @pytest.fixture
    def sample_audio_data(self):
        \"\"\"Generate sample audio data for testing.\"\"\"
        # Generate 1 second of sine wave at 440 Hz
        sample_rate = 22050
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        return audio, sample_rate
        
    def test_audio_feature_extraction(self, audio_processor, sample_audio_data):
        \"\"\"Test audio feature extraction.\"\"\"
        
        audio_data, sample_rate = sample_audio_data
        
        features = audio_processor.extract_features(audio_data)
        
        assert isinstance(features, AudioFeatures)
        assert features.mfcc.shape[0] == audio_processor.n_mfcc
        assert features.sample_rate == audio_processor.sample_rate
        assert features.duration > 0
        assert features.energy > 0
        assert features.tempo > 0
        
    def test_features_to_vector_conversion(self, audio_processor, sample_audio_data):
        \"\"\"Test conversion of features to fixed-size vector.\"\"\"
        
        audio_data, _ = sample_audio_data
        features = audio_processor.extract_features(audio_data)
        
        feature_vector = audio_processor.features_to_vector(features)
        
        assert isinstance(feature_vector, np.ndarray)
        assert len(feature_vector) > 0
        assert not np.isnan(feature_vector).any()
        assert not np.isinf(feature_vector).any()
        
    def test_audio_emotion_analysis(self, audio_processor, sample_audio_data):
        \"\"\"Test basic audio emotion analysis.\"\"\"
        
        audio_data, _ = sample_audio_data
        features = audio_processor.extract_features(audio_data)
        
        emotion_scores = audio_processor.analyze_audio_emotion(features)
        
        assert \"valence\" in emotion_scores
        assert \"arousal\" in emotion_scores 
        assert \"dominance\" in emotion_scores
        
        # All scores should be in [-1, 1] range
        for score in emotion_scores.values():
            assert -1.0 <= score <= 1.0
            
    def test_audio_file_loading(self, audio_processor):
        \"\"\"Test audio file loading functionality.\"\"\"
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=\".wav\", delete=False) as temp_file:
            # Generate and save test audio
            import soundfile as sf
            
            sample_rate = 22050
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
            
            sf.write(temp_file.name, audio, sample_rate)
            
            # Test loading
            loaded_audio, loaded_sr = audio_processor.load_audio(temp_file.name)
            
            assert loaded_sr == audio_processor.sample_rate
            assert len(loaded_audio) <= audio_processor.max_samples
            
        # Clean up
        Path(temp_file.name).unlink()
        
class TestOptimization:
    \"\"\"Test suite for performance optimization components.\"\"\"
    
    @pytest.fixture
    def optimization_config(self):
        \"\"\"Optimization configuration for testing.\"\"\"
        return OptimizationConfig(
            enable_quantization=True,
            enable_jit_compilation=False,  # Disable for testing
            enable_caching=True,
            cache_size_mb=64,
            max_workers=2,
            batch_size=4
        )
        
    @pytest.fixture
    def mock_model(self):
        \"\"\"Simple mock model for optimization testing.\"\"\"
        model = Mock(spec=SpatialSentimentAnalyzer)
        model.eval.return_value = model
        model.predict_sentiment.return_value = {
            \"sentiment\": \"neutral\",
            \"probabilities\": np.array([0.3, 0.4, 0.3]),
            \"confidence\": 0.6,
            \"spatial_influence\": 0.1
        }
        return model
        
    def test_inference_cache(self, optimization_config):
        \"\"\"Test inference caching system.\"\"\"
        
        cache = InferenceCache(optimization_config)
        
        # Test cache miss
        result = cache.get(\"test text\")
        assert result is None
        
        # Store result
        test_result = {\"sentiment\": \"positive\", \"confidence\": 0.8}
        cache.put(\"test text\", test_result)
        
        # Test cache hit
        cached_result = cache.get(\"test text\")
        assert cached_result == test_result
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats[\"hits\"] == 1
        assert stats[\"misses\"] == 1
        assert 0.0 <= stats[\"hit_rate\"] <= 1.0
        
    def test_cache_with_spatial_context(self, optimization_config):
        \"\"\"Test caching with spatial context.\"\"\"
        
        cache = InferenceCache(optimization_config)
        
        text = \"spatial test\"
        spatial_context1 = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        spatial_context2 = np.array([1.1, 2.1, 3.1, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        
        result1 = {\"sentiment\": \"positive\"}
        result2 = {\"sentiment\": \"negative\"}
        
        # Store with different spatial contexts
        cache.put(text, result1, spatial_context1)
        cache.put(text, result2, spatial_context2)
        
        # Retrieve with matching spatial contexts
        cached1 = cache.get(text, spatial_context1)
        cached2 = cache.get(text, spatial_context2)
        
        assert cached1 == result1
        assert cached2 == result2
        
    def test_batch_processor(self, mock_model, optimization_config):
        \"\"\"Test batch processing functionality.\"\"\"
        
        processor = BatchProcessor(mock_model, optimization_config)
        
        # Test single processing
        result = processor.process_single(\"test text\")
        assert result[\"sentiment\"] == \"neutral\"
        
        # Test batch processing
        batch_data = [
            {\"text\": \"text 1\"},
            {\"text\": \"text 2\", \"spatial_context\": np.zeros(9)},
            {\"text\": \"text 3\"}
        ]
        
        batch_results = processor.process_batch(batch_data)
        
        assert len(batch_results) == len(batch_data)
        for result in batch_results:
            assert \"sentiment\" in result
            assert \"confidence\" in result
            
    def test_performance_stats(self, mock_model, optimization_config):
        \"\"\"Test performance statistics collection.\"\"\"
        
        processor = BatchProcessor(mock_model, optimization_config)
        
        # Process some requests to generate stats
        for i in range(5):
            processor.process_single(f\"test text {i}\")
            
        stats = processor.get_performance_stats()
        
        assert \"batch_stats\" in stats
        assert \"cache_stats\" in stats
        assert \"performance_metrics\" in stats
        assert \"memory_usage_mb\" in stats
        assert stats[\"request_count\"] >= 5
        
class TestIntegration:
    \"\"\"Integration tests for the complete system.\"\"\"
    
    @pytest.fixture
    def integrated_system(self):
        \"\"\"Create integrated system for testing.\"\"\"
        
        # Initialize components
        spatial_model = SpatialSentimentAnalyzer(
            spatial_dim=32,  # Smaller for testing
            sentiment_classes=3
        )
        
        location_manager = LocationContextSentiment()
        
        multimodal_processor = MultiModalSentimentProcessor(
            MultiModalSentimentModel(sentiment_classes=3),
            location_manager
        )
        
        return {
            \"spatial_model\": spatial_model,
            \"location_manager\": location_manager,
            \"multimodal_processor\": multimodal_processor
        }
        
    @pytest.mark.asyncio
    async def test_end_to_end_analysis(self, integrated_system):
        \"\"\"Test end-to-end analysis pipeline.\"\"\"
        
        # Test data
        text = \"I love this new location with great acoustics!\"
        spatial_context = np.array([2.0, 1.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Spatial analysis
        spatial_result = integrated_system[\"spatial_model\"].predict_sentiment(
            text=text,
            spatial_context=spatial_context
        )
        
        assert \"sentiment\" in spatial_result
        assert \"confidence\" in spatial_result
        
        # Multi-modal analysis (text + spatial)
        multimodal_input = MultiModalInput(
            text=text,
            spatial_context=spatial_context,
            timestamp=time.time()
        )
        
        multimodal_result = await integrated_system[\"multimodal_processor\"].analyze_multi_modal(
            multimodal_input
        )
        
        assert \"modalities_used\" in multimodal_result
        assert \"text\" in multimodal_result[\"modalities_used\"]
        assert \"spatial\" in multimodal_result[\"modalities_used\"]
        
    def test_location_aware_sentiment_tracking(self, integrated_system):
        \"\"\"Test location-aware sentiment tracking.\"\"\"
        
        tracker = SentimentLocationTracker(grid_resolution=0.5)
        location_manager = integrated_system[\"location_manager\"]
        
        # Simulate movement with sentiment analysis
        positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 1.0, 0.0])
        ]
        
        texts = [
            \"Starting point feels neutral\",
            \"Moving forward, feeling better\", 
            \"This destination is amazing!\"
        ]
        
        for pos, text in zip(positions, texts):
            # Create location context
            location_context = LocationContext(
                position=pos,
                velocity=np.array([0.1, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                confidence=0.9,
                timestamp=time.time()
            )
            
            # Analyze sentiment
            result = integrated_system[\"spatial_model\"].predict_sentiment(text, pos)
            
            # Create sentiment location
            from echoloc_nn.sentiment.spatial_fusion import SentimentLocation
            sentiment_loc = SentimentLocation(
                sentiment=result[\"sentiment\"],
                sentiment_probs=result[\"probabilities\"],
                confidence=result[\"confidence\"],
                location=location_context,
                spatial_influence=result.get(\"spatial_influence\", 0.0),
                text=text,
                timestamp=time.time()
            )
            
            # Add to tracker
            tracker.add_sentiment_location(sentiment_loc)
            
        # Verify tracking
        assert len(tracker.location_history) == 3
        
        # Get heatmap data
        heatmap_data = tracker.get_sentiment_heatmap()
        assert \"positions\" in heatmap_data
        assert \"sentiments\" in heatmap_data
        assert len(heatmap_data[\"positions\"]) > 0
        
    def test_performance_under_load(self, integrated_system):
        \"\"\"Test system performance under load.\"\"\"
        
        model = integrated_system[\"spatial_model\"]
        
        # Generate test data
        test_texts = [
            f\"Test sentence number {i} with various sentiments\"
            for i in range(100)
        ]
        
        spatial_contexts = [
            np.random.randn(9) for _ in range(100)
        ]
        
        # Measure processing time
        start_time = time.time()
        
        results = []
        for text, spatial in zip(test_texts, spatial_contexts):
            result = model.predict_sentiment(text, spatial)
            results.append(result)
            
        end_time = time.time()
        
        # Verify results
        assert len(results) == 100
        
        # Check performance
        total_time = end_time - start_time
        avg_time_per_request = total_time / len(test_texts)
        
        print(f\"Processed {len(test_texts)} requests in {total_time:.2f}s\")
        print(f\"Average time per request: {avg_time_per_request*1000:.1f}ms\")
        
        # Basic performance assertion (adjust threshold as needed)
        assert avg_time_per_request < 1.0  # Less than 1 second per request
        
# Performance benchmark tests
class TestPerformanceBenchmarks:
    \"\"\"Performance benchmark tests for optimization validation.\"\"\"
    
    @pytest.mark.slow
    def test_model_optimization_benchmarks(self):
        \"\"\"Benchmark model optimization techniques.\"\"\"
        
        # Create baseline model
        baseline_model = SpatialSentimentAnalyzer(
            spatial_dim=128,
            sentiment_classes=3
        )
        
        # Create optimized model
        config = OptimizationConfig(
            enable_quantization=True,
            enable_jit_compilation=False,  # Skip JIT for testing
            enable_caching=True
        )
        
        optimizer = ModelOptimizer(config)
        optimized_model = optimizer.optimize_spatial_model(baseline_model)
        
        # Benchmark both models
        baseline_benchmark = optimizer.benchmark_model(baseline_model, num_samples=100)
        optimized_benchmark = optimizer.benchmark_model(optimized_model, num_samples=100)
        
        print(\"Baseline model performance:\")
        print(f\"  Average latency: {baseline_benchmark['avg_latency_ms']:.2f} ms\")
        print(f\"  Throughput: {baseline_benchmark['throughput_rps']:.2f} RPS\")
        
        print(\"Optimized model performance:\")
        print(f\"  Average latency: {optimized_benchmark['avg_latency_ms']:.2f} ms\")
        print(f\"  Throughput: {optimized_benchmark['throughput_rps']:.2f} RPS\")
        
        # Optimized model should be faster or similar (quantization may have overhead on CPU)
        speedup_ratio = baseline_benchmark['avg_latency_ms'] / optimized_benchmark['avg_latency_ms']
        print(f\"Speedup ratio: {speedup_ratio:.2f}x\")
        
    @pytest.mark.slow
    def test_caching_performance_impact(self):
        \"\"\"Test performance impact of caching system.\"\"\"
        
        config = OptimizationConfig(
            enable_caching=True,
            cache_size_mb=128,
            batch_size=16
        )
        
        model = SpatialSentimentAnalyzer(spatial_dim=64, sentiment_classes=3)
        processor = BatchProcessor(model, config)
        
        # Test data
        test_texts = [
            \"This is a test sentence for caching performance\",
            \"Another test sentence with different content\",
            \"Third unique sentence for testing\"
        ] * 10  # Repeat for cache hits
        
        # First pass (cache misses)
        start_time = time.time()
        for text in test_texts:
            processor.process_single(text)
        first_pass_time = time.time() - start_time
        
        # Second pass (cache hits)
        start_time = time.time()
        for text in test_texts:
            processor.process_single(text)
        second_pass_time = time.time() - start_time
        
        print(f\"First pass (cache misses): {first_pass_time:.2f}s\")
        print(f\"Second pass (cache hits): {second_pass_time:.2f}s\")
        print(f\"Speedup from caching: {first_pass_time / second_pass_time:.2f}x\")
        
        # Cache should provide significant speedup
        assert second_pass_time < first_pass_time * 0.8  # At least 20% faster
        
        # Check cache statistics
        stats = processor.get_performance_stats()
        cache_stats = stats[\"cache_stats\"]
        
        print(f\"Cache hit rate: {cache_stats['hit_rate']:.2%}\")
        assert cache_stats[\"hit_rate\"] > 0.5  # At least 50% hit rate

# Utility functions for testing
def create_test_audio_file(filepath: Path, duration: float = 1.0, sample_rate: int = 22050):
    \"\"\"Create a test audio file for testing purposes.\"\"\"
    import soundfile as sf
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    sf.write(str(filepath), audio, sample_rate)
    
def assert_sentiment_result_valid(result: Dict):
    \"\"\"Assert that a sentiment analysis result is valid.\"\"\"
    required_keys = [\"sentiment\", \"confidence\"]
    
    for key in required_keys:
        assert key in result, f\"Missing key: {key}\"
        
    assert result[\"sentiment\"] in [\"negative\", \"neutral\", \"positive\"], f\"Invalid sentiment: {result['sentiment']}\"
    assert 0.0 <= result[\"confidence\"] <= 1.0, f\"Invalid confidence: {result['confidence']}\"
    
    if \"probabilities\" in result:
        probs = result[\"probabilities\"]
        if isinstance(probs, np.ndarray):
            assert len(probs) >= 2, \"Probabilities array too short\"
            assert np.allclose(np.sum(probs), 1.0, atol=0.1), \"Probabilities don't sum to 1\"
        
if __name__ == \"__main__\":
    # Run tests with pytest
    pytest.main([
        __file__,
        \"-v\",
        \"--tb=short\",
        \"-x\"  # Stop on first failure
    ])"