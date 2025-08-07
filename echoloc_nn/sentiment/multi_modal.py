"""
Multi-modal sentiment analysis combining text, audio, and spatial data.

Integrates multiple sensory inputs with EchoLoc spatial positioning for
comprehensive emotional understanding in real-world environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import librosa
import soundfile as sf
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .models import MultiModalSentimentModel, SpatialSentimentAnalyzer
from .spatial_fusion import LocationContextSentiment, LocationContext
from ..utils.error_handling import handle_errors, EchoLocError
from ..utils.validation import validate_input

@dataclass
class AudioFeatures:
    \"\"\"Audio feature representation for sentiment analysis.\"\"\"
    mfcc: np.ndarray  # Mel-frequency cepstral coefficients
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    chroma: np.ndarray
    mel_spectrogram: np.ndarray
    tempo: float
    energy: float
    duration: float
    sample_rate: int = 22050

@dataclass 
class MultiModalInput:
    \"\"\"Input data for multi-modal sentiment analysis.\"\"\"
    text: Optional[str] = None
    audio_data: Optional[np.ndarray] = None
    audio_file_path: Optional[str] = None
    spatial_context: Optional[np.ndarray] = None
    location_context: Optional[LocationContext] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AudioSpatialSentiment:
    \"\"\"Audio processing with spatial awareness for sentiment analysis.\"\"\"
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 13,
        n_mels: int = 128,
        hop_length: int = 512,
        max_duration: float = 30.0  # seconds
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.max_samples = int(sample_rate * max_duration)
        
        # Audio processing parameters
        self.window_size = 2048
        self.feature_dims = {
            \"mfcc\": n_mfcc,
            \"spectral_centroid\": 1,
            \"spectral_rolloff\": 1, 
            \"zero_crossing_rate\": 1,
            \"chroma\": 12,
            \"tempo\": 1,
            \"energy\": 1
        }
        
    @handle_errors
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        \"\"\"Load audio file and resample if necessary.\"\"\"
        
        try:
            audio_data, orig_sr = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # Resample if needed
            if orig_sr != self.sample_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=orig_sr, 
                    target_sr=self.sample_rate
                )
                
            # Truncate or pad to max duration
            if len(audio_data) > self.max_samples:
                audio_data = audio_data[:self.max_samples]
            else:
                audio_data = np.pad(audio_data, (0, self.max_samples - len(audio_data)))
                
            return audio_data, self.sample_rate
            
        except Exception as e:
            raise EchoLocError(f\"Failed to load audio file {file_path}: {e}\")
            
    @handle_errors
    def extract_features(self, audio_data: np.ndarray, spatial_context: Optional[np.ndarray] = None) -> AudioFeatures:
        \"\"\"Extract comprehensive audio features for sentiment analysis.\"\"\"
        
        validate_input(audio_data, \"audio_data\", expected_type=np.ndarray)
        
        # Ensure audio is not empty
        if len(audio_data) == 0:
            raise EchoLocError(\"Empty audio data\")
            
        # Basic audio properties
        duration = len(audio_data) / self.sample_rate
        energy = np.sum(audio_data ** 2) / len(audio_data)
        
        # MFCC features (most important for speech emotion)
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Zero crossing rate (indicates voice vs music)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio_data,
            hop_length=self.hop_length
        )
        
        # Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return AudioFeatures(
            mfcc=mfcc,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=zero_crossing_rate,
            chroma=chroma,
            mel_spectrogram=mel_spectrogram,
            tempo=float(tempo),
            energy=float(energy),
            duration=duration,
            sample_rate=self.sample_rate
        )
        
    def features_to_vector(self, features: AudioFeatures) -> np.ndarray:
        \"\"\"Convert audio features to fixed-size vector for neural networks.\"\"\"
        
        # Statistical summaries of time-varying features
        feature_vectors = []
        
        # MFCC statistics
        mfcc_stats = np.concatenate([
            np.mean(features.mfcc, axis=1),
            np.std(features.mfcc, axis=1),
            np.max(features.mfcc, axis=1),
            np.min(features.mfcc, axis=1)
        ])
        feature_vectors.append(mfcc_stats)
        
        # Spectral feature statistics
        spectral_stats = np.concatenate([
            np.mean(features.spectral_centroid),
            np.std(features.spectral_centroid),
            np.mean(features.spectral_rolloff),
            np.std(features.spectral_rolloff),
            np.mean(features.zero_crossing_rate),
            np.std(features.zero_crossing_rate)
        ])
        feature_vectors.append(spectral_stats.flatten())
        
        # Chroma statistics
        chroma_stats = np.concatenate([
            np.mean(features.chroma, axis=1),
            np.std(features.chroma, axis=1)
        ])
        feature_vectors.append(chroma_stats)
        
        # Scalar features
        scalar_features = np.array([
            features.tempo,
            features.energy,
            features.duration
        ])
        feature_vectors.append(scalar_features)
        
        # Combine all features
        combined_vector = np.concatenate(feature_vectors)
        
        return combined_vector
        
    def analyze_audio_emotion(self, audio_features: AudioFeatures) -> Dict[str, float]:
        \"\"\"Analyze emotional content of audio features.\"\"\"
        
        # Simple rule-based emotion analysis
        # In practice, this would use a trained model
        
        emotion_scores = {
            \"valence\": 0.0,  # positive/negative
            \"arousal\": 0.0,  # excited/calm
            \"dominance\": 0.0  # confident/submissive
        }
        
        # Tempo-based arousal
        if audio_features.tempo > 120:
            emotion_scores[\"arousal\"] += 0.3
        elif audio_features.tempo < 80:
            emotion_scores[\"arousal\"] -= 0.2
            
        # Energy-based arousal
        if audio_features.energy > 0.1:
            emotion_scores[\"arousal\"] += 0.2
            emotion_scores[\"dominance\"] += 0.1
            
        # Spectral features for valence
        spectral_mean = np.mean(audio_features.spectral_centroid)
        if spectral_mean > 2000:  # Higher frequencies often indicate positive emotion
            emotion_scores[\"valence\"] += 0.2
            
        # MFCC variance for arousal (more variation = more excited)
        mfcc_var = np.var(audio_features.mfcc)
        if mfcc_var > 100:
            emotion_scores[\"arousal\"] += 0.1
            
        # Normalize to [-1, 1] range
        for key in emotion_scores:
            emotion_scores[key] = np.clip(emotion_scores[key], -1.0, 1.0)
            
        return emotion_scores
        
class TextLocationSentiment:
    \"\"\"Text sentiment analysis with detailed location context.\"\"\"
    
    def __init__(
        self,
        sentiment_model: SpatialSentimentAnalyzer,
        location_manager: LocationContextSentiment
    ):
        self.sentiment_model = sentiment_model
        self.location_manager = location_manager
        
    @handle_errors
    def analyze_with_context(
        self, 
        text: str, 
        location_context: Optional[LocationContext] = None
    ) -> Dict[str, Union[str, float, np.ndarray]]:
        \"\"\"Analyze text sentiment with rich location context.\"\"\"
        
        # Get location context if not provided
        if location_context is None:
            location_context = self.location_manager.get_current_location()
            
        # Extract spatial features
        spatial_features = self.location_manager.get_spatial_features(location_context)
        
        # Analyze sentiment
        result = self.sentiment_model.predict_sentiment(
            text=text,
            spatial_context=spatial_features[:9]  # Use first 9 features
        )
        
        # Add location context information
        result[\"location\"] = {
            \"position\": location_context.position.tolist(),
            \"zone\": location_context.zone,
            \"confidence\": location_context.confidence,
            \"movement_pattern\": self._analyze_movement_pattern(location_context)
        }
        
        # Add zone-based sentiment bias
        zone_bias = self.location_manager.get_zone_sentiment_bias(location_context.zone)
        result[\"zone_sentiment_bias\"] = zone_bias
        
        return result
        
    def _analyze_movement_pattern(self, location_context: LocationContext) -> Dict[str, float]:
        \"\"\"Analyze movement pattern for emotional context.\"\"\"
        
        speed = np.linalg.norm(location_context.velocity)
        acceleration_mag = np.linalg.norm(location_context.acceleration)
        
        return {
            \"speed_ms\": float(speed),
            \"acceleration_ms2\": float(acceleration_mag),
            \"movement_type\": self._classify_movement(speed),
            \"agitation_level\": min(1.0, acceleration_mag / 2.0)  # normalized
        }
        
    def _classify_movement(self, speed: float) -> str:
        \"\"\"Classify movement type based on speed.\"\"\"
        if speed < 0.1:
            return \"stationary\"
        elif speed < 0.8:
            return \"slow_walk\"
        elif speed < 1.5:
            return \"normal_walk\"
        elif speed < 3.0:
            return \"fast_walk\"
        else:
            return \"running\"
            
class MultiModalSentimentProcessor:
    \"\"\"Main processor for multi-modal sentiment analysis.\"\"\"
    
    def __init__(
        self,
        model: MultiModalSentimentModel,
        location_manager: LocationContextSentiment,
        audio_processor: Optional[AudioSpatialSentiment] = None
    ):
        self.model = model
        self.location_manager = location_manager
        self.audio_processor = audio_processor or AudioSpatialSentiment()
        self.text_processor = TextLocationSentiment(
            self.model.text_analyzer, 
            location_manager
        )
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    @handle_errors
    async def analyze_multi_modal(
        self,
        input_data: MultiModalInput
    ) -> Dict[str, Any]:
        \"\"\"Analyze sentiment using all available modalities.\"\"\"
        
        results = {
            \"timestamp\": input_data.timestamp or time.time(),
            \"modalities_used\": [],
            \"text_analysis\": None,
            \"audio_analysis\": None,
            \"spatial_analysis\": None,
            \"fused_analysis\": None
        }
        
        # Prepare tasks for parallel processing
        tasks = []
        
        # Text processing
        if input_data.text:
            tasks.append(self._process_text_async(input_data.text, input_data.location_context))
            results[\"modalities_used\"].append(\"text\")
            
        # Audio processing  
        if input_data.audio_data is not None or input_data.audio_file_path:
            tasks.append(self._process_audio_async(input_data))
            results[\"modalities_used\"].append(\"audio\")
            
        # Spatial processing
        if input_data.spatial_context is not None or input_data.location_context:
            tasks.append(self._process_spatial_async(input_data.location_context))
            results[\"modalities_used\"].append(\"spatial\")
            
        # Execute all tasks concurrently
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Assign results
            result_idx = 0
            if \"text\" in results[\"modalities_used\"]:
                results[\"text_analysis\"] = task_results[result_idx] if not isinstance(task_results[result_idx], Exception) else None
                result_idx += 1
                
            if \"audio\" in results[\"modalities_used\"]:
                results[\"audio_analysis\"] = task_results[result_idx] if not isinstance(task_results[result_idx], Exception) else None
                result_idx += 1
                
            if \"spatial\" in results[\"modalities_used\"]:
                results[\"spatial_analysis\"] = task_results[result_idx] if not isinstance(task_results[result_idx], Exception) else None
                
        # Multi-modal fusion
        if len(results[\"modalities_used\"]) > 1:
            results[\"fused_analysis\"] = await self._fuse_modalities(input_data, results)
            
        return results
        
    async def _process_text_async(self, text: str, location_context: Optional[LocationContext]):
        \"\"\"Process text sentiment asynchronously.\"\"\"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.text_processor.analyze_with_context,
            text,
            location_context
        )
        
    async def _process_audio_async(self, input_data: MultiModalInput):
        \"\"\"Process audio sentiment asynchronously.\"\"\"
        loop = asyncio.get_event_loop()
        
        def process_audio():
            # Load audio if file path provided
            if input_data.audio_file_path:
                audio_data, _ = self.audio_processor.load_audio(input_data.audio_file_path)
            else:
                audio_data = input_data.audio_data
                
            # Extract features
            features = self.audio_processor.extract_features(
                audio_data, input_data.spatial_context
            )
            
            # Analyze emotion
            emotion_scores = self.audio_processor.analyze_audio_emotion(features)
            
            return {
                \"features\": features,
                \"emotion_scores\": emotion_scores,
                \"feature_vector\": self.audio_processor.features_to_vector(features)
            }
            
        return await loop.run_in_executor(self.executor, process_audio)
        
    async def _process_spatial_async(self, location_context: Optional[LocationContext]):
        \"\"\"Process spatial context asynchronously.\"\"\"
        loop = asyncio.get_event_loop()
        
        def process_spatial():
            if location_context is None:
                location_context = self.location_manager.get_current_location()
                
            spatial_features = self.location_manager.get_spatial_features(location_context)
            movement_summary = self.location_manager.get_movement_summary()
            
            return {
                \"location_context\": location_context,
                \"spatial_features\": spatial_features,
                \"movement_summary\": movement_summary
            }
            
        return await loop.run_in_executor(self.executor, process_spatial)
        
    async def _fuse_modalities(self, input_data: MultiModalInput, results: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Fuse multiple modalities for final sentiment prediction.\"\"\"
        
        # Prepare inputs for multi-modal model
        text_input = input_data.text
        
        # Audio features
        audio_features = None
        if results[\"audio_analysis\"]:
            audio_features = torch.tensor(
                results[\"audio_analysis\"][\"feature_vector\"],
                dtype=torch.float32
            ).unsqueeze(0)
            
        # Spatial context
        spatial_context = None
        if results[\"spatial_analysis\"]:
            spatial_features = results[\"spatial_analysis\"][\"spatial_features\"]
            if len(spatial_features) >= 9:
                spatial_context = torch.tensor(
                    spatial_features[:9],
                    dtype=torch.float32
                ).unsqueeze(0)
                
        # Run multi-modal model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                text_input=text_input,
                audio_features=audio_features,
                spatial_context=spatial_context
            )
            
        # Process outputs
        sentiment_probs = outputs[\"sentiment_probs\"].cpu().numpy()
        modality_weights = outputs[\"modality_weights\"].cpu().numpy()
        
        predicted_class = np.argmax(sentiment_probs, axis=1)[0]
        class_names = [\"very_negative\", \"negative\", \"neutral\", \"positive\", \"very_positive\"]
        
        return {
            \"sentiment\": class_names[predicted_class],
            \"sentiment_probs\": sentiment_probs[0],
            \"modality_weights\": {
                \"text\": float(modality_weights[0]),
                \"audio\": float(modality_weights[1]),
                \"spatial\": float(modality_weights[2])
            },
            \"confidence\": float(np.max(sentiment_probs))
        }"