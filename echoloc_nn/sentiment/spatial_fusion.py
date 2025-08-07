"""
Spatial-sentiment fusion components.

Integrates EchoLoc ultrasonic positioning with sentiment analysis for 
context-aware emotional understanding based on location and movement.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import time

from ..inference.locator import EchoLocator
from ..utils.validation import validate_input
from ..utils.error_handling import handle_errors

@dataclass
class LocationContext:
    """Spatial context information for sentiment analysis."""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz] 
    acceleration: np.ndarray  # [ax, ay, az]
    confidence: float
    timestamp: float
    zone: Optional[str] = None
    proximity_objects: Optional[List[str]] = None

@dataclass 
class SentimentLocation:
    """Combined sentiment and location data."""
    sentiment: str
    sentiment_probs: np.ndarray
    confidence: float
    location: LocationContext
    spatial_influence: float
    text: str
    timestamp: float

class LocationContextSentiment:
    """
    Manages spatial context for sentiment analysis.
    
    Tracks location history, movement patterns, and environmental zones
    to provide rich spatial context for sentiment understanding.
    """
    
    def __init__(
        self,
        echoloc_model: Optional[EchoLocator] = None,
        history_length: int = 50,
        zone_definitions: Optional[Dict[str, Dict]] = None
    ):
        self.echoloc = echoloc_model
        self.history_length = history_length
        
        # Location history tracking
        self.location_history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length)
        
        # Zone definitions (spatial regions with semantic meaning)
        self.zones = zone_definitions or self._default_zones()
        
        # Movement pattern analysis
        self.movement_analyzer = MovementPatternAnalyzer()
        
    def _default_zones(self) -> Dict[str, Dict]:
        """Default spatial zone definitions."""
        return {
            "workspace": {
                "bounds": [[-2, 2], [-2, 2], [0, 3]],  # x, y, z ranges
                "sentiment_bias": 0.1,  # slight positive bias
                "description": "Primary work area"
            },
            "social_area": {
                "bounds": [[-5, 5], [3, 8], [0, 3]],
                "sentiment_bias": 0.2,  # more positive bias
                "description": "Social interaction zone"
            },
            "quiet_zone": {
                "bounds": [[8, 12], [-2, 2], [0, 3]],
                "sentiment_bias": -0.1,  # slight negative bias (stress/focus)
                "description": "Quiet concentration area"
            }
        }
        
    @handle_errors
    def get_current_location(self, echo_data: Optional[np.ndarray] = None) -> LocationContext:
        """Get current location context from EchoLoc system."""
        
        if self.echoloc is None:
            # Return dummy location for testing
            return LocationContext(
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                confidence=0.5,
                timestamp=time.time()
            )
            
        # Get position from EchoLoc
        if echo_data is not None:
            position, confidence = self.echoloc.locate(echo_data)
        else:
            # Use last known position or default
            if self.location_history:
                last_loc = self.location_history[-1]
                position = last_loc.position
                confidence = last_loc.confidence * 0.8  # decay confidence
            else:
                position = np.array([0.0, 0.0, 0.0])
                confidence = 0.1
                
        current_time = time.time()
        
        # Calculate velocity and acceleration
        velocity, acceleration = self._calculate_motion(position, current_time)
        
        # Determine spatial zone
        zone = self._determine_zone(position)
        
        # Create location context
        context = LocationContext(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            confidence=confidence,
            timestamp=current_time,
            zone=zone
        )
        
        # Update history
        self.location_history.append(context)
        
        return context
        
    def _calculate_motion(self, position: np.ndarray, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        \"\"\"Calculate velocity and acceleration from position history.\"\"\"
        
        if len(self.location_history) < 2:
            return np.zeros(3), np.zeros(3)
            
        # Calculate velocity (finite difference)
        last_context = self.location_history[-1]
        dt = timestamp - last_context.timestamp
        
        if dt > 0:
            velocity = (position - last_context.position) / dt
        else:
            velocity = last_context.velocity
            
        # Calculate acceleration
        if len(self.velocity_history) > 0:
            last_velocity = self.velocity_history[-1]
            if dt > 0:
                acceleration = (velocity - last_velocity) / dt
            else:
                acceleration = np.zeros(3)
        else:
            acceleration = np.zeros(3)
            
        # Store velocity for next calculation
        self.velocity_history.append(velocity)
        
        return velocity, acceleration
        
    def _determine_zone(self, position: np.ndarray) -> Optional[str]:
        \"\"\"Determine which spatial zone the position falls into.\"\"\"
        
        for zone_name, zone_config in self.zones.items():
            bounds = zone_config[\"bounds\"]
            
            # Check if position is within zone bounds
            in_zone = True
            for i, (min_bound, max_bound) in enumerate(bounds):
                if not (min_bound <= position[i] <= max_bound):
                    in_zone = False
                    break
                    
            if in_zone:
                return zone_name
                
        return None
        
    def get_spatial_features(self, context: LocationContext) -> np.ndarray:
        \"\"\"Extract numerical spatial features for model input.\"\"\"
        
        # Base features: position, velocity, acceleration
        features = np.concatenate([
            context.position,
            context.velocity, 
            context.acceleration
        ])
        
        # Add zone information as one-hot encoding
        zone_features = np.zeros(len(self.zones))
        if context.zone:
            zone_idx = list(self.zones.keys()).index(context.zone)
            zone_features[zone_idx] = 1.0
            
        # Movement pattern features
        movement_features = self.movement_analyzer.extract_features(self.location_history)
        
        # Combine all features
        all_features = np.concatenate([
            features,  # 9 features
            zone_features,  # len(zones) features
            movement_features  # variable length
        ])
        
        return all_features
        
    def get_zone_sentiment_bias(self, zone: Optional[str]) -> float:
        \"\"\"Get sentiment bias for a spatial zone.\"\"\"
        if zone and zone in self.zones:
            return self.zones[zone].get(\"sentiment_bias\", 0.0)
        return 0.0
        
    def add_zone(self, name: str, bounds: List[List[float]], 
                sentiment_bias: float = 0.0, description: str = \"\"):
        \"\"\"Add a new spatial zone definition.\"\"\"
        self.zones[name] = {
            \"bounds\": bounds,
            \"sentiment_bias\": sentiment_bias,
            \"description\": description
        }
        
    def get_movement_summary(self) -> Dict[str, float]:
        \"\"\"Get summary of recent movement patterns.\"\"\"
        return self.movement_analyzer.get_summary(self.location_history)

class MovementPatternAnalyzer:
    \"\"\"Analyzes movement patterns for spatial-sentiment context.\"\"\"
    
    def __init__(self):
        self.pattern_thresholds = {
            \"stationary\": 0.1,  # m/s
            \"walking\": 1.5,     # m/s
            \"rapid\": 3.0        # m/s
        }
        
    def extract_features(self, location_history: deque) -> np.ndarray:
        \"\"\"Extract movement pattern features.\"\"\"
        
        if len(location_history) < 2:
            return np.zeros(6)  # Default feature vector
            
        # Calculate movement statistics
        velocities = [loc.velocity for loc in location_history]
        speeds = [np.linalg.norm(vel) for vel in velocities]
        
        if not speeds:
            return np.zeros(6)
            
        # Basic statistics
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        speed_std = np.std(speeds)
        
        # Movement categorization
        stationary_ratio = sum(1 for s in speeds if s < self.pattern_thresholds[\"stationary\"]) / len(speeds)
        walking_ratio = sum(1 for s in speeds if self.pattern_thresholds[\"stationary\"] <= s < self.pattern_thresholds[\"walking\"]) / len(speeds)
        rapid_ratio = sum(1 for s in speeds if s >= self.pattern_thresholds[\"rapid\"]) / len(speeds)
        
        features = np.array([
            avg_speed,
            max_speed,
            speed_std,
            stationary_ratio,
            walking_ratio,
            rapid_ratio
        ])
        
        return features
        
    def get_summary(self, location_history: deque) -> Dict[str, float]:
        \"\"\"Get human-readable movement summary.\"\"\"
        
        features = self.extract_features(location_history)
        
        if len(features) == 6:
            return {
                \"average_speed_ms\": features[0],
                \"max_speed_ms\": features[1],
                \"speed_variability\": features[2],
                \"time_stationary\": features[3],
                \"time_walking\": features[4],
                \"time_rapid_movement\": features[5]
            }
        else:
            return {\"no_data\": True}
            
    def classify_current_movement(self, current_speed: float) -> str:
        \"\"\"Classify current movement type.\"\"\"
        
        if current_speed < self.pattern_thresholds[\"stationary\"]:
            return \"stationary\"
        elif current_speed < self.pattern_thresholds[\"walking\"]:
            return \"walking\"
        elif current_speed < self.pattern_thresholds[\"rapid\"]:
            return \"running\"
        else:
            return \"rapid_movement\"