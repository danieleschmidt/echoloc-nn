"""
Spatial-aware sentiment analysis models.

Combines transformer-based sentiment analysis with spatial positioning data
from the EchoLoc ultrasonic localization system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel
from ..utils.validation import validate_input, validate_tensor
from ..utils.error_handling import handle_errors, EchoLocError

class SpatialSentimentAnalyzer(nn.Module):
    """
    Transformer-based sentiment analyzer with spatial context integration.
    
    Processes text sentiment while considering spatial location, movement
    patterns, and environmental context from ultrasonic positioning.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        spatial_dim: int = 128,
        sentiment_classes: int = 3,  # negative, neutral, positive
        max_sequence_length: int = 512,
        dropout: float = 0.1,
        spatial_weight: float = 0.3
    ):
        super().__init__()
        
        # Load pre-trained transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Configuration
        self.spatial_dim = spatial_dim
        self.sentiment_classes = sentiment_classes
        self.max_length = max_sequence_length
        self.spatial_weight = spatial_weight
        
        # Get transformer output dimension
        transformer_dim = self.transformer.config.hidden_size
        
        # Spatial encoding layers
        self.spatial_encoder = nn.Sequential(
            nn.Linear(9, spatial_dim),  # 3D pos + 3D velocity + 3D acceleration
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim, spatial_dim),
            nn.ReLU()
        )
        
        # Spatial attention mechanism
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=spatial_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(transformer_dim + spatial_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Sentiment classification head
        self.sentiment_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, sentiment_classes)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Spatial influence head (how much location affects sentiment)
        self.spatial_influence_head = nn.Sequential(
            nn.Linear(spatial_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    @handle_errors
    def forward(
        self,
        text_input: Union[str, List[str]],
        spatial_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with text and spatial context.
        
        Args:
            text_input: Input text string(s) 
            spatial_context: Tensor of shape (batch, 9) containing:
                           [x, y, z, vx, vy, vz, ax, ay, az]
            attention_mask: Text attention mask
            
        Returns:
            Dictionary containing:
            - sentiment_logits: Raw sentiment scores
            - sentiment_probs: Softmax probabilities  
            - confidence: Prediction confidence
            - spatial_influence: How much location affected prediction
        """
        
        # Handle text input
        if isinstance(text_input, str):
            text_input = [text_input]
            
        # Tokenize text
        encoding = self.tokenizer(
            text_input,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"]
        if attention_mask is None:
            attention_mask = encoding["attention_mask"]
            
        batch_size = input_ids.size(0)
        
        # Process text through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        text_features = transformer_outputs.last_hidden_state[:, 0, :]
        
        # Process spatial context if provided
        if spatial_context is not None:
            validate_tensor(spatial_context, expected_shape=(batch_size, 9))
            
            # Encode spatial features
            spatial_features = self.spatial_encoder(spatial_context)
            
            # Apply spatial attention 
            spatial_attended, _ = self.spatial_attention(
                spatial_features.unsqueeze(1),
                spatial_features.unsqueeze(1), 
                spatial_features.unsqueeze(1)
            )
            spatial_attended = spatial_attended.squeeze(1)
            
            # Compute spatial influence
            spatial_influence = self.spatial_influence_head(spatial_attended)
            
            # Fuse text and spatial features
            combined_features = torch.cat([text_features, spatial_attended], dim=1)
            fused_features = self.fusion_layer(combined_features)
            
            # Weight fusion based on spatial influence
            weighted_features = (
                (1 - self.spatial_weight + self.spatial_weight * spatial_influence) * fused_features
            )
            
        else:
            # Text-only processing
            weighted_features = self.fusion_layer(
                torch.cat([text_features, torch.zeros(batch_size, self.spatial_dim)], dim=1)
            )
            spatial_influence = torch.zeros(batch_size, 1)
        
        # Generate predictions
        sentiment_logits = self.sentiment_head(weighted_features)
        sentiment_probs = F.softmax(sentiment_logits, dim=1)
        confidence = self.confidence_head(weighted_features)
        
        return {
            "sentiment_logits": sentiment_logits,
            "sentiment_probs": sentiment_probs,
            "confidence": confidence,
            "spatial_influence": spatial_influence,
            "text_features": text_features,
            "spatial_features": spatial_attended if spatial_context is not None else None
        }
        
    def predict_sentiment(
        self,
        text: Union[str, List[str]],
        spatial_context: Optional[np.ndarray] = None
    ) -> Dict[str, Union[str, float, np.ndarray]]:
        """
        High-level sentiment prediction interface.
        
        Args:
            text: Input text
            spatial_context: Optional spatial context array
            
        Returns:
            Dictionary with sentiment prediction results
        """
        
        self.eval()
        with torch.no_grad():
            # Convert spatial context to tensor if provided
            spatial_tensor = None
            if spatial_context is not None:
                spatial_tensor = torch.tensor(
                    spatial_context, dtype=torch.float32
                ).unsqueeze(0) if spatial_context.ndim == 1 else torch.tensor(
                    spatial_context, dtype=torch.float32
                )
            
            # Forward pass
            outputs = self.forward(text, spatial_tensor)
            
            # Extract predictions
            probs = outputs["sentiment_probs"].cpu().numpy()
            confidence = outputs["confidence"].cpu().numpy()
            spatial_influence = outputs["spatial_influence"].cpu().numpy()
            
            # Get predicted classes
            predicted_classes = np.argmax(probs, axis=1)
            class_names = ["negative", "neutral", "positive"]
            
            results = {
                "sentiment": [class_names[cls] for cls in predicted_classes],
                "probabilities": probs,
                "confidence": confidence.squeeze(),
                "spatial_influence": spatial_influence.squeeze()
            }
            
            # Single input case
            if isinstance(text, str):
                results = {k: v[0] if isinstance(v, (list, np.ndarray)) and len(v) == 1 else v 
                          for k, v in results.items()}
                
            return results

class MultiModalSentimentModel(nn.Module):
    """
    Multi-modal sentiment analysis combining text, audio, and spatial data.
    
    Integrates multiple data streams for comprehensive sentiment understanding
    in spatially-aware environments.
    """
    
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        audio_features_dim: int = 128,
        spatial_dim: int = 64,
        fusion_dim: int = 256,
        sentiment_classes: int = 5,  # very negative to very positive
        dropout: float = 0.15
    ):
        super().__init__()
        
        # Text processing
        self.text_analyzer = SpatialSentimentAnalyzer(
            model_name=text_model_name,
            spatial_dim=spatial_dim,
            sentiment_classes=sentiment_classes,
            dropout=dropout
        )
        
        # Audio processing  
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_features_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final fusion and classification
        self.final_fusion = nn.Sequential(
            nn.Linear(128 + 64 + spatial_dim, fusion_dim),  # text + audio + spatial
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 128),
            nn.ReLU()
        )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, sentiment_classes)
        )
        
        # Modality importance weights
        self.modality_weights = nn.Parameter(torch.ones(3))  # text, audio, spatial
        
    def forward(
        self,
        text_input: Union[str, List[str]],
        audio_features: Optional[torch.Tensor] = None,
        spatial_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Multi-modal forward pass."""
        
        # Process text with spatial context
        text_outputs = self.text_analyzer(text_input, spatial_context)
        text_features = text_outputs["text_features"]
        
        batch_size = text_features.size(0)
        feature_list = [text_features]
        
        # Process audio features
        if audio_features is not None:
            audio_encoded = self.audio_encoder(audio_features)
            feature_list.append(audio_encoded)
        else:
            feature_list.append(torch.zeros(batch_size, 64))
            
        # Add spatial features
        if spatial_context is not None and text_outputs["spatial_features"] is not None:
            feature_list.append(text_outputs["spatial_features"])
        else:
            feature_list.append(torch.zeros(batch_size, self.text_analyzer.spatial_dim))
        
        # Apply modality weighting
        weights = F.softmax(self.modality_weights, dim=0)
        weighted_features = []
        for i, features in enumerate(feature_list):
            weighted_features.append(weights[i] * features)
        
        # Concatenate all features
        all_features = torch.cat(weighted_features, dim=1)
        
        # Final fusion and classification
        fused = self.final_fusion(all_features)
        sentiment_logits = self.final_classifier(fused)
        
        return {
            "sentiment_logits": sentiment_logits,
            "sentiment_probs": F.softmax(sentiment_logits, dim=1),
            "modality_weights": weights,
            "text_outputs": text_outputs
        }