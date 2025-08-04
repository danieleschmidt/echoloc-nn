"""
Loss functions for EchoLoc-NN training.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionLoss(nn.Module):
    """
    Position estimation loss function.
    
    Supports various distance metrics for position regression
    with optional uncertainty weighting.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean",
        uncertainty_weighting: bool = False
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.uncertainty_weighting = uncertainty_weighting
        
        if self.loss_type not in ["mse", "mae", "smooth_l1", "huber"]:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        true_positions: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute position loss.
        
        Args:
            pred_positions: Predicted positions (batch_size, 3)
            true_positions: True positions (batch_size, 3)  
            confidence: Confidence scores (batch_size, 1) for uncertainty weighting
            
        Returns:
            Position loss tensor
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(pred_positions, true_positions, reduction='none')
        elif self.loss_type == "mae":
            loss = F.l1_loss(pred_positions, true_positions, reduction='none')
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(pred_positions, true_positions, reduction='none')
        elif self.loss_type == "huber":
            loss = F.huber_loss(pred_positions, true_positions, reduction='none', delta=0.1)
            
        # Sum over coordinate dimensions
        loss = torch.sum(loss, dim=1)  # (batch_size,)
        
        # Apply uncertainty weighting if enabled
        if self.uncertainty_weighting and confidence is not None:
            # Higher confidence -> lower uncertainty -> higher weight
            weights = confidence.squeeze(-1)  # (batch_size,)
            loss = loss * weights
            
        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class ConfidenceLoss(nn.Module):
    """
    Confidence estimation loss function.
    
    Trains the model to predict confidence scores that correlate
    with actual position accuracy.
    """
    
    def __init__(
        self,
        loss_type: str = "mse_based",
        alpha: float = 1.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(
        self,
        pred_confidence: torch.Tensor,
        pred_positions: torch.Tensor,
        true_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute confidence loss.
        
        Args:
            pred_confidence: Predicted confidence (batch_size, 1)
            pred_positions: Predicted positions (batch_size, 3)
            true_positions: True positions (batch_size, 3)
            
        Returns:
            Confidence loss tensor
        """
        # Compute position errors
        position_errors = torch.norm(pred_positions - true_positions, dim=1)  # (batch_size,)
        
        if self.loss_type == "mse_based":
            # Target confidence based on inverse of position error
            # High accuracy -> high confidence, low accuracy -> low confidence
            max_error = 1.0  # Maximum expected error in meters
            target_confidence = torch.exp(-self.alpha * position_errors / max_error)
            
            # MSE loss between predicted and target confidence
            loss = F.mse_loss(pred_confidence.squeeze(-1), target_confidence, reduction='none')
            
        elif self.loss_type == "ranking":
            # Ranking loss: more accurate predictions should have higher confidence
            batch_size = pred_confidence.size(0)
            
            # Create pairwise comparisons
            errors_expanded = position_errors.unsqueeze(1).expand(-1, batch_size)
            errors_transposed = position_errors.unsqueeze(0).expand(batch_size, -1)
            
            conf_expanded = pred_confidence.squeeze(-1).unsqueeze(1).expand(-1, batch_size)
            conf_transposed = pred_confidence.squeeze(-1).unsqueeze(0).expand(batch_size, -1)
            
            # Mask for valid comparisons (where errors are different)
            error_diff = errors_expanded - errors_transposed
            valid_mask = torch.abs(error_diff) > 1e-6
            
            # Ranking loss: if error_i < error_j, then confidence_i should > confidence_j
            conf_diff = conf_expanded - conf_transposed
            ranking_loss = torch.relu(-error_diff * conf_diff)  # Hinge loss
            
            loss = ranking_loss[valid_mask]
            
        elif self.loss_type == "calibration":
            # Calibration loss: confidence should match actual accuracy
            # Bin predictions by confidence and compute calibration error
            n_bins = 10
            bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=pred_confidence.device)
            
            calibration_errors = []
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                # Find predictions in this confidence bin
                in_bin = (pred_confidence.squeeze(-1) > bin_lower) & (pred_confidence.squeeze(-1) <= bin_upper)
                
                if torch.sum(in_bin) > 0:
                    # Average confidence in bin
                    bin_confidence = torch.mean(pred_confidence.squeeze(-1)[in_bin])
                    
                    # Average accuracy in bin (inverse of error)
                    bin_errors = position_errors[in_bin]
                    bin_accuracy = torch.mean(torch.exp(-bin_errors))
                    
                    # Calibration error for this bin
                    calibration_error = torch.abs(bin_confidence - bin_accuracy)
                    calibration_errors.append(calibration_error)
            
            if calibration_errors:
                loss = torch.stack(calibration_errors)
            else:
                loss = torch.zeros(1, device=pred_confidence.device)
                
        else:
            raise ValueError(f"Unknown confidence loss type: {self.loss_type}")
            
        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for position and confidence estimation.
    
    Balances position accuracy and confidence calibration with
    configurable weights.
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        confidence_weight: float = 0.1,
        position_loss_type: str = "mse",
        confidence_loss_type: str = "mse_based",
        uncertainty_weighting: bool = True
    ):
        super().__init__()
        
        self.position_weight = position_weight
        self.confidence_weight = confidence_weight
        
        self.position_loss = PositionLoss(
            loss_type=position_loss_type,
            uncertainty_weighting=uncertainty_weighting
        )
        
        self.confidence_loss = ConfidenceLoss(
            loss_type=confidence_loss_type
        )
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        pred_confidence: torch.Tensor,
        true_positions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_positions: Predicted positions (batch_size, 3)
            pred_confidence: Predicted confidence (batch_size, 1)
            true_positions: True positions (batch_size, 3)
            
        Returns:
            Dictionary with individual and total losses
        """
        # Compute individual losses
        pos_loss = self.position_loss(pred_positions, true_positions, pred_confidence)
        conf_loss = self.confidence_loss(pred_confidence, pred_positions, true_positions)
        
        # Combine losses
        total_loss = (
            self.position_weight * pos_loss + 
            self.confidence_weight * conf_loss
        )
        
        return {
            'position_loss': pos_loss,
            'confidence_loss': conf_loss,
            'total_loss': total_loss
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning.
    
    Encourages similar echo patterns to have similar representations
    and dissimilar patterns to have different representations.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 1.0,
        distance_metric: str = "cosine"
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Feature embeddings (batch_size, embedding_dim)
            labels: Position labels for automatic pair generation
            positive_pairs: Explicit positive pairs (n_pairs, 2)
            negative_pairs: Explicit negative pairs (n_pairs, 2)
            
        Returns:
            Contrastive loss tensor
        """
        batch_size = embeddings.size(0)
        
        if positive_pairs is None and negative_pairs is None:
            # Generate pairs based on position labels
            if labels is None:
                raise ValueError("Must provide either labels or explicit pairs")
                
            # Compute pairwise distances in position space
            pos_distances = torch.cdist(labels, labels, p=2)
            
            # Define positive pairs (close in position) and negative pairs (far in position)
            close_threshold = 0.1  # 10cm
            far_threshold = 1.0    # 1m
            
            positive_mask = (pos_distances < close_threshold) & (pos_distances > 0)
            negative_mask = pos_distances > far_threshold
            
            positive_pairs = torch.nonzero(positive_mask, as_tuple=False)
            negative_pairs = torch.nonzero(negative_mask, as_tuple=False)
        
        # Compute embedding distances
        if self.distance_metric == "cosine":
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
            distance_matrix = 1 - similarity_matrix
        elif self.distance_metric == "euclidean":
            distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Compute positive loss (minimize distance for similar pairs)
        if len(positive_pairs) > 0:
            pos_distances = distance_matrix[positive_pairs[:, 0], positive_pairs[:, 1]]
            positive_loss = torch.mean(pos_distances)
        else:
            positive_loss = torch.tensor(0.0, device=embeddings.device)
        
        # Compute negative loss (maximize distance for dissimilar pairs)
        if len(negative_pairs) > 0:
            neg_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
            negative_loss = torch.mean(torch.relu(self.margin - neg_distances))
        else:
            negative_loss = torch.tensor(0.0, device=embeddings.device)
        
        return positive_loss + negative_loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for domain adaptation.
    
    Helps models generalize across different acoustic environments
    by learning domain-invariant representations.
    """
    
    def __init__(self, domain_classifier: nn.Module):
        super().__init__()
        self.domain_classifier = domain_classifier
        
    def forward(
        self,
        features: torch.Tensor,
        domain_labels: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Compute adversarial domain adaptation loss.
        
        Args:
            features: Feature representations (batch_size, feature_dim)
            domain_labels: Domain labels (batch_size,)
            alpha: Gradient reversal strength
            
        Returns:
            Adversarial loss tensor
        """
        # Apply gradient reversal
        reversed_features = GradientReversalFunction.apply(features, alpha)
        
        # Domain classification
        domain_pred = self.domain_classifier(reversed_features)
        
        # Cross-entropy loss for domain classification
        domain_loss = F.cross_entropy(domain_pred, domain_labels)
        
        return domain_loss


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None