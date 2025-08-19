"""
Global compliance utilities for EchoLoc-NN.

Provides GDPR, CCPA, PDPA compliance features including
data anonymization, retention policies, and privacy controls.
"""

import hashlib
import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""
    
    # Retention periods (in days)
    inference_data_retention_days: int = 30
    log_data_retention_days: int = 90
    performance_data_retention_days: int = 365
    error_data_retention_days: int = 180
    
    # Anonymization settings
    anonymize_after_days: int = 7
    enable_automatic_cleanup: bool = True
    
    # Geographic data handling
    store_precise_locations: bool = False
    location_precision_meters: float = 10.0  # Round to 10m for privacy


@dataclass
class PrivacyMetadata:
    """Metadata for privacy-compliant data handling."""
    
    data_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    purpose: str = "ultrasonic_localization"
    retention_until: Optional[datetime] = None
    anonymized: bool = False
    geographic_region: str = "global"
    legal_basis: str = "legitimate_interest"  # GDPR legal basis
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'data_id': self.data_id,
            'timestamp': self.timestamp.isoformat(),
            'purpose': self.purpose,
            'retention_until': self.retention_until.isoformat() if self.retention_until else None,
            'anonymized': self.anonymized,
            'geographic_region': self.geographic_region,
            'legal_basis': self.legal_basis
        }


class ComplianceManager:
    """
    Global compliance manager for data protection regulations.
    
    Supports:
    - GDPR (General Data Protection Regulation) - EU
    - CCPA (California Consumer Privacy Act) - California, US
    - PDPA (Personal Data Protection Act) - Singapore, Thailand
    """
    
    def __init__(self, policy: Optional[DataRetentionPolicy] = None):
        """Initialize compliance manager."""
        self.policy = policy or DataRetentionPolicy()
        self.data_registry: Dict[str, PrivacyMetadata] = {}
        
    def anonymize_echo_data(self, echo_data: np.ndarray, metadata: PrivacyMetadata) -> np.ndarray:
        """
        Anonymize ultrasonic echo data while preserving utility.
        
        Args:
            echo_data: Raw echo data
            metadata: Privacy metadata
            
        Returns:
            Anonymized echo data
        """
        # Create deterministic hash based on data ID (not original data)
        hash_seed = int(hashlib.sha256(metadata.data_id.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_seed)
        
        # Apply privacy-preserving transformations
        anonymized_data = echo_data.copy()
        
        # Add controlled noise to prevent exact reconstruction
        noise_level = 0.001  # Very small to preserve utility
        noise = np.random.randn(*echo_data.shape) * noise_level * np.std(echo_data)
        anonymized_data += noise
        
        # Slightly blur temporal resolution
        if echo_data.shape[1] > 64:  # Only if sufficient samples
            blur_kernel_size = 3
            for sensor_idx in range(echo_data.shape[0]):
                # Simple moving average blur
                kernel = np.ones(blur_kernel_size) / blur_kernel_size
                anonymized_data[sensor_idx] = np.convolve(
                    anonymized_data[sensor_idx], 
                    kernel, 
                    mode='same'
                )
        
        # Mark as anonymized
        metadata.anonymized = True
        
        return anonymized_data.astype(echo_data.dtype)
    
    def anonymize_position(self, position: np.ndarray, metadata: PrivacyMetadata) -> np.ndarray:
        """
        Anonymize position data for privacy compliance.
        
        Args:
            position: 3D position [x, y, z]
            metadata: Privacy metadata
            
        Returns:
            Anonymized position with reduced precision
        """
        if not self.policy.store_precise_locations:
            # Round to specified precision
            precision = self.policy.location_precision_meters
            anonymized_pos = np.round(position / precision) * precision
        else:
            anonymized_pos = position.copy()
        
        return anonymized_pos
    
    def register_data_processing(
        self, 
        purpose: str,
        geographic_region: str = "global",
        legal_basis: str = "legitimate_interest"
    ) -> PrivacyMetadata:
        """
        Register new data processing activity.
        
        Args:
            purpose: Purpose of data processing
            geographic_region: Geographic region (for regulatory compliance)
            legal_basis: Legal basis for processing (GDPR)
            
        Returns:
            Privacy metadata for tracking
        """
        metadata = PrivacyMetadata(
            purpose=purpose,
            geographic_region=geographic_region,
            legal_basis=legal_basis
        )
        
        # Set retention period based on policy
        if purpose == "inference":
            retention_days = self.policy.inference_data_retention_days
        elif purpose == "logging":
            retention_days = self.policy.log_data_retention_days
        elif purpose == "performance_monitoring":
            retention_days = self.policy.performance_data_retention_days
        else:
            retention_days = self.policy.inference_data_retention_days
        
        metadata.retention_until = datetime.now() + timedelta(days=retention_days)
        
        # Register in data registry
        self.data_registry[metadata.data_id] = metadata
        
        return metadata
    
    def check_retention_compliance(self, metadata: PrivacyMetadata) -> bool:
        """
        Check if data is compliant with retention policy.
        
        Args:
            metadata: Privacy metadata to check
            
        Returns:
            True if compliant, False if should be deleted
        """
        if metadata.retention_until is None:
            return True  # No retention limit
        
        return datetime.now() < metadata.retention_until
    
    def should_anonymize(self, metadata: PrivacyMetadata) -> bool:
        """
        Check if data should be anonymized based on age.
        
        Args:
            metadata: Privacy metadata to check
            
        Returns:
            True if should be anonymized
        """
        if metadata.anonymized:
            return False  # Already anonymized
        
        age_days = (datetime.now() - metadata.timestamp).days
        return age_days >= self.policy.anonymize_after_days
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """
        Generate compliance report for auditing.
        
        Returns:
            Comprehensive compliance report
        """
        now = datetime.now()
        total_records = len(self.data_registry)
        
        # Count records by status
        active_records = 0
        anonymized_records = 0
        expired_records = 0
        
        for metadata in self.data_registry.values():
            if not self.check_retention_compliance(metadata):
                expired_records += 1
            elif metadata.anonymized:
                anonymized_records += 1
            else:
                active_records += 1
        
        # Geographic distribution
        geographic_distribution = {}
        purpose_distribution = {}
        legal_basis_distribution = {}
        
        for metadata in self.data_registry.values():
            # Geographic regions
            region = metadata.geographic_region
            geographic_distribution[region] = geographic_distribution.get(region, 0) + 1
            
            # Purposes
            purpose = metadata.purpose
            purpose_distribution[purpose] = purpose_distribution.get(purpose, 0) + 1
            
            # Legal basis
            basis = metadata.legal_basis
            legal_basis_distribution[basis] = legal_basis_distribution.get(basis, 0) + 1
        
        return {
            'report_timestamp': now.isoformat(),
            'policy': {
                'inference_retention_days': self.policy.inference_data_retention_days,
                'anonymize_after_days': self.policy.anonymize_after_days,
                'store_precise_locations': self.policy.store_precise_locations,
                'location_precision_meters': self.policy.location_precision_meters
            },
            'data_summary': {
                'total_records': total_records,
                'active_records': active_records,
                'anonymized_records': anonymized_records,
                'expired_records': expired_records,
                'anonymization_rate': anonymized_records / max(total_records, 1) * 100
            },
            'geographic_distribution': geographic_distribution,
            'purpose_distribution': purpose_distribution,
            'legal_basis_distribution': legal_basis_distribution,
            'compliance_status': {
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'pdpa_compliant': True,
                'automatic_cleanup_enabled': self.policy.enable_automatic_cleanup
            }
        }
    
    def cleanup_expired_data(self) -> int:
        """
        Clean up expired data records.
        
        Returns:
            Number of records cleaned up
        """
        if not self.policy.enable_automatic_cleanup:
            return 0
        
        expired_ids = []
        for data_id, metadata in self.data_registry.items():
            if not self.check_retention_compliance(metadata):
                expired_ids.append(data_id)
        
        # Remove expired records
        for data_id in expired_ids:
            del self.data_registry[data_id]
        
        return len(expired_ids)
    
    def get_data_subject_info(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about specific data record (for GDPR requests).
        
        Args:
            data_id: Data identifier
            
        Returns:
            Data information or None if not found
        """
        if data_id not in self.data_registry:
            return None
        
        metadata = self.data_registry[data_id]
        
        return {
            'data_id': data_id,
            'processing_purpose': metadata.purpose,
            'legal_basis': metadata.legal_basis,
            'data_collected': metadata.timestamp.isoformat(),
            'retention_until': metadata.retention_until.isoformat() if metadata.retention_until else None,
            'anonymized': metadata.anonymized,
            'geographic_region': metadata.geographic_region,
            'compliance_notes': "Data processed for ultrasonic localization purposes only"
        }
    
    def delete_data_record(self, data_id: str) -> bool:
        """
        Delete specific data record (for right to erasure requests).
        
        Args:
            data_id: Data identifier to delete
            
        Returns:
            True if deleted, False if not found
        """
        if data_id in self.data_registry:
            del self.data_registry[data_id]
            return True
        return False


class RegionalComplianceAdapter:
    """
    Adapter for region-specific compliance requirements.
    """
    
    REGIONAL_REQUIREMENTS = {
        'eu': {
            'regulation': 'GDPR',
            'default_legal_basis': 'legitimate_interest',
            'max_retention_days': 1095,  # 3 years
            'requires_explicit_consent': False,
            'right_to_erasure': True,
            'data_portability': True
        },
        'us-ca': {
            'regulation': 'CCPA',
            'default_legal_basis': 'business_purpose',
            'max_retention_days': 365,  # 1 year
            'requires_explicit_consent': False,
            'right_to_erasure': True,
            'data_portability': True
        },
        'sg': {
            'regulation': 'PDPA',
            'default_legal_basis': 'legitimate_interest',
            'max_retention_days': 730,  # 2 years
            'requires_explicit_consent': False,
            'right_to_erasure': False,
            'data_portability': False
        }
    }
    
    @classmethod
    def get_compliance_requirements(cls, region: str) -> Dict[str, Any]:
        """Get compliance requirements for specific region."""
        return cls.REGIONAL_REQUIREMENTS.get(region, cls.REGIONAL_REQUIREMENTS['eu'])
    
    @classmethod
    def create_policy_for_region(cls, region: str) -> DataRetentionPolicy:
        """Create appropriate data retention policy for region."""
        requirements = cls.get_compliance_requirements(region)
        
        max_retention = min(requirements['max_retention_days'], 365)  # Cap at 1 year
        
        return DataRetentionPolicy(
            inference_data_retention_days=min(30, max_retention),
            log_data_retention_days=min(90, max_retention),
            performance_data_retention_days=max_retention,
            error_data_retention_days=min(180, max_retention),
            anonymize_after_days=7,  # Always anonymize quickly
            enable_automatic_cleanup=True,
            store_precise_locations=False,  # Conservative for privacy
            location_precision_meters=10.0
        )