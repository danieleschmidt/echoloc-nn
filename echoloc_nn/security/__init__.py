"""
EchoLoc-NN Security Framework

Enterprise-grade security components for quantum-spatial localization systems.
Includes encryption, authentication, secure communications, and threat detection.

Components:
- crypto: Cryptographic utilities and secure key management
- authentication: Multi-factor authentication and access control
- secure_comm: Encrypted communication protocols
- threat_detection: Security anomaly and intrusion detection
- data_protection: Privacy-preserving computation and data sanitization
"""

from .crypto import (
    SecureKeyManager,
    QuantumResistantCrypto,
    EncryptionEngine
)

from .authentication import (
    MultiFactorAuth,
    AccessController,
    SessionManager
)

from .secure_comm import (
    SecureChannel,
    MessageAuthenticator,
    SecureProtocol
)

# from .threat_detection import (
#     IntrusionDetector,
#     AnomalyMonitor,
#     SecurityAnalyzer
# )

# from .data_protection import (
#     PrivacyPreserver,
#     DataSanitizer,
#     SecureComputation
# )

__all__ = [
    # Cryptography
    "SecureKeyManager",
    "QuantumResistantCrypto",
    "EncryptionEngine",
    
    # Authentication
    "MultiFactorAuth",
    "AccessController", 
    "SessionManager",
    
    # Secure communications
    "SecureChannel",
    "EncryptedProtocol",
    "TLSManager",
    
    # Threat detection
    "IntrusionDetector",
    "AnomalyMonitor",
    "SecurityAnalyzer",
    
    # Data protection
    "PrivacyPreserver",
    "DataSanitizer",
    "SecureComputation"
]