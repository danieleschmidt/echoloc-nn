"""
Global utilities for EchoLoc-NN.

This module provides internationalization, compliance, and 
multi-region deployment capabilities for global use.
"""

from .i18n import (
    I18nManager,
    get_message,
    set_language,
    get_current_language,
    get_supported_languages
)

from .compliance import (
    ComplianceManager,
    DataRetentionPolicy,
    PrivacyMetadata,
    RegionalComplianceAdapter
)

__all__ = [
    "I18nManager",
    "get_message",
    "set_language", 
    "get_current_language",
    "get_supported_languages",
    "ComplianceManager",
    "DataRetentionPolicy",
    "PrivacyMetadata",
    "RegionalComplianceAdapter"
]