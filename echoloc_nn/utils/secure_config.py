"""
Secure Configuration Management for EchoLoc-NN

Provides secure configuration loading with environment variable support,
secret management, and validation.
"""

import os
import json
import secrets
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import hashlib
import hmac

@dataclass
class SecurityConfig:
    """Security configuration parameters."""
    
    # Authentication settings
    enable_authentication: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    # Encryption settings
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation_iterations: int = 100000
    use_hardware_security: bool = False
    
    # Communication security
    require_https: bool = True
    min_tls_version: str = "TLSv1.2"
    verify_certificates: bool = True
    
    # Input validation
    max_input_size_bytes: int = 1048576  # 1MB
    allowed_file_extensions: list = field(default_factory=lambda: ['.json', '.yaml', '.txt'])
    sanitize_inputs: bool = True
    
    # Logging and monitoring
    log_security_events: bool = True
    log_file_path: str = "/var/log/echoloc_security.log"
    enable_intrusion_detection: bool = True
    
    # Quantum security settings
    quantum_key_length: int = 256
    quantum_entropy_threshold: float = 0.95
    secure_quantum_random: bool = True


class SecureConfigManager:
    """
    Secure configuration manager with environment variable support.
    
    Loads configuration from multiple sources with proper validation
    and secret management.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.secrets_cache: Dict[str, str] = {}
        self.security_config = SecurityConfig()
        
        # Load configuration
        self._load_configuration()
        self._validate_security_settings()
    
    def _load_configuration(self):
        """Load configuration from multiple sources."""
        # 1. Load from config file if provided
        if self.config_file and Path(self.config_file).exists():
            self._load_from_file(self.config_file)
        
        # 2. Override with environment variables
        self._load_from_environment()
        
        # 3. Apply security defaults
        self._apply_security_defaults()
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            self.config.update(file_config)
        except Exception as e:
            # Log error but don't fail - fall back to env vars
            pass
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # Database settings
            'DATABASE_URL': 'database.url',
            'DATABASE_USER': 'database.user',
            'DATABASE_PASSWORD': 'database.password',
            'DATABASE_NAME': 'database.name',
            
            # API settings
            'API_KEY': 'api.key',
            'API_SECRET': 'api.secret',
            'API_BASE_URL': 'api.base_url',
            
            # Security settings
            'SECRET_KEY': 'security.secret_key',
            'JWT_SECRET': 'security.jwt_secret',
            'ENCRYPTION_KEY': 'security.encryption_key',
            
            # Quantum settings
            'QUANTUM_BACKEND': 'quantum.backend',
            'QUANTUM_TOKEN': 'quantum.token',
            
            # System settings
            'LOG_LEVEL': 'logging.level',
            'DEBUG_MODE': 'system.debug',
            'ENVIRONMENT': 'system.environment',
            
            # Hardware settings
            'SENSOR_PORT': 'hardware.sensor_port',
            'ARDUINO_PORT': 'hardware.arduino_port',
            'GPIO_PINS': 'hardware.gpio_pins'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_config(config_key, value)
    
    def _set_nested_config(self, key_path: str, value: str):
        """Set nested configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        final_key = keys[-1]
        
        # Type conversion for common boolean and numeric values
        if value.lower() in ('true', 'false'):
            config[final_key] = value.lower() == 'true'
        elif value.isdigit():
            config[final_key] = int(value)
        elif self._is_float(value):
            config[final_key] = float(value)
        else:
            config[final_key] = value
    
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _apply_security_defaults(self):
        """Apply security defaults if not configured."""
        security_defaults = {
            'security': {
                'secret_key': self._generate_secret_key(),
                'jwt_secret': self._generate_secret_key(),
                'encryption_key': self._generate_encryption_key(),
                'enable_https': True,
                'session_timeout': 1800,  # 30 minutes
                'max_login_attempts': 5
            },
            'quantum': {
                'secure_random': True,
                'key_length': 256,
                'entropy_threshold': 0.95
            },
            'system': {
                'debug': False,
                'environment': 'production'
            }
        }
        
        for section, defaults in security_defaults.items():
            if section not in self.config:
                self.config[section] = {}
            
            for key, default_value in defaults.items():
                if key not in self.config[section]:
                    self.config[section][key] = default_value
    
    def _generate_secret_key(self) -> str:
        """Generate a cryptographically secure secret key."""
        return secrets.token_urlsafe(32)
    
    def _generate_encryption_key(self) -> str:
        """Generate a cryptographically secure encryption key."""
        return secrets.token_hex(32)
    
    def _validate_security_settings(self):
        """Validate security settings and warn about issues."""
        issues = []
        
        # Check for debug mode in production
        if (self.get('system.environment') == 'production' and 
            self.get('system.debug', False)):
            issues.append("Debug mode is enabled in production environment")
        
        # Check for HTTPS requirement
        if not self.get('security.enable_https', True):
            issues.append("HTTPS is not enforced")
        
        # Check secret key strength
        secret_key = self.get('security.secret_key', '')
        if len(secret_key) < 32:
            issues.append("Secret key is too short (should be at least 32 characters)")
        
        # Check session timeout
        session_timeout = self.get('security.session_timeout', 0)
        if session_timeout > 3600:  # 1 hour
            issues.append("Session timeout is very long (over 1 hour)")
        
        # Log security issues
        if issues:
            # In production, you would log these to a security log
            print(f"Security configuration warnings: {'; '.join(issues)}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_secret(self, key_path: str) -> Optional[str]:
        """Get a secret value (cached and secured)."""
        if key_path in self.secrets_cache:
            return self.secrets_cache[key_path]
        
        secret = self.get(key_path)
        if secret:
            # Cache the secret (in production, consider more secure caching)
            self.secrets_cache[key_path] = str(secret)
        
        return secret
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        self._set_nested_config(key_path, str(value))
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get('system.environment', 'development') == 'production'
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('system.debug', False) and not self.is_production()
    
    def get_database_url(self) -> Optional[str]:
        """Get database URL with secure credential handling."""
        # Check for full URL first
        db_url = self.get('database.url')
        if db_url:
            return db_url
        
        # Build URL from components
        user = self.get_secret('database.user')
        password = self.get_secret('database.password')
        host = self.get('database.host', 'localhost')
        port = self.get('database.port', 5432)
        name = self.get('database.name', 'echoloc')
        
        if user and password:
            return f"postgresql://{user}:{password}@{host}:{port}/{name}"
        
        return None
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration object."""
        # Update security config from loaded configuration
        security_section = self.config.get('security', {})
        
        self.security_config.enable_authentication = security_section.get(
            'enable_authentication', True
        )
        self.security_config.session_timeout_minutes = security_section.get(
            'session_timeout', 30
        ) // 60
        self.security_config.require_https = security_section.get(
            'enable_https', True
        )
        
        return self.security_config
    
    def validate_input(self, input_data: str, max_length: int = None) -> bool:
        """Validate input data for security."""
        if not isinstance(input_data, str):
            return False
        
        # Check length
        max_len = max_length or self.security_config.max_input_size_bytes
        if len(input_data.encode('utf-8')) > max_len:
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            '<script',
            'javascript:',
            'eval(',
            'exec(',
            '../',
            '..\\',
            '\x00'  # Null byte
        ]
        
        input_lower = input_data.lower()
        for pattern in dangerous_patterns:
            if pattern in input_lower:
                return False
        
        return True
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input data."""
        if not self.security_config.sanitize_inputs:
            return input_data
        
        # Basic sanitization
        sanitized = input_data.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Escape dangerous characters for JSON/XML contexts
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;')
        sanitized = sanitized.replace("'", '&#x27;')
        
        return sanitized


# Global configuration manager instance
_config_manager: Optional[SecureConfigManager] = None

def get_config() -> SecureConfigManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        config_file = os.getenv('ECHOLOC_CONFIG_FILE')
        _config_manager = SecureConfigManager(config_file)
    return _config_manager

def get_secret(key_path: str) -> Optional[str]:
    """Get a secret from configuration."""
    return get_config().get_secret(key_path)

def is_production() -> bool:
    """Check if running in production."""
    return get_config().is_production()

def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    return get_config().is_debug_enabled()

def validate_input(input_data: str, max_length: int = None) -> bool:
    """Validate input for security."""
    return get_config().validate_input(input_data, max_length)

def sanitize_input(input_data: str) -> str:
    """Sanitize input data."""
    return get_config().sanitize_input(input_data)