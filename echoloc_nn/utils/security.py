"""
Security utilities and input sanitization for EchoLoc-NN.
"""

import os
import re
import hashlib
import hmac
import secrets
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from pathlib import Path
from .exceptions import SecurityError
from .logging_config import get_logger


class InputSanitizer:
    """
    Input sanitization and validation for security.
    
    Prevents injection attacks, validates file paths,
    and sanitizes user inputs.
    """
    
    def __init__(self):
        self.logger = get_logger('input_sanitizer')
        
        # Allowed file extensions
        self.allowed_model_extensions = {'.pt', '.pth', '.ckpt', '.pkl'}
        self.allowed_config_extensions = {'.yaml', '.yml', '.json'}
        self.allowed_data_extensions = {'.npy', '.npz', '.csv', '.h5', '.hdf5'}
        
        # Dangerous patterns to reject
        self.dangerous_patterns = [
            r'\.\./',  # Directory traversal
            r'__[a-zA-Z_]+__',  # Python special methods
            r'eval\s*\(',  # eval() calls
            r'exec\s*\(',  # exec() calls
            r'import\s+',  # import statements
            r'subprocess',  # subprocess calls
            r'os\.',  # os module calls
            r'sys\.',  # sys module calls
            r'<script',  # XSS attempts
            r'javascript:',  # JavaScript injection
        ]
        
        self.pattern_regex = re.compile('|'.join(self.dangerous_patterns), re.IGNORECASE)
    
    def sanitize_file_path(
        self, 
        file_path: str, 
        allowed_extensions: Optional[set] = None,
        base_directory: Optional[str] = None
    ) -> str:
        """
        Sanitize and validate file path.
        
        Args:
            file_path: File path to sanitize
            allowed_extensions: Set of allowed file extensions
            base_directory: Base directory to restrict access to
            
        Returns:
            Sanitized file path
            
        Raises:
            SecurityError: If path is invalid or dangerous
        """
        if not isinstance(file_path, str):
            raise SecurityError(
                "File path must be string",
                error_code="INVALID_PATH_TYPE"
            )
        
        # Check for dangerous patterns
        if self.pattern_regex.search(file_path):
            raise SecurityError(
                "File path contains dangerous patterns",
                error_code="DANGEROUS_PATH_PATTERN",
                details={"path": file_path}
            )
        
        # Resolve and normalize path
        try:
            resolved_path = Path(file_path).resolve()
        except Exception as e:
            raise SecurityError(
                f"Invalid file path: {e}",
                error_code="PATH_RESOLUTION_ERROR",
                details={"path": file_path}
            )
        
        # Check if path exists and is a file
        if not resolved_path.exists():
            raise SecurityError(
                "File does not exist",
                error_code="FILE_NOT_FOUND",
                details={"path": str(resolved_path)}
            )
        
        if not resolved_path.is_file():
            raise SecurityError(
                "Path is not a file",
                error_code="NOT_A_FILE",
                details={"path": str(resolved_path)}
            )
        
        # Check base directory restriction
        if base_directory:
            base_path = Path(base_directory).resolve()
            try:
                resolved_path.relative_to(base_path)
            except ValueError:
                raise SecurityError(
                    f"File path outside allowed directory: {base_directory}",
                    error_code="PATH_OUTSIDE_BASE",
                    details={"path": str(resolved_path), "base": str(base_path)}
                )
        
        # Check file extension
        if allowed_extensions:
            if resolved_path.suffix.lower() not in allowed_extensions:
                raise SecurityError(
                    f"File extension not allowed: {resolved_path.suffix}",
                    error_code="INVALID_FILE_EXTENSION",
                    details={
                        "extension": resolved_path.suffix,
                        "allowed": list(allowed_extensions)
                    }
                )
        
        # Check file size (prevent massive files)
        file_size = resolved_path.stat().st_size
        max_size = 1024 * 1024 * 1024  # 1GB limit
        if file_size > max_size:
            raise SecurityError(
                f"File too large: {file_size} bytes",
                error_code="FILE_TOO_LARGE",
                details={"size": file_size, "max_size": max_size}
            )
        
        return str(resolved_path)
    
    def sanitize_model_path(self, model_path: str, base_directory: Optional[str] = None) -> str:
        """Sanitize model file path."""
        return self.sanitize_file_path(
            model_path, 
            self.allowed_model_extensions,
            base_directory
        )
    
    def sanitize_config_path(self, config_path: str, base_directory: Optional[str] = None) -> str:
        """Sanitize configuration file path."""
        return self.sanitize_file_path(
            config_path,
            self.allowed_config_extensions,
            base_directory
        )
    
    def sanitize_string_input(self, input_str: str, max_length: int = 1000) -> str:
        """
        Sanitize string input.
        
        Args:
            input_str: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If string is dangerous
        """
        if not isinstance(input_str, str):
            raise SecurityError(
                "Input must be string",
                error_code="INVALID_INPUT_TYPE"
            )
        
        # Check length
        if len(input_str) > max_length:
            raise SecurityError(
                f"String too long: {len(input_str)} > {max_length}",
                error_code="STRING_TOO_LONG",
                details={"length": len(input_str), "max_length": max_length}
            )
        
        # Check for dangerous patterns
        if self.pattern_regex.search(input_str):
            raise SecurityError(
                "String contains dangerous patterns",
                error_code="DANGEROUS_STRING_PATTERN",
                details={"input": input_str[:100]}  # Only log first 100 chars
            )
        
        # Remove null bytes and control characters
        sanitized = input_str.replace('\x00', '').replace('\r', '').replace('\n', ' ')
        
        # Limit to printable ASCII + common Unicode
        sanitized = ''.join(c for c in sanitized if c.isprintable() or c.isspace())
        
        return sanitized.strip()
    
    def validate_numpy_array(
        self,
        array: np.ndarray,
        max_size_mb: float = 100.0,
        allowed_dtypes: Optional[List[str]] = None
    ) -> bool:
        """
        Validate NumPy array for security.
        
        Args:
            array: NumPy array to validate
            max_size_mb: Maximum size in MB
            allowed_dtypes: List of allowed data types
            
        Returns:
            True if valid
            
        Raises:
            SecurityError: If array is invalid
        """
        if not isinstance(array, np.ndarray):
            raise SecurityError(
                "Input must be NumPy array",
                error_code="INVALID_ARRAY_TYPE"
            )
        
        # Check size
        size_mb = array.nbytes / 1024 / 1024
        if size_mb > max_size_mb:
            raise SecurityError(
                f"Array too large: {size_mb:.2f} MB > {max_size_mb} MB",
                error_code="ARRAY_TOO_LARGE",
                details={"size_mb": size_mb, "max_size_mb": max_size_mb}
            )
        
        # Check data type
        if allowed_dtypes:
            if str(array.dtype) not in allowed_dtypes:
                raise SecurityError(
                    f"Array dtype not allowed: {array.dtype}",
                    error_code="INVALID_ARRAY_DTYPE",
                    details={"dtype": str(array.dtype), "allowed": allowed_dtypes}
                )
        
        # Check for dangerous values
        if not np.all(np.isfinite(array)):
            raise SecurityError(
                "Array contains non-finite values",
                error_code="NON_FINITE_ARRAY_VALUES"
            )
        
        return True
    
    def validate_torch_tensor(
        self,
        tensor: torch.Tensor,
        max_size_mb: float = 100.0,
        allowed_dtypes: Optional[List[torch.dtype]] = None
    ) -> bool:
        """
        Validate PyTorch tensor for security.
        
        Args:
            tensor: PyTorch tensor to validate
            max_size_mb: Maximum size in MB
            allowed_dtypes: List of allowed data types
            
        Returns:
            True if valid
            
        Raises:
            SecurityError: If tensor is invalid
        """
        if not isinstance(tensor, torch.Tensor):
            raise SecurityError(
                "Input must be PyTorch tensor",
                error_code="INVALID_TENSOR_TYPE"
            )
        
        # Check size
        size_mb = tensor.element_size() * tensor.numel() / 1024 / 1024
        if size_mb > max_size_mb:
            raise SecurityError(
                f"Tensor too large: {size_mb:.2f} MB > {max_size_mb} MB",
                error_code="TENSOR_TOO_LARGE",
                details={"size_mb": size_mb, "max_size_mb": max_size_mb}
            )
        
        # Check data type
        if allowed_dtypes:
            if tensor.dtype not in allowed_dtypes:
                raise SecurityError(
                    f"Tensor dtype not allowed: {tensor.dtype}",
                    error_code="INVALID_TENSOR_DTYPE",
                    details={"dtype": str(tensor.dtype), "allowed": [str(dt) for dt in allowed_dtypes]}
                )
        
        # Check for dangerous values
        if not torch.all(torch.isfinite(tensor)):
            raise SecurityError(
                "Tensor contains non-finite values",
                error_code="NON_FINITE_TENSOR_VALUES"
            )
        
        return True


class SecurityValidator:
    """
    Security validation and authentication utilities.
    
    Provides secure configuration validation, API key management,
    and data integrity checks.
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        self.logger = get_logger('security_validator')
        
        # Generate or use provided secret key
        if secret_key:
            self.secret_key = secret_key.encode('utf-8')
        else:
            self.secret_key = secrets.token_bytes(32)
        
        self.sanitizer = InputSanitizer()
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def hash_data(self, data: bytes) -> str:
        """Generate SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()
    
    def sign_data(self, data: bytes) -> str:
        """Generate HMAC signature for data."""
        return hmac.new(
            self.secret_key,
            data,
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, data: bytes, signature: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.sign_data(data)
        return hmac.compare_digest(expected_signature, signature)
    
    def validate_model_integrity(self, model_path: str) -> Dict[str, Any]:
        """
        Validate model file integrity.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Integrity validation results
        """
        # Sanitize path first
        safe_path = self.sanitizer.sanitize_model_path(model_path)
        
        # Read file and compute hash
        with open(safe_path, 'rb') as f:
            file_data = f.read()
        
        file_hash = self.hash_data(file_data)
        file_size = len(file_data)
        
        # Basic PyTorch model validation
        is_pytorch_model = False
        try:
            # Try to load as PyTorch model (read-only check)
            checkpoint = torch.load(safe_path, map_location='cpu')
            is_pytorch_model = True
            
            # Check for suspicious content
            suspicious_keys = [
                'eval', 'exec', '__import__', 'subprocess', 'os', 'sys'
            ]
            
            def check_dict_recursively(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if any(sus in str(key).lower() for sus in suspicious_keys):
                            return f"Suspicious key found: {path}.{key}"
                        result = check_dict_recursively(value, f"{path}.{key}")
                        if result:
                            return result
                elif isinstance(obj, str):
                    if any(sus in obj.lower() for sus in suspicious_keys):
                        return f"Suspicious string content at {path}"
                return None
            
            suspicious_content = check_dict_recursively(checkpoint)
            
        except Exception as e:
            self.logger.warning(f"Could not validate model structure: {e}")
            suspicious_content = f"Model loading error: {e}"
        
        return {
            'file_path': safe_path,
            'file_hash': file_hash,
            'file_size': file_size,
            'is_pytorch_model': is_pytorch_model,
            'suspicious_content': suspicious_content,
            'is_safe': suspicious_content is None and is_pytorch_model
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration for security issues.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check for dangerous configuration values
        def check_value_recursively(obj, key_path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_value_recursively(value, f"{key_path}.{key}" if key_path else key)
            elif isinstance(obj, str):
                # Check for file paths
                if ('path' in key_path.lower() or 'file' in key_path.lower()):
                    try:
                        self.sanitizer.sanitize_string_input(obj)
                    except SecurityError as e:
                        issues.append(f"Dangerous path in {key_path}: {e}")
                
                # Check for command injection
                dangerous_chars = ['&', '|', ';', '`', '$', '(', ')']
                if any(char in obj for char in dangerous_chars):
                    issues.append(f"Potentially dangerous characters in {key_path}")
            
            elif isinstance(obj, (int, float)):
                # Check for unreasonable values
                if abs(obj) > 1e10:
                    issues.append(f"Extremely large numeric value in {key_path}: {obj}")
        
        check_value_recursively(config)
        
        # Check specific security-relevant configurations
        if 'log_dir' in config:
            try:
                self.sanitizer.sanitize_file_path(config['log_dir'])
            except SecurityError as e:
                issues.append(f"Invalid log directory: {e}")
        
        if 'checkpoint_dir' in config:
            try:
                self.sanitizer.sanitize_file_path(config['checkpoint_dir'])
            except SecurityError as e:
                issues.append(f"Invalid checkpoint directory: {e}")
        
        return {
            'is_safe': len(issues) == 0,
            'issues': issues,
            'config_hash': self.hash_data(str(config).encode())
        }
    
    def create_secure_environment_config(self) -> Dict[str, Any]:
        """Create secure default environment configuration."""
        return {
            'security': {
                'input_validation': True,
                'file_path_sanitization': True,
                'max_file_size_mb': 1024,  # 1GB
                'max_array_size_mb': 100,  # 100MB
                'allowed_model_extensions': list(self.sanitizer.allowed_model_extensions),
                'enable_model_signature_verification': True,
                'log_security_events': True
            },
            'resource_limits': {
                'max_memory_gb': 8,
                'max_gpu_memory_gb': 4,
                'max_inference_time_ms': 1000,
                'max_batch_size': 64
            },
            'logging': {
                'level': 'INFO',
                'enable_audit_log': True,
                'sanitize_log_data': True
            }
        }
    
    def audit_log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None
    ):
        """
        Log security-relevant events for auditing.
        
        Args:
            event_type: Type of security event
            details: Event details
            user_id: Optional user identifier
            source_ip: Optional source IP address
        """
        audit_entry = {
            'event_type': event_type,
            'timestamp': time.time(),
            'details': details,
            'user_id': user_id,
            'source_ip': source_ip,
            'event_hash': self.hash_data(str(details).encode())
        }
        
        self.logger.info(
            f"Security audit: {event_type}",
            extra={'audit_entry': audit_entry}
        )


class SecureConfigLoader:
    """
    Secure configuration file loader with validation.
    """
    
    def __init__(self, validator: SecurityValidator):
        self.validator = validator
        self.logger = get_logger('secure_config_loader')
    
    def load_config(
        self,
        config_path: str,
        schema: Optional[Dict[str, Any]] = None,
        base_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Securely load and validate configuration file.
        
        Args:
            config_path: Path to configuration file
            schema: Optional schema for validation
            base_directory: Base directory restriction
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            SecurityError: If configuration is invalid or dangerous
        """
        # Sanitize file path
        safe_path = self.validator.sanitizer.sanitize_config_path(
            config_path, base_directory
        )
        
        # Load configuration
        try:
            if safe_path.endswith('.yaml') or safe_path.endswith('.yml'):
                import yaml
                with open(safe_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif safe_path.endswith('.json'):
                import json
                with open(safe_path, 'r') as f:
                    config = json.load(f)
            else:
                raise SecurityError(
                    f"Unsupported config format: {safe_path}",
                    error_code="UNSUPPORTED_CONFIG_FORMAT"
                )
        except Exception as e:
            raise SecurityError(
                f"Failed to load configuration: {e}",
                error_code="CONFIG_LOAD_ERROR",
                details={"path": safe_path}
            )
        
        # Validate configuration security
        validation_result = self.validator.validate_configuration(config)
        if not validation_result['is_safe']:
            raise SecurityError(
                "Configuration contains security issues",
                error_code="UNSAFE_CONFIGURATION",
                details={"issues": validation_result['issues']}
            )
        
        # Schema validation if provided
        if schema:
            self._validate_schema(config, schema)
        
        # Log successful load
        self.validator.audit_log_event(
            'config_loaded',
            {'path': safe_path, 'config_hash': validation_result['config_hash']}
        )
        
        return config
    
    def _validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]):
        """Validate configuration against schema."""
        # Simple schema validation - could be enhanced with jsonschema
        def validate_recursive(obj, schema_obj, path=""):
            if isinstance(schema_obj, dict):
                if 'type' in schema_obj:
                    expected_type = schema_obj['type']
                    if expected_type == 'int' and not isinstance(obj, int):
                        raise SecurityError(f"Invalid type at {path}: expected int")
                    elif expected_type == 'float' and not isinstance(obj, (int, float)):
                        raise SecurityError(f"Invalid type at {path}: expected float")
                    elif expected_type == 'str' and not isinstance(obj, str):
                        raise SecurityError(f"Invalid type at {path}: expected string")
                    elif expected_type == 'bool' and not isinstance(obj, bool):
                        raise SecurityError(f"Invalid type at {path}: expected bool")
                
                if 'required' in schema_obj:
                    for required_key in schema_obj['required']:
                        if required_key not in obj:
                            raise SecurityError(f"Missing required key: {path}.{required_key}")
        
        validate_recursive(config, schema)