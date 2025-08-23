"""
Authentication and access control for EchoLoc-NN.
"""

from typing import Dict, List, Any, Optional, Tuple
import hashlib
import secrets
import time
import hmac
from dataclasses import dataclass
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class User:
    """User representation."""
    
    username: str
    password_hash: str
    roles: List[str]
    created_at: float
    last_login: Optional[float] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None


@dataclass
class Session:
    """User session representation."""
    
    session_id: str
    username: str
    created_at: float
    last_activity: float
    expires_at: float
    roles: List[str]
    
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        return time.time() < self.expires_at


class MultiFactorAuth:
    """Multi-factor authentication implementation."""
    
    def __init__(self):
        """Initialize MFA system."""
        self.totp_window = 30  # 30-second TOTP window
        logger.info("MultiFactorAuth initialized")
    
    def generate_secret(self) -> str:
        """Generate MFA secret for a user."""
        return secrets.token_urlsafe(32)
    
    def generate_totp_code(self, secret: str, timestamp: Optional[int] = None) -> str:
        """Generate TOTP code."""
        if timestamp is None:
            timestamp = int(time.time())
        
        # Simple TOTP implementation (not cryptographically secure for production)
        time_step = timestamp // self.totp_window
        combined = f"{secret}:{time_step}"
        
        hash_obj = hashlib.sha256(combined.encode())
        code = int(hash_obj.hexdigest()[:6], 16) % 1000000
        
        return f"{code:06d}"
    
    def verify_totp_code(self, secret: str, provided_code: str, timestamp: Optional[int] = None) -> bool:
        """Verify TOTP code."""
        if timestamp is None:
            timestamp = int(time.time())
        
        # Check current and previous time window for clock skew
        for offset in [0, -self.totp_window]:
            expected_code = self.generate_totp_code(secret, timestamp + offset)
            if hmac.compare_digest(expected_code, provided_code):
                return True
        
        return False
    
    def setup_user_mfa(self, username: str) -> Tuple[str, str]:
        """Setup MFA for a user."""
        secret = self.generate_secret()
        qr_url = f"otpauth://totp/EchoLoc-NN:{username}?secret={secret}&issuer=EchoLoc-NN"
        
        logger.info(f"MFA setup initiated for user: {username}")
        return secret, qr_url


class AccessController:
    """Role-based access control."""
    
    def __init__(self):
        """Initialize access controller."""
        self.permissions = {
            'admin': ['read', 'write', 'delete', 'manage_users', 'manage_system'],
            'user': ['read', 'write'],
            'readonly': ['read'],
            'guest': []
        }
        logger.info("AccessController initialized")
    
    def has_permission(self, user_roles: List[str], required_permission: str) -> bool:
        """Check if user has required permission."""
        for role in user_roles:
            if role in self.permissions and required_permission in self.permissions[role]:
                return True
        return False
    
    def add_role_permission(self, role: str, permission: str):
        """Add permission to a role."""
        if role not in self.permissions:
            self.permissions[role] = []
        
        if permission not in self.permissions[role]:
            self.permissions[role].append(permission)
            logger.info(f"Permission '{permission}' added to role '{role}'")
    
    def remove_role_permission(self, role: str, permission: str):
        """Remove permission from a role."""
        if role in self.permissions and permission in self.permissions[role]:
            self.permissions[role].remove(permission)
            logger.info(f"Permission '{permission}' removed from role '{role}'")
    
    def get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role."""
        return self.permissions.get(role, []).copy()


class SessionManager:
    """Manage user sessions."""
    
    def __init__(self, session_timeout: int = 3600):
        """
        Initialize session manager.
        
        Args:
            session_timeout: Session timeout in seconds (default: 1 hour)
        """
        self.session_timeout = session_timeout
        self.sessions = {}
        logger.info(f"SessionManager initialized (timeout: {session_timeout}s)")
    
    def create_session(self, username: str, user_roles: List[str]) -> Session:
        """Create a new session."""
        session_id = secrets.token_urlsafe(32)
        current_time = time.time()
        
        session = Session(
            session_id=session_id,
            username=username,
            created_at=current_time,
            last_activity=current_time,
            expires_at=current_time + self.session_timeout,
            roles=user_roles.copy()
        )
        
        self.sessions[session_id] = session
        logger.info(f"Session created for user: {username}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        
        if session is None:
            return None
        
        if not session.is_valid():
            self.invalidate_session(session_id)
            return None
        
        # Update last activity
        session.last_activity = time.time()
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        if session_id in self.sessions:
            username = self.sessions[session_id].username
            del self.sessions[session_id]
            logger.info(f"Session invalidated for user: {username}")
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if session.expires_at < current_time
        ]
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_active_sessions(self) -> List[Session]:
        """Get all active sessions."""
        return [s for s in self.sessions.values() if s.is_valid()]
    
    def extend_session(self, session_id: str, additional_time: int = None) -> bool:
        """Extend session timeout."""
        session = self.get_session(session_id)
        
        if session is None:
            return False
        
        if additional_time is None:
            additional_time = self.session_timeout
        
        session.expires_at = time.time() + additional_time
        logger.info(f"Session extended for user: {session.username}")
        return True