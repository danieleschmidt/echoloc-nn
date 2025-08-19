"""
Internationalization (i18n) support for EchoLoc-NN.

Provides multi-language support for user-facing messages, errors,
and documentation with fallback to English.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

class I18nManager:
    """
    Internationalization manager for multi-language support.
    
    Supports: English, Spanish, French, German, Japanese, Chinese
    """
    
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Español', 
        'fr': 'Français',
        'de': 'Deutsch',
        'ja': '日本語',
        'zh': '中文'
    }
    
    # Default messages in English
    DEFAULT_MESSAGES = {
        'system': {
            'initializing': 'Initializing EchoLoc-NN system...',
            'ready': 'System ready for operation',
            'shutting_down': 'Shutting down system',
            'error_occurred': 'An error occurred',
            'operation_completed': 'Operation completed successfully'
        },
        'localization': {
            'starting_inference': 'Starting localization inference',
            'position_estimated': 'Position estimated',
            'confidence_low': 'Low confidence in position estimate',
            'inference_failed': 'Localization inference failed',
            'using_fallback': 'Using fallback position'
        },
        'performance': {
            'cache_hit': 'Cache hit - using cached result',
            'cache_miss': 'Cache miss - computing new result',
            'high_latency': 'High inference latency detected',
            'memory_warning': 'High memory usage warning',
            'scaling_up': 'Scaling up worker processes',
            'scaling_down': 'Scaling down worker processes'
        },
        'errors': {
            'invalid_input': 'Invalid input data provided',
            'model_load_failed': 'Failed to load model',
            'hardware_error': 'Hardware communication error',
            'timeout_error': 'Operation timed out',
            'circuit_breaker_open': 'Circuit breaker is open',
            'validation_failed': 'Input validation failed'
        },
        'health': {
            'status_healthy': 'System status: Healthy',
            'status_degraded': 'System status: Degraded',
            'status_critical': 'System status: Critical',
            'performance_good': 'Performance metrics within normal range',
            'performance_poor': 'Performance metrics below acceptable threshold'
        },
        'compliance': {
            'gdpr_notice': 'Data processing complies with GDPR requirements',
            'ccpa_notice': 'Data processing complies with CCPA requirements', 
            'data_retention': 'Data retention policy applied',
            'privacy_preserved': 'User privacy has been preserved'
        }
    }
    
    # Translations for each supported language
    TRANSLATIONS = {
        'es': {  # Spanish
            'system': {
                'initializing': 'Inicializando sistema EchoLoc-NN...',
                'ready': 'Sistema listo para operar',
                'shutting_down': 'Cerrando sistema',
                'error_occurred': 'Ocurrió un error',
                'operation_completed': 'Operación completada exitosamente'
            },
            'localization': {
                'starting_inference': 'Iniciando inferencia de localización',
                'position_estimated': 'Posición estimada',
                'confidence_low': 'Baja confianza en estimación de posición',
                'inference_failed': 'Falló la inferencia de localización',
                'using_fallback': 'Usando posición de respaldo'
            },
            'errors': {
                'invalid_input': 'Datos de entrada inválidos',
                'model_load_failed': 'Error al cargar modelo',
                'hardware_error': 'Error de comunicación de hardware',
                'timeout_error': 'Operación expiró',
                'circuit_breaker_open': 'Interruptor de circuito está abierto',
                'validation_failed': 'Validación de entrada falló'
            }
        },
        'fr': {  # French
            'system': {
                'initializing': 'Initialisation du système EchoLoc-NN...',
                'ready': 'Système prêt pour fonctionnement',
                'shutting_down': 'Arrêt du système',
                'error_occurred': 'Une erreur s\'est produite',
                'operation_completed': 'Opération terminée avec succès'
            },
            'localization': {
                'starting_inference': 'Démarrage de l\'inférence de localisation',
                'position_estimated': 'Position estimée',
                'confidence_low': 'Faible confiance dans l\'estimation de position',
                'inference_failed': 'Échec de l\'inférence de localisation',
                'using_fallback': 'Utilisation de la position de secours'
            },
            'errors': {
                'invalid_input': 'Données d\'entrée invalides',
                'model_load_failed': 'Échec du chargement du modèle',
                'hardware_error': 'Erreur de communication matérielle',
                'timeout_error': 'Opération expirée',
                'circuit_breaker_open': 'Le disjoncteur est ouvert',
                'validation_failed': 'Échec de la validation d\'entrée'
            }
        },
        'de': {  # German
            'system': {
                'initializing': 'EchoLoc-NN System wird initialisiert...',
                'ready': 'System bereit für Betrieb',
                'shutting_down': 'System wird heruntergefahren',
                'error_occurred': 'Ein Fehler ist aufgetreten',
                'operation_completed': 'Operation erfolgreich abgeschlossen'
            },
            'localization': {
                'starting_inference': 'Lokalisierungsinferenz wird gestartet',
                'position_estimated': 'Position geschätzt',
                'confidence_low': 'Geringe Konfidenz in Positionsschätzung',
                'inference_failed': 'Lokalisierungsinferenz fehlgeschlagen',
                'using_fallback': 'Verwende Fallback-Position'
            }
        },
        'ja': {  # Japanese
            'system': {
                'initializing': 'EchoLoc-NNシステムを初期化中...',
                'ready': 'システム動作準備完了',
                'shutting_down': 'システムをシャットダウン中',
                'error_occurred': 'エラーが発生しました',
                'operation_completed': '操作が正常に完了しました'
            },
            'localization': {
                'starting_inference': '位置推定推論を開始',
                'position_estimated': '位置が推定されました',
                'confidence_low': '位置推定の信頼度が低い',
                'inference_failed': '位置推定推論に失敗',
                'using_fallback': 'フォールバック位置を使用'
            }
        },
        'zh': {  # Chinese (Simplified)
            'system': {
                'initializing': '正在初始化EchoLoc-NN系统...',
                'ready': '系统已准备好运行',
                'shutting_down': '正在关闭系统',
                'error_occurred': '发生错误',
                'operation_completed': '操作成功完成'
            },
            'localization': {
                'starting_inference': '开始定位推理',
                'position_estimated': '位置已估算',
                'confidence_low': '位置估算置信度低',
                'inference_failed': '定位推理失败',
                'using_fallback': '使用后备位置'
            }
        }
    }
    
    def __init__(self, language: str = 'en'):
        """Initialize i18n manager with specified language."""
        self.current_language = language if language in self.SUPPORTED_LANGUAGES else 'en'
        
    def set_language(self, language: str) -> bool:
        """Set current language. Returns True if successful."""
        if language in self.SUPPORTED_LANGUAGES:
            self.current_language = language
            return True
        return False
    
    def get_message(self, category: str, key: str, **kwargs) -> str:
        """
        Get localized message.
        
        Args:
            category: Message category ('system', 'localization', etc.)
            key: Message key within category
            **kwargs: Format parameters for the message
            
        Returns:
            Localized message string
        """
        # Try current language first
        if (self.current_language in self.TRANSLATIONS and
            category in self.TRANSLATIONS[self.current_language] and
            key in self.TRANSLATIONS[self.current_language][category]):
            message = self.TRANSLATIONS[self.current_language][category][key]
        # Fall back to English
        elif (category in self.DEFAULT_MESSAGES and
              key in self.DEFAULT_MESSAGES[category]):
            message = self.DEFAULT_MESSAGES[category][key]
        # Last resort
        else:
            message = f"[Missing translation: {category}.{key}]"
        
        # Format with provided kwargs
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError):
            return message
    
    def get_language_info(self) -> Dict[str, str]:
        """Get information about current language."""
        return {
            'code': self.current_language,
            'name': self.SUPPORTED_LANGUAGES[self.current_language],
            'available_languages': list(self.SUPPORTED_LANGUAGES.keys())
        }
    
    def get_all_messages(self) -> Dict[str, Any]:
        """Get all messages in current language."""
        if self.current_language == 'en':
            return self.DEFAULT_MESSAGES
        
        # Merge translations with defaults as fallback
        result = {}
        for category in self.DEFAULT_MESSAGES:
            result[category] = self.DEFAULT_MESSAGES[category].copy()
            if (self.current_language in self.TRANSLATIONS and
                category in self.TRANSLATIONS[self.current_language]):
                result[category].update(self.TRANSLATIONS[self.current_language][category])
        
        return result


# Global i18n instance
_i18n = I18nManager()

def get_message(category: str, key: str, **kwargs) -> str:
    """Get localized message using global i18n instance."""
    return _i18n.get_message(category, key, **kwargs)

def set_language(language: str) -> bool:
    """Set global language."""
    return _i18n.set_language(language)

def get_current_language() -> str:
    """Get current language code."""
    return _i18n.current_language

def get_supported_languages() -> Dict[str, str]:
    """Get all supported languages."""
    return I18nManager.SUPPORTED_LANGUAGES.copy()