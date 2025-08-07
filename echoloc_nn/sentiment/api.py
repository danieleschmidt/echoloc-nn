"""
REST API for spatial-aware sentiment analysis.

Provides HTTP endpoints for real-time sentiment analysis with 
EchoLoc ultrasonic positioning integration.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
import asyncio
import time
import json
import hashlib
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

from .models import SpatialSentimentAnalyzer, MultiModalSentimentModel
from .spatial_fusion import LocationContextSentiment, LocationContext, SentimentLocation
from .real_time import StreamingSentimentAnalyzer, StreamingConfig
from .multi_modal import MultiModalSentimentProcessor, MultiModalInput, AudioSpatialSentiment
from ..utils.error_handling import handle_errors, EchoLocError
from ..utils.validation import validate_input
from ..utils.security import SecurityManager, APIKeyManager, RateLimiter
from ..utils.monitoring import PerformanceMonitor, MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class SentimentRequest(BaseModel):
    \"\"\"Request model for sentiment analysis.\"\"\"
    text: str = Field(..., min_length=1, max_length=10000, description=\"Text to analyze\")
    include_spatial: bool = Field(False, description=\"Include spatial context\")
    spatial_context: Optional[List[float]] = Field(None, description=\"9D spatial vector [x,y,z,vx,vy,vz,ax,ay,az]\")
    zone: Optional[str] = Field(None, description=\"Spatial zone identifier\")
    priority: bool = Field(False, description=\"High priority processing\")
    
    @validator('spatial_context')
    def validate_spatial_context(cls, v):
        if v is not None:
            if len(v) != 9:
                raise ValueError('Spatial context must have exactly 9 values')
            if any(abs(x) > 1000 for x in v):  # Reasonable bounds check
                raise ValueError('Spatial context values out of reasonable range')
        return v
        
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        # Basic content filtering
        forbidden = ['<script', 'javascript:', 'data:']
        if any(word in v.lower() for word in forbidden):
            raise ValueError('Text contains forbidden content')
        return v.strip()

class MultiModalRequest(BaseModel):
    \"\"\"Request model for multi-modal sentiment analysis.\"\"\"
    text: Optional[str] = Field(None, min_length=1, max_length=10000)
    audio_base64: Optional[str] = Field(None, description=\"Base64 encoded audio data\")
    spatial_context: Optional[List[float]] = Field(None, description=\"9D spatial vector\")
    zone: Optional[str] = Field(None)
    modalities: List[str] = Field([\"text\"], description=\"Modalities to use\")
    
    @validator('modalities')
    def validate_modalities(cls, v):
        allowed = ['text', 'audio', 'spatial']
        if not all(mod in allowed for mod in v):
            raise ValueError(f'Invalid modalities. Allowed: {allowed}')
        if len(v) == 0:
            raise ValueError('At least one modality must be specified')
        return v

class SentimentResponse(BaseModel):
    \"\"\"Response model for sentiment analysis.\"\"\"
    sentiment: str = Field(..., description=\"Predicted sentiment class\")
    confidence: float = Field(..., ge=0, le=1, description=\"Prediction confidence\")
    probabilities: Dict[str, float] = Field(..., description=\"Class probabilities\")
    spatial_influence: Optional[float] = Field(None, description=\"Spatial influence on prediction\")
    zone: Optional[str] = Field(None, description=\"Spatial zone\")
    processing_time_ms: float = Field(..., description=\"Processing time in milliseconds\")
    timestamp: str = Field(..., description=\"ISO timestamp\")
    request_id: str = Field(..., description=\"Unique request identifier\")

class StreamingResponse(BaseModel):
    \"\"\"Response model for streaming sentiment analysis.\"\"\"
    status: str = Field(..., description=\"Streaming status\")
    metrics: Dict[str, Any] = Field(..., description=\"Performance metrics\")
    recent_results: List[Dict[str, Any]] = Field([], description=\"Recent analysis results\")

class HealthResponse(BaseModel):
    \"\"\"Health check response.\"\"\"
    status: str
    version: str
    uptime_seconds: float
    models_loaded: Dict[str, bool]
    memory_usage_mb: float
    active_streams: int

# Global state
app_state = {
    \"spatial_sentiment_model\": None,
    \"multimodal_model\": None,
    \"location_manager\": None,
    \"streaming_analyzer\": None,
    \"multimodal_processor\": None,
    \"security_manager\": None,
    \"api_key_manager\": None,
    \"rate_limiter\": None,
    \"metrics_collector\": None,
    \"start_time\": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    \"\"\"Application startup and shutdown handlers.\"\"\"
    
    # Startup
    logger.info(\"Starting EchoLoc Sentiment Analysis API\")
    
    try:
        # Initialize security components
        app_state[\"security_manager\"] = SecurityManager()
        app_state[\"api_key_manager\"] = APIKeyManager()
        app_state[\"rate_limiter\"] = RateLimiter(requests_per_minute=60)
        app_state[\"metrics_collector\"] = MetricsCollector()
        
        # Initialize models (would load from files in production)
        app_state[\"spatial_sentiment_model\"] = SpatialSentimentAnalyzer()
        app_state[\"multimodal_model\"] = MultiModalSentimentModel()
        app_state[\"location_manager\"] = LocationContextSentiment()
        
        # Initialize processors
        app_state[\"multimodal_processor\"] = MultiModalSentimentProcessor(
            app_state[\"multimodal_model\"],
            app_state[\"location_manager\"]
        )
        
        # Initialize streaming (optional)
        streaming_config = StreamingConfig(update_frequency_hz=5.0)
        app_state[\"streaming_analyzer\"] = StreamingSentimentAnalyzer(
            app_state[\"spatial_sentiment_model\"],
            app_state[\"location_manager\"],
            streaming_config
        )
        
        logger.info(\"✓ All models and services initialized\")
        
    except Exception as e:
        logger.error(f\"Failed to initialize application: {e}\")
        raise
        
    yield
    
    # Shutdown
    logger.info(\"Shutting down EchoLoc Sentiment Analysis API\")
    
    if app_state[\"streaming_analyzer\"] and app_state[\"streaming_analyzer\"].is_streaming:
        app_state[\"streaming_analyzer\"].stop_streaming()
        
    logger.info(\"✓ Shutdown complete\")

# Create FastAPI app
app = FastAPI(
    title=\"EchoLoc Sentiment Analysis API\",
    description=\"Spatial-aware sentiment analysis with ultrasonic localization\",
    version=\"1.0.0\",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[\"*\"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer(auto_error=False)

def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    \"\"\"Validate API key.\"\"\"
    if not credentials:
        raise HTTPException(status_code=401, detail=\"API key required\")
        
    api_key_manager = app_state.get(\"api_key_manager\")
    if not api_key_manager or not api_key_manager.validate_key(credentials.credentials):
        raise HTTPException(status_code=401, detail=\"Invalid API key\")
        
    return credentials.credentials

def check_rate_limit(api_key: str = Depends(get_api_key)):
    \"\"\"Check rate limiting.\"\"\"
    rate_limiter = app_state.get(\"rate_limiter\")
    if rate_limiter and not rate_limiter.is_allowed(api_key):
        raise HTTPException(status_code=429, detail=\"Rate limit exceeded\")
    return True

# Routes
@app.get(\"/health\", response_model=HealthResponse)
async def health_check():
    \"\"\"Health check endpoint.\"\"\"
    
    import psutil
    
    models_loaded = {
        \"spatial_sentiment\": app_state.get(\"spatial_sentiment_model\") is not None,
        \"multimodal\": app_state.get(\"multimodal_model\") is not None,
        \"location_manager\": app_state.get(\"location_manager\") is not None
    }
    
    return HealthResponse(
        status=\"healthy\" if all(models_loaded.values()) else \"degraded\",
        version=\"1.0.0\",
        uptime_seconds=time.time() - app_state[\"start_time\"],
        models_loaded=models_loaded,
        memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
        active_streams=1 if app_state.get(\"streaming_analyzer\") and 
                          app_state[\"streaming_analyzer\"].is_streaming else 0
    )

@app.post(\"/analyze\", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(check_rate_limit)
):
    \"\"\"Analyze sentiment with optional spatial context.\"\"\"
    
    start_time = time.time()
    request_id = hashlib.md5(f\"{request.text}{time.time()}\".encode()).hexdigest()[:8]
    
    try:
        model = app_state.get(\"spatial_sentiment_model\")
        if not model:
            raise HTTPException(status_code=503, detail=\"Sentiment model not available\")
            
        # Prepare spatial context
        spatial_context = None
        if request.spatial_context:
            spatial_context = np.array(request.spatial_context)
            
        # Run analysis
        with app_state[\"metrics_collector\"].measure(\"sentiment_analysis\"):
            result = model.predict_sentiment(
                text=request.text,
                spatial_context=spatial_context
            )
            
        # Process results
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Convert probabilities to named format
        class_names = [\"negative\", \"neutral\", \"positive\"]
        if isinstance(result[\"probabilities\"], np.ndarray):
            probs = result[\"probabilities\"]
        else:
            probs = [result[\"probabilities\"]]
            
        prob_dict = {name: float(prob) for name, prob in zip(class_names, probs)}
        
        response = SentimentResponse(
            sentiment=result[\"sentiment\"],
            confidence=float(result[\"confidence\"]),
            probabilities=prob_dict,
            spatial_influence=float(result.get(\"spatial_influence\", 0.0)) if request.include_spatial else None,
            zone=request.zone,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow().isoformat() + \"Z\",
            request_id=request_id
        )
        
        # Log metrics asynchronously
        background_tasks.add_task(
            log_request_metrics,
            \"sentiment_analysis\",
            processing_time_ms,
            result[\"sentiment\"],
            float(result[\"confidence\"])
        )
        
        return response
        
    except Exception as e:
        logger.error(f\"Error in sentiment analysis: {e}\")
        raise HTTPException(status_code=500, detail=f\"Analysis failed: {str(e)}\")

@app.post(\"/analyze/multimodal\")
async def analyze_multimodal(
    request: MultiModalRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(check_rate_limit)
):
    \"\"\"Multi-modal sentiment analysis.\"\"\"
    
    start_time = time.time()
    request_id = hashlib.md5(f\"{request.text or 'audio'}{time.time()}\".encode()).hexdigest()[:8]
    
    try:
        processor = app_state.get(\"multimodal_processor\")
        if not processor:
            raise HTTPException(status_code=503, detail=\"Multi-modal processor not available\")
            
        # Prepare input
        input_data = MultiModalInput(
            text=request.text,
            spatial_context=np.array(request.spatial_context) if request.spatial_context else None,
            timestamp=time.time()
        )
        
        # Handle audio data
        if request.audio_base64 and \"audio\" in request.modalities:
            import base64
            import io
            import soundfile as sf
            
            try:
                audio_bytes = base64.b64decode(request.audio_base64)
                audio_buffer = io.BytesIO(audio_bytes)
                audio_data, sample_rate = sf.read(audio_buffer)
                input_data.audio_data = audio_data
            except Exception as e:
                logger.warning(f\"Failed to decode audio: {e}\")
                
        # Run analysis
        with app_state[\"metrics_collector\"].measure(\"multimodal_analysis\"):
            results = await processor.analyze_multi_modal(input_data)
            
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Format response
        response = {
            \"request_id\": request_id,
            \"timestamp\": datetime.utcnow().isoformat() + \"Z\",
            \"processing_time_ms\": processing_time_ms,
            \"modalities_used\": results[\"modalities_used\"],
            \"results\": results,
            \"status\": \"success\"
        }
        
        # Log metrics
        background_tasks.add_task(
            log_request_metrics,
            \"multimodal_analysis\",
            processing_time_ms,
            results.get(\"fused_analysis\", {}).get(\"sentiment\", \"unknown\"),
            results.get(\"fused_analysis\", {}).get(\"confidence\", 0.0)
        )
        
        return response
        
    except Exception as e:
        logger.error(f\"Error in multi-modal analysis: {e}\")
        raise HTTPException(status_code=500, detail=f\"Multi-modal analysis failed: {str(e)}\")

@app.post(\"/streaming/start\")
async def start_streaming(
    _: bool = Depends(check_rate_limit)
):
    \"\"\"Start real-time streaming sentiment analysis.\"\"\"
    
    try:
        analyzer = app_state.get(\"streaming_analyzer\")
        if not analyzer:
            raise HTTPException(status_code=503, detail=\"Streaming analyzer not available\")
            
        if analyzer.is_streaming:
            raise HTTPException(status_code=400, detail=\"Streaming already active\")
            
        analyzer.start_streaming()
        
        return {\"status\": \"started\", \"message\": \"Real-time streaming activated\"}
        
    except Exception as e:
        logger.error(f\"Error starting streaming: {e}\")
        raise HTTPException(status_code=500, detail=f\"Failed to start streaming: {str(e)}\")

@app.post(\"/streaming/stop\")
async def stop_streaming(
    _: bool = Depends(check_rate_limit)
):
    \"\"\"Stop real-time streaming sentiment analysis.\"\"\"
    
    try:
        analyzer = app_state.get(\"streaming_analyzer\")
        if not analyzer:
            raise HTTPException(status_code=503, detail=\"Streaming analyzer not available\")
            
        if not analyzer.is_streaming:
            raise HTTPException(status_code=400, detail=\"Streaming not active\")
            
        analyzer.stop_streaming()
        
        return {\"status\": \"stopped\", \"message\": \"Real-time streaming deactivated\"}
        
    except Exception as e:
        logger.error(f\"Error stopping streaming: {e}\")
        raise HTTPException(status_code=500, detail=f\"Failed to stop streaming: {str(e)}\")

@app.get(\"/streaming/status\", response_model=StreamingResponse)
async def streaming_status(
    _: bool = Depends(check_rate_limit)
):
    \"\"\"Get streaming status and metrics.\"\"\"
    
    try:
        analyzer = app_state.get(\"streaming_analyzer\")
        if not analyzer:
            raise HTTPException(status_code=503, detail=\"Streaming analyzer not available\")
            
        metrics = analyzer.get_metrics()
        recent_results = analyzer.get_results(max_results=10)
        
        # Convert results to serializable format
        serializable_results = []
        for result in recent_results:
            serializable_results.append({
                \"sentiment\": result.sentiment,
                \"confidence\": float(result.confidence),
                \"text\": result.text[:100] + \"...\" if len(result.text) > 100 else result.text,
                \"timestamp\": result.timestamp,
                \"spatial_influence\": float(result.spatial_influence)
            })
            
        return StreamingResponse(
            status=\"active\" if analyzer.is_streaming else \"inactive\",
            metrics={
                \"total_predictions\": metrics.total_predictions,
                \"avg_latency_ms\": metrics.avg_latency_ms,
                \"predictions_per_second\": metrics.predictions_per_second,
                \"sentiment_distribution\": metrics.sentiment_distribution,
                \"confidence_distribution\": metrics.confidence_distribution
            },
            recent_results=serializable_results
        )
        
    except Exception as e:
        logger.error(f\"Error getting streaming status: {e}\")
        raise HTTPException(status_code=500, detail=f\"Failed to get status: {str(e)}\")

@app.post(\"/streaming/analyze\")
async def streaming_analyze(
    request: SentimentRequest,
    _: bool = Depends(check_rate_limit)
):
    \"\"\"Add text to streaming analysis queue.\"\"\"
    
    try:
        analyzer = app_state.get(\"streaming_analyzer\")
        if not analyzer:
            raise HTTPException(status_code=503, detail=\"Streaming analyzer not available\")
            
        if not analyzer.is_streaming:
            raise HTTPException(status_code=400, detail=\"Streaming not active\")
            
        # Queue text for analysis
        result = analyzer.analyze_text(request.text, priority=request.priority)
        
        if result:
            # Direct analysis (not streaming mode)
            return {
                \"status\": \"analyzed\",
                \"result\": {
                    \"sentiment\": result.sentiment,
                    \"confidence\": float(result.confidence),
                    \"timestamp\": result.timestamp
                }
            }
        else:
            # Queued for streaming analysis
            return {
                \"status\": \"queued\",
                \"message\": \"Text queued for streaming analysis\"
            }
            
    except Exception as e:
        logger.error(f\"Error in streaming analyze: {e}\")
        raise HTTPException(status_code=500, detail=f\"Streaming analysis failed: {str(e)}\")

@app.get(\"/metrics\")
async def get_metrics(
    _: bool = Depends(check_rate_limit)
):
    \"\"\"Get API performance metrics.\"\"\"
    
    try:
        collector = app_state.get(\"metrics_collector\")
        if not collector:
            return {\"metrics\": \"not available\"}
            
        metrics = collector.get_metrics()
        
        return {
            \"api_metrics\": metrics,
            \"uptime_seconds\": time.time() - app_state[\"start_time\"],
            \"models_loaded\": {
                \"spatial_sentiment\": app_state.get(\"spatial_sentiment_model\") is not None,
                \"multimodal\": app_state.get(\"multimodal_model\") is not None
            }
        }
        
    except Exception as e:
        logger.error(f\"Error getting metrics: {e}\")
        raise HTTPException(status_code=500, detail=f\"Failed to get metrics: {str(e)}\")

# Background task functions
async def log_request_metrics(
    endpoint: str,
    processing_time_ms: float,
    sentiment: str,
    confidence: float
):
    \"\"\"Log request metrics asynchronously.\"\"\"
    
    try:
        collector = app_state.get(\"metrics_collector\")
        if collector:
            collector.record_request(
                endpoint=endpoint,
                duration_ms=processing_time_ms,
                sentiment=sentiment,
                confidence=confidence
            )
    except Exception as e:
        logger.error(f\"Failed to log metrics: {e}\")

# Error handlers
@app.exception_handler(EchoLocError)
async def echoloc_exception_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return HTTPException(status_code=422, detail=str(exc))

if __name__ == \"__main__\":
    import uvicorn
    
    uvicorn.run(
        \"echoloc_nn.sentiment.api:app\",
        host=\"0.0.0.0\",
        port=8000,
        log_level=\"info\",
        access_log=True,
        reload=False
    )"