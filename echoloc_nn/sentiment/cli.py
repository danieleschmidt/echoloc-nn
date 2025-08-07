"""
Command-line interface for EchoLoc sentiment analysis.

Provides batch processing, real-time analysis, and integration
with the ultrasonic localization system.
"""

import click
import asyncio
import json
import csv
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from datetime import datetime

from .models import SpatialSentimentAnalyzer, MultiModalSentimentModel
from .spatial_fusion import LocationContextSentiment, LocationContext
from .real_time import StreamingSentimentAnalyzer, StreamingConfig
from .multi_modal import MultiModalSentimentProcessor, MultiModalInput, AudioSpatialSentiment
from ..utils.error_handling import handle_errors, EchoLocError
from ..utils.validation import validate_input
from ..utils.monitoring import PerformanceMonitor

# Global state for CLI
cli_state = {
    \"models_loaded\": False,
    \"spatial_model\": None,
    \"multimodal_model\": None,
    \"location_manager\": None,
    \"streaming_analyzer\": None,
    \"multimodal_processor\": None,
    \"performance_monitor\": None
}

def initialize_models(model_path: Optional[str] = None, 
                     config_path: Optional[str] = None) -> bool:
    \"\"\"Initialize sentiment analysis models.\"\"\"
    
    try:
        click.echo(\"🚀 Initializing EchoLoc Sentiment Analysis models...\")
        
        # Performance monitoring
        cli_state[\"performance_monitor\"] = PerformanceMonitor(\"SentimentCLI\")
        
        # Initialize base models
        with cli_state[\"performance_monitor\"].measure(\"model_loading\"):
            cli_state[\"spatial_model\"] = SpatialSentimentAnalyzer(
                model_name=\"distilbert-base-uncased\",
                spatial_dim=128,
                sentiment_classes=3
            )
            
            cli_state[\"multimodal_model\"] = MultiModalSentimentModel(
                sentiment_classes=5
            )
            
        # Initialize location manager
        cli_state[\"location_manager\"] = LocationContextSentiment()
        
        # Initialize multi-modal processor
        cli_state[\"multimodal_processor\"] = MultiModalSentimentProcessor(
            cli_state[\"multimodal_model\"],
            cli_state[\"location_manager\"]
        )
        
        # Initialize streaming analyzer
        streaming_config = StreamingConfig(
            update_frequency_hz=5.0,
            max_buffer_size=100,
            confidence_threshold=0.7
        )
        
        cli_state[\"streaming_analyzer\"] = StreamingSentimentAnalyzer(
            cli_state[\"spatial_model\"],
            cli_state[\"location_manager\"],
            streaming_config
        )
        
        cli_state[\"models_loaded\"] = True
        
        click.echo(\"✅ Models initialized successfully\")
        return True
        
    except Exception as e:
        click.echo(f\"❌ Failed to initialize models: {e}\", err=True)
        return False

def check_models_loaded():
    \"\"\"Ensure models are loaded before processing.\"\"\"
    if not cli_state[\"models_loaded\"]:
        if not initialize_models():
            click.echo(\"❌ Cannot proceed without loaded models\", err=True)
            sys.exit(1)

@click.group()
@click.option('--model-path', help='Path to pre-trained model')
@click.option('--config-path', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def cli(model_path, config_path, verbose):
    \"\"\"EchoLoc Sentiment Analysis CLI - Spatial-aware emotion understanding.\"\"\"
    
    if verbose:
        click.echo(\"EchoLoc Sentiment Analysis CLI v1.0.0\")
        click.echo(\"Spatial-aware sentiment analysis with ultrasonic localization\")
    
    # Initialize models if paths provided
    if model_path or config_path:
        if not initialize_models(model_path, config_path):
            sys.exit(1)

@cli.command()
@click.argument('text')
@click.option('--spatial', '-s', help='Spatial context as comma-separated values (x,y,z,vx,vy,vz,ax,ay,az)')
@click.option('--zone', help='Spatial zone identifier')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv', 'text']), default='text')
def analyze(text, spatial, zone, output, output_format):
    \"\"\"Analyze sentiment of a single text with optional spatial context.\"\"\"
    
    check_models_loaded()
    
    try:
        # Parse spatial context
        spatial_context = None
        if spatial:
            try:
                values = [float(x.strip()) for x in spatial.split(',')]
                if len(values) != 9:
                    raise ValueError(\"Spatial context must have 9 values\")
                spatial_context = np.array(values)
            except ValueError as e:
                click.echo(f\"❌ Invalid spatial context: {e}\", err=True)
                return
        
        # Analyze sentiment
        start_time = time.time()
        
        with cli_state[\"performance_monitor\"].measure(\"sentiment_analysis\"):
            result = cli_state[\"spatial_model\"].predict_sentiment(
                text=text,
                spatial_context=spatial_context
            )
        
        processing_time = time.time() - start_time
        
        # Format results
        analysis_result = {
            \"text\": text,
            \"sentiment\": result[\"sentiment\"],
            \"confidence\": float(result[\"confidence\"]),
            \"probabilities\": {
                \"negative\": float(result[\"probabilities\"][0]) if isinstance(result[\"probabilities\"], np.ndarray) else 0.0,
                \"neutral\": float(result[\"probabilities\"][1]) if isinstance(result[\"probabilities\"], np.ndarray) and len(result[\"probabilities\"]) > 1 else 0.0,
                \"positive\": float(result[\"probabilities\"][2]) if isinstance(result[\"probabilities\"], np.ndarray) and len(result[\"probabilities\"]) > 2 else 0.0
            },
            \"spatial_influence\": float(result.get(\"spatial_influence\", 0.0)),
            \"zone\": zone,
            \"processing_time_ms\": processing_time * 1000,
            \"timestamp\": datetime.utcnow().isoformat() + \"Z\"
        }
        
        # Output results
        if output:
            save_results([analysis_result], output, output_format)
            click.echo(f\"✅ Results saved to {output}\")
        else:
            display_results([analysis_result], output_format)
            
    except Exception as e:
        click.echo(f\"❌ Analysis failed: {e}\", err=True)

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--spatial-column', help='Column name for spatial context')
@click.option('--zone-column', help='Column name for spatial zones')
@click.option('--text-column', default='text', help='Column name for text data')
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']), default='json')
@click.option('--batch-size', default=32, help='Batch size for processing')
@click.option('--progress', is_flag=True, help='Show progress bar')
def batch(input_file, spatial_column, zone_column, text_column, output, output_format, batch_size, progress):
    \"\"\"Batch process sentiment analysis from CSV or JSON file.\"\"\"
    
    check_models_loaded()
    
    try:
        # Load input data
        input_path = Path(input_file)
        
        if input_path.suffix.lower() == '.csv':
            data = load_csv_data(input_path, text_column, spatial_column, zone_column)
        elif input_path.suffix.lower() == '.json':
            data = load_json_data(input_path, text_column, spatial_column, zone_column)
        else:
            click.echo(f\"❌ Unsupported file format: {input_path.suffix}\", err=True)
            return
            
        if not data:
            click.echo(\"❌ No data loaded from input file\", err=True)
            return
            
        click.echo(f\"📁 Loaded {len(data)} records from {input_file}\")
        
        # Process in batches
        results = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        with click.progressbar(
            length=len(data),
            show_eta=True,
            label=\"Processing\"
        ) as bar:
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(data))
                batch_data = data[start_idx:end_idx]
                
                # Process batch
                batch_results = process_batch(batch_data)
                results.extend(batch_results)
                
                if progress:
                    bar.update(len(batch_data))
                    
        # Save results
        save_results(results, output, output_format)
        
        # Summary statistics
        sentiment_counts = {}
        total_confidence = 0
        
        for result in results:
            sentiment = result[\"sentiment\"]
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_confidence += result[\"confidence\"]
            
        avg_confidence = total_confidence / len(results) if results else 0
        
        click.echo(f\"\
✅ Processed {len(results)} records\")
        click.echo(f\"📊 Average confidence: {avg_confidence:.3f}\")
        click.echo(\"📈 Sentiment distribution:\")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(results)) * 100
            click.echo(f\"   {sentiment}: {count} ({percentage:.1f}%)\")
            
        click.echo(f\"💾 Results saved to {output}\")
        
    except Exception as e:
        click.echo(f\"❌ Batch processing failed: {e}\", err=True)

@cli.command()
@click.option('--duration', default=60, help='Duration in seconds (0 for infinite)')
@click.option('--frequency', default=5.0, help='Update frequency in Hz')
@click.option('--output', '-o', help='Output file for results')
@click.option('--show-metrics', is_flag=True, help='Show real-time metrics')
def stream(duration, frequency, output, show_metrics):
    \"\"\"Start real-time streaming sentiment analysis.\"\"\"
    
    check_models_loaded()
    
    try:
        # Configure streaming
        config = StreamingConfig(
            update_frequency_hz=frequency,
            max_buffer_size=1000,
            confidence_threshold=0.5
        )
        
        analyzer = cli_state[\"streaming_analyzer\"]
        analyzer.config = config
        
        # Start streaming
        analyzer.start_streaming()
        
        click.echo(f\"🔄 Started streaming sentiment analysis at {frequency} Hz\")
        click.echo(\"📝 Enter text to analyze (Ctrl+C to stop):\")
        
        results = []
        start_time = time.time()
        
        try:
            while True:
                # Check duration limit
                if duration > 0 and (time.time() - start_time) > duration:
                    break
                    
                # Get user input with timeout
                try:
                    import select
                    import sys
                    
                    # Check if input is available
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        text = input().strip()
                        if text:
                            # Analyze text
                            result = analyzer.analyze_text(text)
                            if result:
                                results.append({
                                    \"text\": result.text,
                                    \"sentiment\": result.sentiment,
                                    \"confidence\": float(result.confidence),
                                    \"spatial_influence\": float(result.spatial_influence),
                                    \"timestamp\": result.timestamp
                                })
                                
                                click.echo(f\"  → {result.sentiment} ({result.confidence:.3f})\")
                            else:
                                click.echo(\"  → Queued for processing\")
                                
                except EOFError:
                    break
                except ImportError:
                    # Fallback for systems without select
                    text = input().strip()
                    if text:
                        result = analyzer.analyze_text(text)
                        if result:
                            click.echo(f\"  → {result.sentiment} ({result.confidence:.3f})\")
                            
                # Show metrics
                if show_metrics and len(results) % 10 == 0 and results:
                    display_streaming_metrics(analyzer.get_metrics())
                    
        except KeyboardInterrupt:
            click.echo(\"\
🛑 Stopping streaming...\")
            
        finally:
            analyzer.stop_streaming()
            
        # Save results if requested
        if output and results:
            save_results(results, output, \"json\")
            click.echo(f\"💾 Results saved to {output}\")
            
        # Final metrics
        final_metrics = analyzer.get_metrics()
        click.echo(f\"\
📊 Final Statistics:\")
        click.echo(f\"   Total predictions: {final_metrics.total_predictions}\")
        click.echo(f\"   Average latency: {final_metrics.avg_latency_ms:.1f} ms\")
        click.echo(f\"   Predictions per second: {final_metrics.predictions_per_second:.1f}\")
        
    except Exception as e:
        click.echo(f\"❌ Streaming failed: {e}\", err=True)

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--text', help='Optional text to analyze alongside audio')
@click.option('--spatial', help='Spatial context as comma-separated values')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text']), default='text')
def multimodal(audio_file, text, spatial, output, output_format):
    \"\"\"Multi-modal sentiment analysis with audio and optional text/spatial data.\"\"\"
    
    check_models_loaded()
    
    try:
        # Prepare input data
        input_data = MultiModalInput(
            text=text,
            audio_file_path=audio_file,
            timestamp=time.time()
        )
        
        # Parse spatial context
        if spatial:
            try:
                values = [float(x.strip()) for x in spatial.split(',')]
                if len(values) == 9:
                    input_data.spatial_context = np.array(values)
            except ValueError:
                click.echo(\"⚠️  Invalid spatial context, ignoring\", err=True)
                
        # Run multi-modal analysis
        click.echo(\"🎵 Processing multi-modal sentiment analysis...\")
        
        start_time = time.time()
        results = asyncio.run(
            cli_state[\"multimodal_processor\"].analyze_multi_modal(input_data)
        )
        processing_time = time.time() - start_time
        
        # Format results
        analysis_result = {
            \"audio_file\": audio_file,
            \"text\": text,
            \"modalities_used\": results[\"modalities_used\"],
            \"processing_time_ms\": processing_time * 1000,
            \"timestamp\": datetime.utcnow().isoformat() + \"Z\",
            \"results\": results
        }
        
        # Output results
        if output:
            save_results([analysis_result], output, output_format)
            click.echo(f\"✅ Results saved to {output}\")
        else:
            display_multimodal_results(analysis_result, output_format)
            
    except Exception as e:
        click.echo(f\"❌ Multi-modal analysis failed: {e}\", err=True)

@cli.command()
def info():
    \"\"\"Display system information and model status.\"\"\"
    
    click.echo(\"🎯 EchoLoc Sentiment Analysis System Information\")
    click.echo(\"=\" * 50)
    
    # System info
    import platform
    import torch
    
    click.echo(f\"🖥️  Platform: {platform.system()} {platform.release()}\")
    click.echo(f\"🐍 Python: {platform.python_version()}\")
    click.echo(f\"🔥 PyTorch: {torch.__version__}\")
    click.echo(f\"🚀 CUDA Available: {torch.cuda.is_available()}\")
    
    if torch.cuda.is_available():
        click.echo(f\"   Device: {torch.cuda.get_device_name()}\")
        click.echo(f\"   Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB\")
        
    # Model status
    click.echo(\"\
📦 Model Status:\")
    click.echo(f\"   Models Loaded: {cli_state['models_loaded']}\")
    click.echo(f\"   Spatial Model: {'✅' if cli_state['spatial_model'] else '❌'}\")
    click.echo(f\"   Multimodal Model: {'✅' if cli_state['multimodal_model'] else '❌'}\")
    click.echo(f\"   Location Manager: {'✅' if cli_state['location_manager'] else '❌'}\")
    click.echo(f\"   Streaming Analyzer: {'✅' if cli_state['streaming_analyzer'] else '❌'}\")
    
    # Performance info
    if cli_state[\"performance_monitor\"]:
        metrics = cli_state[\"performance_monitor\"].get_metrics()
        if metrics:
            click.echo(\"\
⚡ Performance Metrics:\")
            for operation, stats in metrics.items():
                click.echo(f\"   {operation}: {stats['avg_duration_ms']:.1f} ms avg\")
                
# Helper functions
def load_csv_data(file_path: Path, text_col: str, spatial_col: str, zone_col: str) -> List[Dict]:
    \"\"\"Load data from CSV file.\"\"\"
    
    import pandas as pd
    
    df = pd.read_csv(file_path)
    
    data = []
    for _, row in df.iterrows():
        item = {\"text\": row[text_col]}
        
        if spatial_col and spatial_col in df.columns:
            try:
                spatial_str = str(row[spatial_col])
                values = [float(x.strip()) for x in spatial_str.split(',')]
                if len(values) == 9:
                    item[\"spatial_context\"] = values
            except:
                pass
                
        if zone_col and zone_col in df.columns:
            item[\"zone\"] = str(row[zone_col])
            
        data.append(item)
        
    return data

def load_json_data(file_path: Path, text_col: str, spatial_col: str, zone_col: str) -> List[Dict]:
    \"\"\"Load data from JSON file.\"\"\"
    
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
        
    if not isinstance(raw_data, list):
        raw_data = [raw_data]
        
    data = []
    for item in raw_data:
        if text_col in item:
            processed_item = {\"text\": item[text_col]}
            
            if spatial_col and spatial_col in item:
                spatial_data = item[spatial_col]
                if isinstance(spatial_data, list) and len(spatial_data) == 9:
                    processed_item[\"spatial_context\"] = spatial_data
                    
            if zone_col and zone_col in item:
                processed_item[\"zone\"] = item[zone_col]
                
            data.append(processed_item)
            
    return data

def process_batch(batch_data: List[Dict]) -> List[Dict]:
    \"\"\"Process a batch of data.\"\"\"
    
    results = []
    
    for item in batch_data:
        try:
            spatial_context = None
            if \"spatial_context\" in item:
                spatial_context = np.array(item[\"spatial_context\"])
                
            result = cli_state[\"spatial_model\"].predict_sentiment(
                text=item[\"text\"],
                spatial_context=spatial_context
            )
            
            analysis_result = {
                \"text\": item[\"text\"],
                \"sentiment\": result[\"sentiment\"],
                \"confidence\": float(result[\"confidence\"]),
                \"spatial_influence\": float(result.get(\"spatial_influence\", 0.0)),
                \"zone\": item.get(\"zone\"),
                \"timestamp\": datetime.utcnow().isoformat() + \"Z\"
            }
            
            if isinstance(result[\"probabilities\"], np.ndarray):
                analysis_result[\"probabilities\"] = result[\"probabilities\"].tolist()
            else:
                analysis_result[\"probabilities\"] = result[\"probabilities\"]
                
            results.append(analysis_result)
            
        except Exception as e:
            click.echo(f\"⚠️  Failed to process item: {e}\", err=True)
            
    return results

def save_results(results: List[Dict], output_path: str, format_type: str):
    \"\"\"Save results to file.\"\"\"
    
    output_file = Path(output_path)
    
    if format_type == 'json':
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    elif format_type == 'csv':
        if results:
            fieldnames = results[0].keys()
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
                
def display_results(results: List[Dict], format_type: str):
    \"\"\"Display results to console.\"\"\"
    
    for result in results:
        if format_type == 'json':
            click.echo(json.dumps(result, indent=2))
        elif format_type == 'text':
            click.echo(f\"📝 Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}\")
            click.echo(f\"😊 Sentiment: {result['sentiment']} ({result['confidence']:.3f} confidence)\")
            if result.get('spatial_influence'):
                click.echo(f\"📍 Spatial influence: {result['spatial_influence']:.3f}\")
            if result.get('zone'):
                click.echo(f\"🏢 Zone: {result['zone']}\")
            click.echo(f\"⏰ Processed: {result['timestamp']}\")
            click.echo(\"-\" * 40)
            
def display_multimodal_results(result: Dict, format_type: str):
    \"\"\"Display multi-modal results.\"\"\"
    
    if format_type == 'json':
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f\"🎵 Audio file: {result['audio_file']}\")
        if result['text']:
            click.echo(f\"📝 Text: {result['text']}\")
        click.echo(f\"🔧 Modalities used: {', '.join(result['modalities_used'])}\")
        click.echo(f\"⏱️  Processing time: {result['processing_time_ms']:.1f} ms\")
        
        # Show fused analysis if available
        if result['results'].get('fused_analysis'):
            fused = result['results']['fused_analysis']
            click.echo(f\"\
🎯 Final Result:\")
            click.echo(f\"   Sentiment: {fused['sentiment']}\")
            click.echo(f\"   Confidence: {fused['confidence']:.3f}\")
            click.echo(f\"   Modality weights: {fused['modality_weights']}\")
            
def display_streaming_metrics(metrics):
    \"\"\"Display streaming metrics.\"\"\"
    
    click.echo(f\"\
📊 Streaming Metrics:\")
    click.echo(f\"   Predictions: {metrics.total_predictions}\")
    click.echo(f\"   Avg Latency: {metrics.avg_latency_ms:.1f} ms\")
    click.echo(f\"   Rate: {metrics.predictions_per_second:.1f} pred/sec\")
    
    if metrics.sentiment_distribution:
        click.echo(\"   Sentiment distribution:\")
        for sentiment, count in metrics.sentiment_distribution.items():
            click.echo(f\"     {sentiment}: {count}\")

if __name__ == '__main__':
    cli()"