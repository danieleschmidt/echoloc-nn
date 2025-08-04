"""
Ultrasonic sensor array interface and configuration.
"""

from typing import List, Tuple, Optional, Dict, Any, Iterator
from dataclasses import dataclass
import numpy as np
import time
import threading
from queue import Queue, Empty
import serial
import yaml


@dataclass
class SensorConfig:
    """Configuration for a single ultrasonic sensor."""
    
    id: int
    position: Tuple[float, float]  # (x, y) position in meters
    frequency: float = 40000  # Operating frequency in Hz
    beam_width: float = 15.0  # Beam width in degrees
    max_range: float = 4.0  # Maximum range in meters
    min_range: float = 0.02  # Minimum range in meters
    trigger_pin: Optional[int] = None  # Arduino trigger pin
    echo_pin: Optional[int] = None  # Arduino echo pin


class UltrasonicArray:
    """
    Interface for ultrasonic sensor arrays.
    
    Provides unified interface for controlling multiple ultrasonic
    sensors and collecting synchronized echo data.
    """
    
    def __init__(
        self,
        sensor_configs: List[SensorConfig],
        sample_rate: int = 250000,
        buffer_size: int = 4096,
        port: Optional[str] = None
    ):
        self.sensor_configs = sensor_configs
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.port = port
        
        self.n_sensors = len(sensor_configs)
        self.is_connected = False
        self.is_streaming = False
        
        # Communication interface
        self.serial_conn: Optional[serial.Serial] = None
        self.data_queue = Queue(maxsize=100)
        self.stream_thread: Optional[threading.Thread] = None
        
        # Calibration data
        self.calibration_data: Dict[str, Any] = {}
        
    @classmethod
    def from_config(cls, config_path: str) -> 'UltrasonicArray':
        """Create array from YAML configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        sensors = []
        for sensor_data in config['sensors']:
            sensors.append(SensorConfig(**sensor_data))
            
        return cls(
            sensor_configs=sensors,
            sample_rate=config.get('sample_rate', 250000),
            buffer_size=config.get('buffer_size', 4096),
            port=config.get('port')
        )
    
    @classmethod
    def create_square_array(
        cls,
        spacing: float = 0.1,
        center: Tuple[float, float] = (0.0, 0.0)
    ) -> 'UltrasonicArray':
        """Create 4-sensor square array configuration."""
        sensors = [
            SensorConfig(0, (center[0] - spacing/2, center[1] - spacing/2)),
            SensorConfig(1, (center[0] + spacing/2, center[1] - spacing/2)),
            SensorConfig(2, (center[0] + spacing/2, center[1] + spacing/2)),
            SensorConfig(3, (center[0] - spacing/2, center[1] + spacing/2))
        ]
        return cls(sensors)
    
    @classmethod
    def create_linear_array(
        cls,
        n_sensors: int = 4,
        spacing: float = 0.05,
        start_pos: Tuple[float, float] = (0.0, 0.0)
    ) -> 'UltrasonicArray':
        """Create linear sensor array configuration."""
        sensors = []
        for i in range(n_sensors):
            pos = (start_pos[0] + i * spacing, start_pos[1])
            sensors.append(SensorConfig(i, pos))
        return cls(sensors)
    
    def connect(self, port: Optional[str] = None, baudrate: int = 115200) -> bool:
        """
        Connect to sensor array hardware.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0', 'COM3')
            baudrate: Serial communication baud rate
            
        Returns:
            True if connection successful
        """
        if port:
            self.port = port
            
        if not self.port:
            raise ValueError("Port must be specified")
            
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            # Wait for Arduino initialization
            time.sleep(2.0)
            
            # Test connection
            if self._test_connection():
                self.is_connected = True
                print(f"Connected to ultrasonic array on {self.port}")
                return True
            else:
                self.serial_conn.close()
                self.serial_conn = None
                return False
                
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from sensor array."""
        if self.is_streaming:
            self.stop_streaming()
            
        if self.serial_conn and self.is_connected:
            self.serial_conn.close()
            self.serial_conn = None
            self.is_connected = False
            print("Disconnected from ultrasonic array")
    
    def _test_connection(self) -> bool:
        """Test communication with sensor array."""
        if not self.serial_conn:
            return False
            
        try:
            # Send ping command
            self.serial_conn.write(b'PING\\n')
            response = self.serial_conn.readline().decode().strip()
            return response == 'PONG'
        except:
            return False
    
    def transmit_chirp(
        self,
        chirp_data: np.ndarray,
        sensor_id: Optional[int] = None
    ) -> bool:
        """
        Transmit chirp signal through specified sensor.
        
        Args:
            chirp_data: Chirp waveform data
            sensor_id: Sensor ID (None for all sensors)
            
        Returns:
            True if transmission successful
        """
        if not self.is_connected or not self.serial_conn:
            raise RuntimeError("Array not connected")
            
        try:
            # Convert chirp to bytes (simplified)
            chirp_bytes = (chirp_data * 127).astype(np.int8).tobytes()
            
            # Send transmission command
            cmd = f"TX:{sensor_id if sensor_id is not None else 'ALL'}:{len(chirp_bytes)}\\n"
            self.serial_conn.write(cmd.encode())
            
            # Send chirp data
            self.serial_conn.write(chirp_bytes)
            
            # Wait for acknowledgment
            response = self.serial_conn.readline().decode().strip()
            return response == 'TX_OK'
            
        except Exception as e:
            print(f"Transmission failed: {e}")
            return False
    
    def receive_echoes(self, duration: float = 0.1) -> Optional[np.ndarray]:
        """
        Receive echo data from all sensors.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Echo data array (n_sensors, n_samples) or None if failed
        """
        if not self.is_connected or not self.serial_conn:
            raise RuntimeError("Array not connected")
            
        n_samples = int(duration * self.sample_rate)
        
        try:
            # Send receive command
            cmd = f"RX:{duration}:{self.sample_rate}\\n"
            self.serial_conn.write(cmd.encode())
            
            # Wait for data
            expected_bytes = self.n_sensors * n_samples * 2  # 16-bit samples
            data_bytes = self.serial_conn.read(expected_bytes)
            
            if len(data_bytes) != expected_bytes:
                print(f"Received {len(data_bytes)} bytes, expected {expected_bytes}")
                return None
                
            # Convert bytes to numpy array
            raw_data = np.frombuffer(data_bytes, dtype=np.int16)
            echo_data = raw_data.reshape((self.n_sensors, n_samples)).astype(np.float32)
            
            # Convert to voltage range [-1, 1]
            echo_data = echo_data / 32767.0
            
            return echo_data
            
        except Exception as e:
            print(f"Reception failed: {e}")
            return None
    
    def chirp_and_receive(
        self,
        chirp_data: np.ndarray,
        receive_duration: float = 0.1,
        tx_sensor: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Transmit chirp and receive echoes in one operation.
        
        Args:
            chirp_data: Chirp waveform to transmit
            receive_duration: Echo recording duration
            tx_sensor: Transmitting sensor ID (None for all)
            
        Returns:
            Received echo data or None if failed
        """
        if self.transmit_chirp(chirp_data, tx_sensor):
            # Small delay for chirp transmission
            time.sleep(len(chirp_data) / self.sample_rate + 0.001)
            return self.receive_echoes(receive_duration)
        return None
    
    def start_streaming(self, update_rate: float = 20.0):
        """
        Start continuous echo streaming.
        
        Args:
            update_rate: Update rate in Hz
        """
        if self.is_streaming:
            return
            
        if not self.is_connected:
            raise RuntimeError("Array not connected")
            
        self.is_streaming = True
        self.stream_thread = threading.Thread(
            target=self._stream_worker,
            args=(update_rate,),
            daemon=True
        )
        self.stream_thread.start()
        print(f"Started streaming at {update_rate} Hz")
    
    def stop_streaming(self):
        """Stop continuous echo streaming."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
            self.stream_thread = None
        print("Stopped streaming")
    
    def _stream_worker(self, update_rate: float):
        """Worker thread for continuous streaming."""
        period = 1.0 / update_rate
        
        # Default chirp for streaming
        t = np.linspace(0, 0.005, int(0.005 * self.sample_rate))
        chirp = np.sin(2 * np.pi * 40000 * t)
        
        while self.is_streaming:
            start_time = time.time()
            
            # Acquire echo data
            echo_data = self.chirp_and_receive(chirp, 0.05)
            
            if echo_data is not None:
                # Add timestamp
                timestamped_data = {
                    'timestamp': time.time(),
                    'echo_data': echo_data,
                    'sensor_positions': self.get_sensor_positions()
                }
                
                # Put in queue (non-blocking)
                try:
                    self.data_queue.put_nowait(timestamped_data)
                except:
                    # Queue full, remove oldest item
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(timestamped_data)
                    except Empty:
                        pass
            
            # Maintain update rate
            elapsed = time.time() - start_time
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def stream_chirps(self) -> Iterator[Dict[str, Any]]:
        """
        Iterator for streaming echo data.
        
        Yields:
            Dictionary with timestamp, echo_data, and sensor_positions
        """
        if not self.is_streaming:
            raise RuntimeError("Streaming not started")
            
        while self.is_streaming:
            try:
                data = self.data_queue.get(timeout=1.0)
                yield data
            except Empty:
                continue
    
    def get_latest_echo(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get most recent echo data from stream."""
        try:
            return self.data_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def calibrate(self, method: str = "auto") -> bool:
        """
        Calibrate sensor array.
        
        Args:
            method: Calibration method ("auto", "manual", "known_target")
            
        Returns:
            True if calibration successful
        """
        if method == "auto":
            return self._auto_calibrate()
        elif method == "manual":
            return self._manual_calibrate()
        elif method == "known_target":
            return self._known_target_calibrate()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def _auto_calibrate(self) -> bool:
        """Automatic calibration using ambient reflections."""
        print("Starting automatic calibration...")
        
        # Collect calibration data
        calibration_echoes = []
        for i in range(10):
            t = np.linspace(0, 0.005, int(0.005 * self.sample_rate))
            chirp = np.sin(2 * np.pi * 40000 * t)
            
            echo_data = self.chirp_and_receive(chirp, 0.1)
            if echo_data is not None:
                calibration_echoes.append(echo_data)
            time.sleep(0.1)
        
        if not calibration_echoes:
            print("Calibration failed: No echo data received")
            return False
        
        # Compute calibration parameters
        echo_array = np.array(calibration_echoes)
        
        self.calibration_data = {
            'mean_echo': np.mean(echo_array, axis=0),
            'std_echo': np.std(echo_array, axis=0),
            'noise_floor': np.percentile(np.abs(echo_array), 10, axis=0),
            'calibration_time': time.time()
        }
        
        print("Automatic calibration completed")
        return True
    
    def _manual_calibrate(self) -> bool:
        """Manual calibration with user interaction."""
        print("Manual calibration not implemented")
        return False
    
    def _known_target_calibrate(self) -> bool:
        """Calibration using known target positions."""
        print("Known target calibration not implemented")
        return False
    
    def get_sensor_positions(self) -> np.ndarray:
        """Get sensor positions as numpy array."""
        positions = [sensor.position for sensor in self.sensor_configs]
        return np.array(positions)
    
    def get_array_geometry(self) -> Dict[str, Any]:
        """Get array geometry information."""
        positions = self.get_sensor_positions()
        
        return {
            'n_sensors': self.n_sensors,
            'positions': positions,
            'center': np.mean(positions, axis=0),
            'max_baseline': np.max(np.linalg.norm(
                positions[:, None] - positions[None, :], axis=2
            )),
            'array_aperture': np.max(positions, axis=0) - np.min(positions, axis=0)
        }
    
    def save_config(self, config_path: str):
        """Save array configuration to YAML file."""
        config = {
            'sensors': [
                {
                    'id': sensor.id,
                    'position': list(sensor.position),
                    'frequency': sensor.frequency,
                    'beam_width': sensor.beam_width,
                    'max_range': sensor.max_range,
                    'min_range': sensor.min_range
                }
                for sensor in self.sensor_configs
            ],
            'sample_rate': self.sample_rate,
            'buffer_size': self.buffer_size,
            'port': self.port
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()