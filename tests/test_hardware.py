"""
Tests for hardware interface components.
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock, mock_open
import serial

from echoloc_nn.hardware import (
    UltrasonicArray,
    SensorConfig,
    ArduinoInterface,
    ArduinoCommands
)
from echoloc_nn.utils.exceptions import HardwareError


class TestSensorConfig:
    """Test sensor configuration dataclass."""
    
    def test_sensor_config_creation(self):
        """Test basic sensor config creation."""
        config = SensorConfig(
            id=0,
            position=(0.0, 1.0),
            frequency=40000,
            beam_width=15.0,
            max_range=4.0
        )
        
        assert config.id == 0
        assert config.position == (0.0, 1.0)
        assert config.frequency == 40000
        assert config.beam_width == 15.0
        assert config.max_range == 4.0
    
    def test_sensor_config_defaults(self):
        """Test sensor config with default values."""
        config = SensorConfig(id=1, position=(1.0, 0.0))
        
        assert config.frequency == 40000  # Default
        assert config.beam_width == 15.0  # Default
        assert config.max_range == 4.0  # Default
        assert config.min_range == 0.02  # Default


class TestUltrasonicArray:
    """Test ultrasonic array interface."""
    
    def test_array_initialization(self):
        """Test array initialization with sensor configs."""
        sensors = [
            SensorConfig(0, (0.0, 0.0)),
            SensorConfig(1, (0.1, 0.0)),
            SensorConfig(2, (0.1, 0.1)),
            SensorConfig(3, (0.0, 0.1))
        ]
        
        array = UltrasonicArray(sensors, sample_rate=250000, buffer_size=4096)
        
        assert array.n_sensors == 4
        assert array.sample_rate == 250000
        assert array.buffer_size == 4096
        assert not array.is_connected
        assert not array.is_streaming
    
    def test_create_square_array(self):
        """Test square array factory method."""
        array = UltrasonicArray.create_square_array(spacing=0.1)
        
        assert array.n_sensors == 4
        
        positions = array.get_sensor_positions()
        assert positions.shape == (4, 2)
        
        # Check positions form a square
        expected_positions = np.array([
            [-0.05, -0.05],
            [0.05, -0.05],
            [0.05, 0.05],
            [-0.05, 0.05]
        ])
        
        assert np.allclose(positions, expected_positions)
    
    def test_create_linear_array(self):
        """Test linear array factory method."""
        array = UltrasonicArray.create_linear_array(n_sensors=6, spacing=0.05)
        
        assert array.n_sensors == 6
        
        positions = array.get_sensor_positions()
        assert positions.shape == (6, 2)
        
        # Check positions are linear
        expected_x = np.arange(6) * 0.05
        assert np.allclose(positions[:, 0], expected_x)
        assert np.allclose(positions[:, 1], 0.0)  # All y-coordinates should be 0
    
    @patch('serial.Serial')
    def test_connection_success(self, mock_serial):
        """Test successful connection to hardware."""
        # Mock serial connection
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'
        
        array = UltrasonicArray.create_square_array()
        
        success = array.connect('/dev/ttyUSB0')
        
        assert success
        assert array.is_connected
        mock_serial.assert_called_once()
    
    @patch('serial.Serial')
    def test_connection_failure(self, mock_serial):
        """Test connection failure handling."""
        # Mock serial connection failure
        mock_serial.side_effect = serial.SerialException("Port not found")
        
        array = UltrasonicArray.create_square_array()
        
        success = array.connect('/dev/ttyUSB0')
        
        assert not success
        assert not array.is_connected
    
    @patch('serial.Serial')
    def test_ping_test_failure(self, mock_serial):
        """Test connection failure due to ping test."""
        # Mock serial connection but wrong ping response
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'WRONG_RESPONSE\
'
        
        array = UltrasonicArray.create_square_array()
        
        success = array.connect('/dev/ttyUSB0')
        
        assert not success
        assert not array.is_connected
    
    @patch('echoloc_nn.hardware.ultrasonic_array.serial.Serial')
    def test_chirp_transmission(self, mock_serial):
        """Test chirp transmission."""
        # Setup mock
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'  # For connection test
        
        array = UltrasonicArray.create_square_array()
        array.connect('/dev/ttyUSB0')
        
        # Mock successful transmission
        mock_conn.readline.return_value = b'TX_OK\
'
        
        chirp_data = np.sin(2 * np.pi * 40000 * np.linspace(0, 0.005, 1250))
        
        success = array.transmit_chirp(chirp_data, sensor_id=0)
        
        assert success
        mock_conn.write.assert_called()
    
    @patch('echoloc_nn.hardware.ultrasonic_array.serial.Serial')
    def test_echo_reception(self, mock_serial):
        """Test echo data reception."""
        # Setup mock
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'  # For connection test
        
        array = UltrasonicArray.create_square_array()
        array.connect('/dev/ttyUSB0')
        
        # Mock echo data reception
        duration = 0.1
        n_samples = int(duration * array.sample_rate)
        expected_bytes = array.n_sensors * n_samples * 2  # 16-bit samples
        
        # Create fake echo data
        fake_data = (np.random.randn(array.n_sensors * n_samples) * 1000).astype(np.int16)
        mock_conn.read.return_value = fake_data.tobytes()
        
        echo_data = array.receive_echoes(duration)
        
        assert echo_data is not None
        assert echo_data.shape == (array.n_sensors, n_samples)
        assert np.all(np.abs(echo_data) <= 1.0)  # Should be normalized to [-1, 1]
    
    def test_array_geometry_calculation(self):
        """Test array geometry calculations."""
        array = UltrasonicArray.create_square_array(spacing=0.2)
        
        geometry = array.get_array_geometry()
        
        assert 'n_sensors' in geometry
        assert 'positions' in geometry
        assert 'center' in geometry
        assert 'max_baseline' in geometry
        assert 'array_aperture' in geometry
        
        assert geometry['n_sensors'] == 4
        assert np.allclose(geometry['center'], [0.0, 0.0])
        assert np.isclose(geometry['max_baseline'], 0.2 * np.sqrt(2))  # Diagonal
    
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_config_loading(self, mock_file, mock_yaml):
        """Test loading array config from file."""
        # Mock YAML config
        mock_config = {
            'sensors': [
                {'id': 0, 'position': [0.0, 0.0], 'frequency': 40000},
                {'id': 1, 'position': [0.1, 0.0], 'frequency': 40000}
            ],
            'sample_rate': 250000,
            'buffer_size': 4096,
            'port': '/dev/ttyUSB0'
        }
        mock_yaml.return_value = mock_config
        
        array = UltrasonicArray.from_config('test_config.yaml')
        
        assert array.n_sensors == 2
        assert array.sample_rate == 250000
        assert array.port == '/dev/ttyUSB0'
    
    def test_config_saving(self):
        """Test saving array config to file."""
        array = UltrasonicArray.create_square_array()
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('yaml.dump') as mock_dump:
                array.save_config('test_config.yaml')
                
                mock_file.assert_called_once_with('test_config.yaml', 'w')
                mock_dump.assert_called_once()
    
    def test_streaming_initialization(self):
        """Test streaming mode initialization."""
        array = UltrasonicArray.create_square_array()
        
        # Should not be able to start streaming without connection
        with pytest.raises(RuntimeError):
            array.start_streaming()
    
    @patch('echoloc_nn.hardware.ultrasonic_array.serial.Serial')
    def test_calibration(self, mock_serial):
        """Test array calibration."""
        # Setup mock connection
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'
        
        array = UltrasonicArray.create_square_array()
        array.connect('/dev/ttyUSB0')
        
        # Mock calibration data reception
        n_samples = int(0.1 * array.sample_rate)
        fake_data = (np.random.randn(array.n_sensors * n_samples) * 1000).astype(np.int16)
        mock_conn.read.return_value = fake_data.tobytes()
        mock_conn.readline.side_effect = [b'TX_OK\
'] * 10  # For chirp transmissions
        
        success = array.calibrate(method="auto")
        
        assert success
        assert 'calibration_time' in array.calibration_data


class TestArduinoInterface:
    """Test Arduino interface functionality."""
    
    def test_initialization(self):
        """Test Arduino interface initialization."""
        interface = ArduinoInterface('/dev/ttyUSB0', baudrate=115200)
        
        assert interface.port == '/dev/ttyUSB0'
        assert interface.baudrate == 115200
        assert not interface.is_connected
    
    @patch('serial.Serial')
    def test_successful_connection(self, mock_serial):
        """Test successful Arduino connection."""
        # Mock serial connection
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'
        
        interface = ArduinoInterface('/dev/ttyUSB0')
        
        success = interface.connect()
        
        assert success
        assert interface.is_connected
    
    @patch('serial.Serial')
    def test_ping_command(self, mock_serial):
        """Test ping command."""
        # Mock serial connection
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'
        
        interface = ArduinoInterface('/dev/ttyUSB0')
        interface.connect()
        
        # Test ping
        mock_conn.readline.return_value = b'PONG'
        success = interface.ping()
        
        assert success
    
    @patch('serial.Serial')
    def test_sensor_configuration(self, mock_serial):
        """Test sensor configuration."""
        # Mock serial connection
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'
        
        interface = ArduinoInterface('/dev/ttyUSB0')
        interface.connect()
        
        # Test configuration
        mock_conn.readline.return_value = b'CONFIG_OK'
        
        config = {
            'n_sensors': 4,
            'sample_rate': 250000,
            'sensors': [
                {'id': 0, 'trigger_pin': 2, 'echo_pin': 3, 'frequency': 40000},
                {'id': 1, 'trigger_pin': 4, 'echo_pin': 5, 'frequency': 40000},
                {'id': 2, 'trigger_pin': 6, 'echo_pin': 7, 'frequency': 40000},
                {'id': 3, 'trigger_pin': 8, 'echo_pin': 9, 'frequency': 40000}
            ]
        }
        
        success = interface.configure_sensors(config)
        
        assert success
        mock_conn.write.assert_called()
    
    @patch('serial.Serial')
    def test_chirp_transmission(self, mock_serial):
        """Test chirp transmission via Arduino."""
        # Mock serial connection
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'
        
        interface = ArduinoInterface('/dev/ttyUSB0')
        interface.connect()
        
        # Test chirp transmission
        mock_conn.readline.return_value = b'TX_OK'
        
        chirp_data = (np.sin(2 * np.pi * 40000 * np.linspace(0, 0.005, 1250)) * 127).astype(np.int8)
        
        success = interface.transmit_chirp(0, chirp_data.tobytes(), 40000)
        
        assert success
        mock_conn.write.assert_called()
    
    @patch('serial.Serial')
    def test_echo_reception(self, mock_serial):
        """Test echo data reception via Arduino."""
        # Mock serial connection
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'
        
        interface = ArduinoInterface('/dev/ttyUSB0')
        interface.connect()
        
        # Mock echo data
        duration_ms = 100
        sample_rate = 250000
        n_sensors = 4
        n_samples = (duration_ms * sample_rate) // 1000
        expected_bytes = n_sensors * n_samples * 2
        
        fake_data = (np.random.randn(n_sensors * n_samples) * 1000).astype(np.int16)
        mock_conn.read.return_value = fake_data.tobytes()
        
        echo_data = interface.receive_echoes(duration_ms, sample_rate)
        
        assert echo_data is not None
        assert len(echo_data) == expected_bytes
    
    @patch('serial.Serial')
    def test_combined_chirp_receive(self, mock_serial):
        """Test combined chirp transmission and reception."""
        # Mock serial connection
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'
        
        interface = ArduinoInterface('/dev/ttyUSB0')
        interface.connect()
        
        # Mock combined operation
        duration_ms = 100
        n_sensors = 4
        n_samples = (duration_ms * 250000) // 1000
        expected_bytes = n_sensors * n_samples * 2
        
        fake_data = (np.random.randn(n_sensors * n_samples) * 1000).astype(np.int16)
        mock_conn.read.return_value = fake_data.tobytes()
        
        chirp_data = b'\\x00' * 1000  # Dummy chirp data
        
        echo_data = interface.chirp_and_receive(
            chirp_data, duration_ms, tx_sensor=0, frequency=40000
        )
        
        assert echo_data is not None
        assert len(echo_data) == expected_bytes
    
    def test_arduino_sketch_generation(self):
        """Test Arduino sketch code generation."""
        interface = ArduinoInterface('/dev/ttyUSB0')
        
        config = {
            'n_sensors': 4,
            'sample_rate': 250000,
            'sensors': [
                {'id': i, 'trigger_pin': 2 + i*2, 'echo_pin': 3 + i*2, 'frequency': 40000}
                for i in range(4)
            ]
        }
        
        sketch = interface.generate_arduino_sketch(config)
        
        assert isinstance(sketch, str)
        assert len(sketch) > 1000  # Should be substantial code
        assert '#define N_SENSORS 4' in sketch
        assert '#define SAMPLE_RATE 250000' in sketch
        assert 'void setup()' in sketch
        assert 'void loop()' in sketch
    
    def test_status_parsing(self):
        """Test Arduino status parsing."""
        interface = ArduinoInterface('/dev/ttyUSB0')
        
        status_data = b'firmware_version=1.0.0,uptime_ms=12345,free_memory=1024,sensors_active=4'
        
        status = interface._parse_status(status_data)
        
        assert status['firmware_version'] == '1.0.0'
        assert status['uptime_ms'] == 12345
        assert status['free_memory'] == 1024
        assert status['sensors_active'] == 4
    
    def test_sensor_mask_creation(self):
        """Test sensor selection mask creation."""
        interface = ArduinoInterface('/dev/ttyUSB0')
        
        # Test mask for sensors 0, 2, 3
        mask = interface._create_sensor_mask([0, 2, 3])
        
        expected_mask = (1 << 0) | (1 << 2) | (1 << 3)  # Binary: 00001101
        assert mask == expected_mask
        
        # Test all sensors
        mask = interface._create_sensor_mask([0, 1, 2, 3])
        assert mask == 0b00001111


class TestHardwareEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_sensor_positions(self):
        """Test handling of invalid sensor positions."""
        with pytest.raises((ValueError, TypeError)):
            # Non-numeric positions
            SensorConfig(0, ("invalid", "position"))
    
    def test_empty_sensor_list(self):
        """Test handling of empty sensor list."""
        with pytest.raises((ValueError, IndexError)):
            UltrasonicArray([])  # Empty sensor list
    
    @patch('serial.Serial')
    def test_connection_timeout(self, mock_serial):
        """Test connection timeout handling."""
        # Mock serial timeout
        mock_serial.side_effect = serial.SerialTimeoutException("Timeout")
        
        interface = ArduinoInterface('/dev/ttyUSB0', timeout=1.0)
        
        success = interface.connect()
        assert not success
    
    @patch('serial.Serial')
    def test_data_corruption_handling(self, mock_serial):
        """Test handling of corrupted data."""
        # Mock serial connection
        mock_conn = MagicMock()
        mock_serial.return_value = mock_conn
        mock_conn.readline.return_value = b'PONG\
'
        
        array = UltrasonicArray.create_square_array()
        array.connect('/dev/ttyUSB0')
        
        # Mock corrupted data (wrong size)
        mock_conn.read.return_value = b'corrupted_data'
        
        echo_data = array.receive_echoes(0.1)
        
        assert echo_data is None  # Should handle corruption gracefully
    
    def test_large_array_handling(self):
        """Test handling of large sensor arrays."""
        # Create array with many sensors
        sensors = [SensorConfig(i, (i * 0.01, 0.0)) for i in range(16)]
        
        array = UltrasonicArray(sensors)
        
        assert array.n_sensors == 16
        
        positions = array.get_sensor_positions()
        assert positions.shape == (16, 2)
    
    def test_duplicate_sensor_ids(self):
        """Test handling of duplicate sensor IDs."""
        sensors = [
            SensorConfig(0, (0.0, 0.0)),
            SensorConfig(0, (0.1, 0.0))  # Duplicate ID
        ]
        
        # Should create array (IDs are not enforced to be unique in current implementation)
        array = UltrasonicArray(sensors)
        assert array.n_sensors == 2


@pytest.mark.hardware
class TestHardwareIntegration:
    """Integration tests requiring actual hardware."""
    
    @pytest.mark.skip(reason="Requires physical hardware")
    def test_real_arduino_connection(self):
        """Test connection to real Arduino hardware."""
        # This test would require actual Arduino hardware
        interface = ArduinoInterface('/dev/ttyUSB0')
        
        success = interface.connect()
        
        if success:
            # Test basic communication
            ping_success = interface.ping()
            assert ping_success
            
            # Get status
            status = interface.get_status()
            assert status is not None
            
            interface.disconnect()
    
    @pytest.mark.skip(reason="Requires physical hardware")
    def test_real_sensor_array(self):
        """Test with real ultrasonic sensor array."""
        # This test would require actual sensor hardware
        array = UltrasonicArray.create_square_array()
        
        success = array.connect('/dev/ttyUSB0')
        
        if success:
            # Test calibration
            calibration_success = array.calibrate()
            assert calibration_success
            
            # Test basic echo acquisition
            chirp_data = np.sin(2 * np.pi * 40000 * np.linspace(0, 0.005, 1250))
            echo_data = array.chirp_and_receive(chirp_data, 0.1)
            
            assert echo_data is not None
            assert echo_data.shape[0] == array.n_sensors
            
            array.disconnect()


if __name__ == "__main__":
    pytest.main([__file__])