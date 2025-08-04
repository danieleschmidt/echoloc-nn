"""
Arduino interface for ultrasonic sensor control.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import time
import serial
import struct


class ArduinoCommands(Enum):
    """Arduino command codes."""
    PING = b'PING'
    RESET = b'RESET'
    CONFIG = b'CONFIG'
    TRANSMIT = b'TX'
    RECEIVE = b'RX'
    CHIRP_RX = b'CHIRP_RX'
    GET_STATUS = b'STATUS'
    CALIBRATE = b'CAL'


class ArduinoInterface:
    """
    Low-level interface for Arduino-based ultrasonic arrays.
    
    Handles serial communication, command protocol, and firmware
    interaction for ultrasonic sensor control.
    """
    
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 2.0
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        self.connection: Optional[serial.Serial] = None
        self.is_connected = False
        self.firmware_version = ""
        self.arduino_config: Dict[str, Any] = {}
        
    def connect(self) -> bool:
        """Establish connection to Arduino."""
        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            
            # Wait for Arduino boot
            time.sleep(2.0)
            
            # Test connection
            if self.ping():
                self.is_connected = True
                self._get_firmware_info()
                print(f"Connected to Arduino on {self.port} (FW: {self.firmware_version})")
                return True
            else:
                self.connection.close()
                self.connection = None
                return False
                
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino."""
        if self.connection and self.is_connected:
            self.connection.close()
            self.connection = None
            self.is_connected = False
            print("Disconnected from Arduino")
    
    def ping(self) -> bool:
        """Test Arduino communication."""
        try:
            response = self._send_command(ArduinoCommands.PING)
            return response == b'PONG'
        except:
            return False
    
    def reset(self) -> bool:
        """Reset Arduino."""
        try:
            response = self._send_command(ArduinoCommands.RESET)
            time.sleep(2.0)  # Wait for restart
            return response == b'RESET_OK'
        except:
            return False
    
    def configure_sensors(self, config: Dict[str, Any]) -> bool:
        """
        Configure sensor parameters on Arduino.
        
        Args:
            config: Configuration dictionary with sensor parameters
            
        Returns:
            True if configuration successful
        """
        try:
            # Serialize configuration
            config_data = self._serialize_config(config)
            
            # Send configuration command
            self._send_raw_command(ArduinoCommands.CONFIG.value)
            self._send_raw_data(config_data)
            
            response = self._read_response()
            success = response == b'CONFIG_OK'
            
            if success:
                self.arduino_config = config
                
            return success
            
        except Exception as e:
            print(f"Configuration failed: {e}")
            return False
    
    def transmit_chirp(
        self,
        sensor_id: int,
        chirp_data: bytes,
        frequency: float = 40000
    ) -> bool:
        """
        Transmit chirp through specified sensor.
        
        Args:
            sensor_id: Target sensor ID
            chirp_data: Chirp waveform data
            frequency: Transmission frequency
            
        Returns:
            True if transmission successful
        """
        try:
            # Prepare transmission parameters
            tx_params = struct.pack('<IBf', sensor_id, len(chirp_data), frequency)
            
            # Send transmission command
            self._send_raw_command(ArduinoCommands.TRANSMIT.value)
            self._send_raw_data(tx_params + chirp_data)
            
            response = self._read_response()
            return response == b'TX_OK'
            
        except Exception as e:
            print(f"Transmission failed: {e}")
            return False
    
    def receive_echoes(
        self,
        duration_ms: int,
        sample_rate: int = 250000,
        sensors: Optional[List[int]] = None
    ) -> Optional[bytes]:
        """
        Receive echo data from sensors.
        
        Args:
            duration_ms: Recording duration in milliseconds
            sample_rate: Sampling rate in Hz
            sensors: List of sensor IDs (None for all)
            
        Returns:
            Raw echo data bytes or None if failed
        """
        try:
            # Prepare receive parameters
            n_sensors = len(sensors) if sensors else 4  # Default 4 sensors
            sensor_mask = self._create_sensor_mask(sensors) if sensors else 0xFF
            
            rx_params = struct.pack('<HII', duration_ms, sample_rate, sensor_mask)
            
            # Send receive command
            self._send_raw_command(ArduinoCommands.RECEIVE.value)
            self._send_raw_data(rx_params)
            
            # Calculate expected data size
            n_samples = (duration_ms * sample_rate) // 1000
            expected_bytes = n_sensors * n_samples * 2  # 16-bit samples
            
            # Read echo data
            echo_data = self.connection.read(expected_bytes)
            
            if len(echo_data) != expected_bytes:
                print(f"Received {len(echo_data)} bytes, expected {expected_bytes}")
                return None
                
            return echo_data
            
        except Exception as e:
            print(f"Reception failed: {e}")
            return None
    
    def chirp_and_receive(
        self,
        chirp_data: bytes,
        duration_ms: int,
        tx_sensor: int = 0,
        rx_sensors: Optional[List[int]] = None,
        frequency: float = 40000
    ) -> Optional[bytes]:
        """
        Combined chirp transmission and echo reception.
        
        Args:
            chirp_data: Chirp waveform data
            duration_ms: Recording duration in milliseconds
            tx_sensor: Transmitting sensor ID
            rx_sensors: Receiving sensor IDs (None for all)
            frequency: Transmission frequency
            
        Returns:
            Raw echo data bytes or None if failed
        """
        try:
            # Prepare combined parameters
            n_rx_sensors = len(rx_sensors) if rx_sensors else 4
            rx_mask = self._create_sensor_mask(rx_sensors) if rx_sensors else 0xFF
            
            params = struct.pack('<IBfHI', 
                tx_sensor, len(chirp_data), frequency, duration_ms, rx_mask
            )
            
            # Send combined command
            self._send_raw_command(ArduinoCommands.CHIRP_RX.value)
            self._send_raw_data(params + chirp_data)
            
            # Calculate expected data size
            sample_rate = 250000  # Default sample rate
            n_samples = (duration_ms * sample_rate) // 1000
            expected_bytes = n_rx_sensors * n_samples * 2
            
            # Read echo data
            echo_data = self.connection.read(expected_bytes)
            
            if len(echo_data) != expected_bytes:
                print(f"Received {len(echo_data)} bytes, expected {expected_bytes}")
                return None
                
            return echo_data
            
        except Exception as e:
            print(f"Chirp and receive failed: {e}")
            return None
    
    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get Arduino status information."""
        try:
            response = self._send_command(ArduinoCommands.GET_STATUS)
            
            if response.startswith(b'STATUS:'):
                status_data = response[7:]  # Remove 'STATUS:' prefix
                return self._parse_status(status_data)
            
            return None
            
        except Exception as e:
            print(f"Failed to get status: {e}")
            return None
    
    def calibrate_sensors(self) -> bool:
        """Perform sensor calibration on Arduino."""
        try:
            response = self._send_command(ArduinoCommands.CALIBRATE)
            return response == b'CAL_OK'
        except:
            return False
    
    def _send_command(self, command: ArduinoCommands) -> bytes:
        """Send command and receive response."""
        if not self.connection or not self.is_connected:
            raise RuntimeError("Arduino not connected")
            
        self.connection.write(command.value + b'\\n')
        response = self.connection.readline().strip()
        return response
    
    def _send_raw_command(self, command: bytes):
        """Send raw command without newline."""
        if not self.connection or not self.is_connected:
            raise RuntimeError("Arduino not connected")
            
        self.connection.write(command)
    
    def _send_raw_data(self, data: bytes):
        """Send raw binary data."""
        if not self.connection or not self.is_connected:
            raise RuntimeError("Arduino not connected")
            
        self.connection.write(data)
    
    def _read_response(self) -> bytes:
        """Read response from Arduino."""
        if not self.connection:
            raise RuntimeError("Arduino not connected")
            
        return self.connection.readline().strip()
    
    def _get_firmware_info(self):
        """Get firmware version and info."""
        try:
            response = self._send_command(ArduinoCommands.GET_STATUS)
            if response.startswith(b'STATUS:'):
                status_data = response[7:]
                status = self._parse_status(status_data)
                self.firmware_version = status.get('firmware_version', 'Unknown')
        except:
            self.firmware_version = 'Unknown'
    
    def _serialize_config(self, config: Dict[str, Any]) -> bytes:
        """Serialize configuration dictionary to bytes."""
        # Simple binary serialization for Arduino
        data = bytearray()
        
        # Number of sensors
        data.extend(struct.pack('<I', config.get('n_sensors', 4)))
        
        # Sample rate
        data.extend(struct.pack('<I', config.get('sample_rate', 250000)))
        
        # Sensor configurations
        for i in range(config.get('n_sensors', 4)):
            sensor_config = config.get('sensors', [{}])[i] if i < len(config.get('sensors', [])) else {}
            
            # Sensor ID, trigger pin, echo pin, frequency
            data.extend(struct.pack('<IBBF',
                sensor_config.get('id', i),
                sensor_config.get('trigger_pin', 2 + i * 2),
                sensor_config.get('echo_pin', 3 + i * 2),
                sensor_config.get('frequency', 40000.0)
            ))
        
        return bytes(data)
    
    def _create_sensor_mask(self, sensor_ids: List[int]) -> int:
        """Create bitmask for sensor selection."""
        mask = 0
        for sensor_id in sensor_ids:
            if 0 <= sensor_id < 8:  # Support up to 8 sensors
                mask |= (1 << sensor_id)
        return mask
    
    def _parse_status(self, status_data: bytes) -> Dict[str, Any]:
        """Parse status response from Arduino."""
        # Simple status parsing - could be enhanced
        status_str = status_data.decode('utf-8', errors='ignore')
        
        status = {
            'firmware_version': '1.0.0',
            'uptime_ms': 0,
            'free_memory': 0,
            'sensors_active': 4
        }
        
        # Parse key-value pairs
        for pair in status_str.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key in status:
                    try:
                        if key == 'firmware_version':
                            status[key] = value
                        else:
                            status[key] = int(value)
                    except ValueError:
                        pass
        
        return status
    
    def generate_arduino_sketch(self, config: Dict[str, Any]) -> str:
        """
        Generate Arduino sketch code for sensor array.
        
        Args:
            config: Sensor array configuration
            
        Returns:
            Arduino sketch source code
        """
        n_sensors = config.get('n_sensors', 4)
        sample_rate = config.get('sample_rate', 250000)
        
        sketch = f'''
/*
 * EchoLoc-NN Ultrasonic Array Controller
 * Generated automatically for {n_sensors} sensors
 * Sample rate: {sample_rate} Hz
 */

#include <TimerOne.h>

// Configuration
#define N_SENSORS {n_sensors}
#define SAMPLE_RATE {sample_rate}
#define BUFFER_SIZE 4096

// Sensor pins
'''
        
        # Add sensor pin definitions
        sensors = config.get('sensors', [])
        for i in range(n_sensors):
            sensor = sensors[i] if i < len(sensors) else {}
            trigger_pin = sensor.get('trigger_pin', 2 + i * 2)
            echo_pin = sensor.get('echo_pin', 3 + i * 2)
            
            sketch += f'''
#define SENSOR_{i}_TRIG {trigger_pin}
#define SENSOR_{i}_ECHO {echo_pin}
'''
        
        sketch += '''
// Global variables
volatile int16_t adc_buffer[N_SENSORS][BUFFER_SIZE];
volatile int buffer_index = 0;
volatile bool recording = false;

void setup() {
  Serial.begin(115200);
  
  // Initialize sensor pins
'''
        
        # Add pin initialization
        for i in range(n_sensors):
            sketch += f'''
  pinMode(SENSOR_{i}_TRIG, OUTPUT);
  pinMode(SENSOR_{i}_ECHO, INPUT);
  digitalWrite(SENSOR_{i}_TRIG, LOW);
'''
        
        sketch += '''
  
  // Initialize ADC for high-speed sampling
  setupADC();
  
  Serial.println("EchoLoc-NN Array Ready");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\\n');
    processCommand(command);
  }
}

void processCommand(String cmd) {
  if (cmd == "PING") {
    Serial.println("PONG");
  }
  else if (cmd == "RESET") {
    Serial.println("RESET_OK");
    // Perform soft reset
    asm volatile ("  jmp 0");
  }
  else if (cmd.startsWith("TX:")) {
    // Handle transmission command
    handleTransmit(cmd);
  }
  else if (cmd.startsWith("RX:")) {
    // Handle reception command
    handleReceive(cmd);
  }
  else if (cmd.startsWith("CHIRP_RX:")) {
    // Handle combined chirp and receive
    handleChirpReceive(cmd);
  }
  else if (cmd == "STATUS") {
    printStatus();
  }
  else if (cmd == "CAL") {
    performCalibration();
  }
}

void handleTransmit(String cmd) {
  // Parse transmission parameters
  int sensor_id = 0; // Parse from command
  
  // Generate chirp on specified sensor
  generateChirp(sensor_id);
  
  Serial.println("TX_OK");
}

void handleReceive(String cmd) {
  // Parse receive parameters
  int duration_ms = 100; // Parse from command
  
  // Start ADC sampling
  startSampling(duration_ms);
  
  // Send sampled data
  sendSampleData();
}

void handleChirpReceive(String cmd) {
  // Combined chirp transmission and echo reception
  generateChirp(0); // Sensor 0 as transmitter
  delayMicroseconds(100); // Brief delay
  startSampling(100); // 100ms recording
  sendSampleData();
}

void generateChirp(int sensor_id) {
  // Generate 40kHz chirp on specified sensor
  int trig_pin = SENSOR_0_TRIG + sensor_id * 2;
  
  // 5ms LFM chirp (simplified)
  for (int i = 0; i < 200; i++) {
    digitalWrite(trig_pin, HIGH);
    delayMicroseconds(12); // ~40kHz
    digitalWrite(trig_pin, LOW);
    delayMicroseconds(12);
  }
}

void startSampling(int duration_ms) {
  buffer_index = 0;
  recording = true;
  
  // Start timer-based ADC sampling
  Timer1.initialize(1000000 / SAMPLE_RATE); // Microseconds per sample
  Timer1.attachInterrupt(adcSample);
  
  // Wait for sampling to complete
  delay(duration_ms);
  
  Timer1.stop();
  recording = false;
}

void adcSample() {
  if (!recording || buffer_index >= BUFFER_SIZE) return;
  
  // Sample all sensors simultaneously
'''
        
        # Add ADC sampling for each sensor
        for i in range(n_sensors):
            echo_pin = f"SENSOR_{i}_ECHO"
            sketch += f'''
  adc_buffer[{i}][buffer_index] = analogRead({echo_pin}) - 512; // Center around 0
'''
        
        sketch += '''
  
  buffer_index++;
}

void sendSampleData() {
  // Send binary data for all sensors
  for (int sensor = 0; sensor < N_SENSORS; sensor++) {
    for (int sample = 0; sample < buffer_index; sample++) {
      Serial.write((uint8_t*)&adc_buffer[sensor][sample], 2);
    }
  }
}

void setupADC() {
  // Configure ADC for high-speed sampling
  // Set ADC prescaler to 16 (1MHz ADC clock)
  ADCSRA = (ADCSRA & ~0x07) | 0x04;
  
  // Set reference to AVCC
  ADMUX = (1 << REFS0);
}

void printStatus() {
  Serial.print("STATUS:firmware_version=1.0.0,");
  Serial.print("uptime_ms=");
  Serial.print(millis());
  Serial.print(",free_memory=");
  Serial.print(freeMemory());
  Serial.print(",sensors_active=");
  Serial.println(N_SENSORS);
}

void performCalibration() {
  // Basic sensor calibration
  Serial.println("CAL_OK");
}

int freeMemory() {
  char top;
  extern char *__brkval;
  return __brkval ? &top - __brkval : &top - __malloc_heap_start;
}
'''
        
        return sketch.strip()
    
    def upload_sketch(self, sketch_code: str, sketch_path: str = "echoloc_array") -> bool:
        """
        Upload Arduino sketch (requires arduino-cli).
        
        Args:
            sketch_code: Arduino sketch source code
            sketch_path: Path for sketch directory
            
        Returns:
            True if upload successful
        """
        import os
        import subprocess
        
        try:
            # Create sketch directory
            os.makedirs(sketch_path, exist_ok=True)
            
            # Write sketch file
            with open(f"{sketch_path}/{sketch_path}.ino", 'w') as f:
                f.write(sketch_code)
            
            # Compile sketch
            compile_cmd = [
                "arduino-cli", "compile", 
                "--fqbn", "arduino:avr:uno",
                sketch_path
            ]
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Compilation failed: {result.stderr}")
                return False
            
            # Upload sketch
            upload_cmd = [
                "arduino-cli", "upload",
                "-p", self.port,
                "--fqbn", "arduino:avr:uno",
                sketch_path
            ]
            
            result = subprocess.run(upload_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Upload failed: {result.stderr}")
                return False
                
            print(f"Successfully uploaded sketch to {self.port}")
            return True
            
        except Exception as e:
            print(f"Sketch upload failed: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()