import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String

try:
    import serial
    from serial import SerialException
except Exception:  # pragma: no cover - handled by runtime checks
    serial = None

    class SerialException(Exception):
        pass


class MockSerial:
    def __init__(self, node_logger):
        self._logger = node_logger
        self._closed = False

    @property
    def in_waiting(self):
        return 0

    def write(self, data: bytes):
        self._logger.info(f'[mock-serial] write: {data!r}')

    def flush(self):
        pass

    def readline(self):
        return b''

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

    def close(self):
        if not self._closed:
            self._closed = True
            self._logger.info('[mock-serial] closed')


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def _safe_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if text in ('0', 'false', 'no', 'n', 'off'):
        return False

    return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _sanitize_state(value) -> str:
    raw_state = str(value).strip().upper()
    if not raw_state:
        return 'UNKNOWN'

    sanitized = []
    previous_was_underscore = False

    for char in raw_state:
        if char.isalnum():
            sanitized.append(char)
            previous_was_underscore = False
        elif char.isspace() or char in ('-', '_'):
            if not previous_was_underscore:
                sanitized.append('_')
                previous_was_underscore = True

    state = ''.join(sanitized).strip('_')
    return state or 'UNKNOWN'


class McuSerialBridge(Node):
    """
    Numeric-only MCU serial bridge.

    ROS input:
      /desired_speed                 std_msgs/Float64, m/s
      /desired_steering_angle_deg    std_msgs/Float64, degree
      /behavior_state                std_msgs/String

    Serial output:
      CMD,<speed_mps>,<steering_deg>,<behavior_state>\\n
      STOP\\n
      ESTOP\\n
    """

    def __init__(self):
        super().__init__('mcu_serial_bridge')

        # Kept for compatibility with launch/param checks.
        # This implementation intentionally supports numeric_direct only.
        self.declare_parameter('input_mode', 'numeric_direct')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('mock_serial', False)

        # Keep timeout short so the ROS executor is not blocked for long.
        self.declare_parameter('serial_timeout', 0.02)
        self.declare_parameter('write_timeout', 0.10)

        # Arduino usually resets when the USB serial port is opened.
        # Raw Python test worked because it waited before writing.
        self.declare_parameter('arduino_boot_wait_s', 2.5)

        # Periodic command publish rate.
        self.declare_parameter('command_publish_period_s', 0.05)

        # Input topics.
        self.declare_parameter('desired_speed_topic', '/desired_speed')
        self.declare_parameter('desired_steering_angle_deg_topic', '/desired_steering_angle_deg')
        self.declare_parameter('behavior_state_topic', '/behavior_state')

        # Clamp limits.
        self.declare_parameter('max_abs_speed_mps', 1.40)
        self.declare_parameter('max_abs_steering_deg', 20.0)
        self.declare_parameter('allow_reverse', True)

        # Safety states.
        self.declare_parameter(
            'stop_states',
            ['STOP', 'EMERGENCY_STOP', 'ESTOP', 'RED_LIGHT', 'OBSTACLE_STOP'],
        )

        # Input freshness checks.
        self.declare_parameter('speed_input_timeout_s', 0.5)
        self.declare_parameter('steering_input_timeout_s', 0.5)
        self.declare_parameter('state_input_timeout_s', 0.5)

        # Serial read behavior.
        self.declare_parameter('read_period_s', 0.02)
        self.declare_parameter('read_max_lines_per_tick', 20)

        # Optional debugging behavior.
        self.declare_parameter('publish_startup_stop', True)
        self.declare_parameter('flush_after_write', True)

        # Parameters
        self.input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        if self.input_mode != 'numeric_direct':
            self.get_logger().warn(
                f'input_mode={self.input_mode!r} requested, but this bridge is numeric_direct only. '
                'Forcing numeric_direct.'
            )
            self.input_mode = 'numeric_direct'

        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.mock_serial = _safe_bool(self.get_parameter('mock_serial').value, False)

        self.serial_timeout = max(
            0.0,
            _safe_float(self.get_parameter('serial_timeout').value, 0.02),
        )
        self.write_timeout = max(
            0.0,
            _safe_float(self.get_parameter('write_timeout').value, 0.10),
        )
        self.arduino_boot_wait_s = max(
            0.0,
            _safe_float(self.get_parameter('arduino_boot_wait_s').value, 2.5),
        )

        self.command_publish_period_s = max(
            0.001,
            _safe_float(self.get_parameter('command_publish_period_s').value, 0.05),
        )

        self.desired_speed_topic = self.get_parameter(
            'desired_speed_topic'
        ).get_parameter_value().string_value
        self.desired_steering_angle_deg_topic = self.get_parameter(
            'desired_steering_angle_deg_topic'
        ).get_parameter_value().string_value
        self.behavior_state_topic = self.get_parameter(
            'behavior_state_topic'
        ).get_parameter_value().string_value

        self.max_abs_speed_mps = abs(
            _safe_float(self.get_parameter('max_abs_speed_mps').value, 1.40)
        )
        self.max_abs_steering_deg = abs(
            _safe_float(self.get_parameter('max_abs_steering_deg').value, 20.0)
        )
        self.allow_reverse = _safe_bool(self.get_parameter('allow_reverse').value, True)

        self.stop_states = {
            _sanitize_state(state)
            for state in self.get_parameter('stop_states').value
        }

        self.speed_input_timeout_s = max(
            0.0,
            _safe_float(self.get_parameter('speed_input_timeout_s').value, 0.5),
        )
        self.steering_input_timeout_s = max(
            0.0,
            _safe_float(self.get_parameter('steering_input_timeout_s').value, 0.5),
        )
        self.state_input_timeout_s = max(
            0.0,
            _safe_float(self.get_parameter('state_input_timeout_s').value, 0.5),
        )

        self.read_period_s = max(
            0.001,
            _safe_float(self.get_parameter('read_period_s').value, 0.02),
        )
        self.read_max_lines_per_tick = max(
            1,
            int(self.get_parameter('read_max_lines_per_tick').value),
        )

        self.publish_startup_stop = _safe_bool(
            self.get_parameter('publish_startup_stop').value,
            True,
        )
        self.flush_after_write = _safe_bool(
            self.get_parameter('flush_after_write').value,
            True,
        )

        # Publishers
        self.status_pub = self.create_publisher(String, '/vehicle/mcu_status', 10)
        self.tx_pub = self.create_publisher(String, '/vehicle/mcu_tx', 10)

        # Serial
        self.serial_conn = self._open_serial()

        # State
        self.estop_active = False
        self.last_desired_speed = 0.0
        self.last_steering_deg = 0.0
        self.last_behavior_state = 'STOP'

        self.last_speed_time: Optional[float] = None
        self.last_steering_time: Optional[float] = None
        self.last_state_time: Optional[float] = None

        self.last_numeric_safety_reason: Optional[str] = None

        # Subscribers
        self.speed_sub = self.create_subscription(
            Float64,
            self.desired_speed_topic,
            self.desired_speed_callback,
            10,
        )
        self.steering_sub = self.create_subscription(
            Float64,
            self.desired_steering_angle_deg_topic,
            self.desired_steering_angle_deg_callback,
            10,
        )
        self.behavior_state_sub = self.create_subscription(
            String,
            self.behavior_state_topic,
            self.behavior_state_callback,
            10,
        )
        self.estop_sub = self.create_subscription(
            Bool,
            '/vehicle/estop',
            self.estop_callback,
            10,
        )

        # Timers
        self.read_timer = self.create_timer(self.read_period_s, self._read_serial)
        self.numeric_command_timer = self.create_timer(
            self.command_publish_period_s,
            self._publish_numeric_command,
        )

        if self.publish_startup_stop:
            self._send_numeric_stop(reason='startup')

        self.get_logger().info(
            'mcu_serial_bridge started: '
            f'input_mode=numeric_direct, port={self.port}, baudrate={self.baudrate}, '
            f'mock_serial={self.mock_serial}, allow_reverse={self.allow_reverse}, '
            f'arduino_boot_wait_s={self.arduino_boot_wait_s:.2f}, '
            f'command_publish_period_s={self.command_publish_period_s:.3f}'
        )
        self.get_logger().info(
            'numeric_direct subscriptions: '
            f'{self.desired_speed_topic}, {self.desired_steering_angle_deg_topic}, '
            f'{self.behavior_state_topic}'
        )

    # ------------------------------------------------------------
    # Serial
    # ------------------------------------------------------------

    def _open_serial(self):
        if self.mock_serial:
            self.get_logger().info('mock_serial=true: using mock serial backend')
            return MockSerial(self.get_logger())

        if serial is None:
            raise RuntimeError(
                'pyserial is not installed. Install python3-serial or set mock_serial=true.'
            )

        try:
            conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.serial_timeout,
                write_timeout=self.write_timeout,
            )

            # Important:
            # Many Arduino boards reset when the serial port is opened.
            # Wait like the successful raw Python test did.
            if self.arduino_boot_wait_s > 0.0:
                self.get_logger().info(
                    f'opened {self.port}; waiting {self.arduino_boot_wait_s:.2f}s for Arduino boot'
                )
                time.sleep(self.arduino_boot_wait_s)

            # Do NOT reset input buffer here.
            # Keeping it allows boot messages to be published on /vehicle/mcu_status.
            try:
                conn.reset_output_buffer()
            except Exception:
                pass

            return conn

        except Exception as exc:
            raise RuntimeError(f'failed to open serial port {self.port}: {exc}') from exc

    def _publish_tx(self, payload: str, reason: str):
        display = payload.rstrip('\n')
        msg = String()
        msg.data = f'{display} ({reason})'
        self.tx_pub.publish(msg)

    def _publish_status_line(self, line: str):
        msg = String()
        msg.data = line
        self.status_pub.publish(msg)

    def _write_serial(self, payload: str, reason: str):
        try:
            data = payload.encode('ascii')
            self.serial_conn.write(data)

            if self.flush_after_write:
                try:
                    self.serial_conn.flush()
                except Exception:
                    pass

            self._publish_tx(payload, reason)

        except (SerialException, OSError) as exc:
            self.get_logger().error(f'serial write failed for {payload!r}: {exc}')

    def _read_serial(self):
        # Raw Python worked by calling readline() repeatedly.
        # Do the same here; do not rely only on in_waiting.
        for _ in range(self.read_max_lines_per_tick):
            try:
                raw = self.serial_conn.readline()
            except (SerialException, OSError) as exc:
                self.get_logger().error(f'serial read failed: {exc}')
                return

            if not raw:
                return

            line = raw.decode('utf-8', errors='replace').strip()
            if line:
                self._publish_status_line(line)

    # ------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------

    def desired_speed_callback(self, msg: Float64):
        self.last_desired_speed = _safe_float(msg.data, 0.0)
        self.last_speed_time = time.monotonic()

    def desired_steering_angle_deg_callback(self, msg: Float64):
        self.last_steering_deg = _safe_float(msg.data, 0.0)
        self.last_steering_time = time.monotonic()

    def behavior_state_callback(self, msg: String):
        self.last_behavior_state = _sanitize_state(msg.data)
        self.last_state_time = time.monotonic()

    def estop_callback(self, msg: Bool):
        new_state = bool(msg.data)

        if new_state:
            if not self.estop_active:
                self.get_logger().warn('E-stop activated. Sending immediate ESTOP.')
            self.estop_active = True
            self._send_numeric_estop(reason='estop')
            return

        if self.estop_active:
            self.get_logger().info('E-stop released.')

        self.estop_active = False

    # ------------------------------------------------------------
    # Numeric command logic
    # ------------------------------------------------------------

    def _send_numeric_command(self, speed_mps: float, steering_deg: float, behavior_state: str):
        payload = f'CMD,{speed_mps:.3f},{steering_deg:.2f},{behavior_state}\n'
        self._write_serial(payload, reason='numeric_direct')

    def _send_numeric_stop(self, reason: str):
        self._write_serial('STOP\n', reason=reason)

    def _send_numeric_estop(self, reason: str):
        self._write_serial('ESTOP\n', reason=reason)

    def _is_timeout(self, last_time: Optional[float], timeout_s: float, now: float) -> bool:
        if last_time is None:
            return True
        return (now - last_time) > timeout_s

    def _numeric_stop_reason(self, now: float) -> Optional[str]:
        if self._is_timeout(self.last_speed_time, self.speed_input_timeout_s, now):
            return 'speed_timeout'

        if self._is_timeout(self.last_steering_time, self.steering_input_timeout_s, now):
            return 'steering_timeout'

        if self._is_timeout(self.last_state_time, self.state_input_timeout_s, now):
            return 'state_timeout'

        if self.last_behavior_state in self.stop_states:
            return 'state_stop'

        return None

    def _log_numeric_safety_reason(self, reason: Optional[str]):
        if reason == self.last_numeric_safety_reason:
            return

        if reason in ('speed_timeout', 'steering_timeout', 'state_timeout'):
            self.get_logger().warn(f'numeric_direct safety stop: {reason}')
        elif reason == 'state_stop':
            self.get_logger().info(f'numeric_direct STOP state: {self.last_behavior_state}')
        elif reason is None and self.last_numeric_safety_reason is not None:
            self.get_logger().info('numeric_direct command stream recovered')

        self.last_numeric_safety_reason = reason

    def _publish_numeric_command(self):
        if self.estop_active:
            self._send_numeric_estop(reason='estop_hold')
            return

        now = time.monotonic()
        stop_reason = self._numeric_stop_reason(now)

        if stop_reason is not None:
            self._log_numeric_safety_reason(stop_reason)
            self._send_numeric_stop(reason=stop_reason)
            return

        self._log_numeric_safety_reason(None)

        speed_mps = _clamp(
            self.last_desired_speed,
            -self.max_abs_speed_mps,
            self.max_abs_speed_mps,
        )

        if not self.allow_reverse and speed_mps < 0.0:
            speed_mps = 0.0

        steering_deg = _clamp(
            self.last_steering_deg,
            -self.max_abs_steering_deg,
            self.max_abs_steering_deg,
        )

        self._send_numeric_command(speed_mps, steering_deg, self.last_behavior_state)

    # ------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------

    def shutdown(self):
        try:
            self._send_numeric_stop(reason='shutdown')
        except Exception:
            pass

        try:
            self.serial_conn.close()
        except Exception as exc:
            self.get_logger().warn(f'serial close failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = McuSerialBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.shutdown()
            except Exception:
                pass
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()