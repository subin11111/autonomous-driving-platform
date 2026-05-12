import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String

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

    def readline(self):
        return b''

    def close(self):
        if not self._closed:
            self._closed = True
            self._logger.info('[mock-serial] closed')


class McuSerialBridge(Node):
    def __init__(self):
        super().__init__('mcu_serial_bridge')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('linear_deadband', 0.05)
        self.declare_parameter('angular_deadband', 0.05)
        self.declare_parameter('watchdog_timeout', 0.5)
        self.declare_parameter('serial_timeout', 0.01)
        self.declare_parameter('write_timeout', 0.01)
        self.declare_parameter('send_center_command', True)
        self.declare_parameter('mock_serial', False)

        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.linear_deadband = float(self.get_parameter('linear_deadband').value)
        self.angular_deadband = float(self.get_parameter('angular_deadband').value)
        self.watchdog_timeout = float(self.get_parameter('watchdog_timeout').value)
        self.serial_timeout = float(self.get_parameter('serial_timeout').value)
        self.write_timeout = float(self.get_parameter('write_timeout').value)
        self.send_center_command = bool(self.get_parameter('send_center_command').value)
        self.mock_serial = bool(self.get_parameter('mock_serial').value)

        self.status_pub = self.create_publisher(String, '/vehicle/mcu_status', 10)
        self.tx_pub = self.create_publisher(String, '/vehicle/mcu_tx', 10)

        self.cmd_sub = self.create_subscription(Twist, self.cmd_topic, self.cmd_callback, 10)
        self.estop_sub = self.create_subscription(Bool, '/vehicle/estop', self.estop_callback, 10)

        self.serial_conn = self._open_serial()

        self.last_cmd_time: Optional[float] = None
        self.last_sent_drive: Optional[str] = None
        self.last_sent_steer: Optional[str] = None
        self.estop_active = False
        self.watchdog_triggered = False

        self.read_timer = self.create_timer(0.02, self._read_serial)
        self.watchdog_timer = self.create_timer(0.05, self._watchdog_check)

        # Start from a safe stopped state.
        self._send_stop(reason='startup', force=True)

        self.get_logger().info(
            f'mcu_serial_bridge started: port={self.port}, baudrate={self.baudrate}, '
            f'cmd_topic={self.cmd_topic}, watchdog_timeout={self.watchdog_timeout}, mock_serial={self.mock_serial}'
        )

    def _open_serial(self):
        if self.mock_serial:
            self.get_logger().info('mock_serial=true: using mock serial backend')
            return MockSerial(self.get_logger())

        if serial is None:
            raise RuntimeError('pyserial is not installed. Install python3-serial or set mock_serial=true.')

        try:
            conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.serial_timeout,
                write_timeout=self.write_timeout,
            )
            return conn
        except Exception as exc:
            raise RuntimeError(f'failed to open serial port {self.port}: {exc}') from exc

    def _publish_tx(self, cmd_char: str, reason: str):
        msg = String()
        msg.data = f'{cmd_char} ({reason})'
        self.tx_pub.publish(msg)

    def _write_command(self, cmd_char: str, reason: str, force: bool = False):
        if (not force) and cmd_char == ' ' and self.last_sent_drive == ' ':
            return

        try:
            self.serial_conn.write(cmd_char.encode('ascii'))
            self._publish_tx(cmd_char, reason)
        except (SerialException, OSError) as exc:
            self.get_logger().error(f'serial write failed for {cmd_char!r}: {exc}')

    def _send_stop(self, reason: str, force: bool = False):
        self._write_command(' ', reason=reason, force=force)
        self.last_sent_drive = ' '
        self.last_sent_steer = 'C'

    def _map_drive(self, linear_x: float) -> str:
        if linear_x > self.linear_deadband:
            return 'W'
        if linear_x < -self.linear_deadband:
            return 'S'
        return ' '

    def _map_steer(self, angular_z: float) -> Optional[str]:
        if angular_z > self.angular_deadband:
            return 'A'
        if angular_z < -self.angular_deadband:
            return 'D'
        if self.send_center_command:
            return 'C'
        return None

    def estop_callback(self, msg: Bool):
        new_state = bool(msg.data)
        if new_state and not self.estop_active:
            self.get_logger().warn('E-stop activated. Sending immediate stop command.')
            self._send_stop(reason='estop', force=True)
        elif (not new_state) and self.estop_active:
            self.get_logger().info('E-stop released.')

        self.estop_active = new_state

    def cmd_callback(self, msg: Twist):
        self.last_cmd_time = time.monotonic()
        self.watchdog_triggered = False

        if self.estop_active:
            # In E-stop mode, never allow motion commands to pass through.
            self._send_stop(reason='estop_hold', force=True)
            return

        drive_cmd = self._map_drive(msg.linear.x)
        steer_cmd = self._map_steer(msg.angular.z)

        if drive_cmd == ' ':
            if self.last_sent_drive != ' ':
                self._send_stop(reason='cmd_vel_stop', force=False)
            return

        state_changed = (drive_cmd != self.last_sent_drive) or (steer_cmd != self.last_sent_steer)
        if not state_changed:
            return

        # Send drive first, steer second.
        self._write_command(drive_cmd, reason='cmd_vel_drive', force=False)
        self.last_sent_drive = drive_cmd

        if steer_cmd is not None:
            self._write_command(steer_cmd, reason='cmd_vel_steer', force=False)
            self.last_sent_steer = steer_cmd

    def _watchdog_check(self):
        if self.estop_active:
            return

        now = time.monotonic()
        if self.last_cmd_time is None:
            if self.last_sent_drive != ' ':
                self._send_stop(reason='watchdog_init', force=True)
            return

        elapsed = now - self.last_cmd_time
        if elapsed > self.watchdog_timeout:
            if not self.watchdog_triggered:
                self.get_logger().warn(f'Watchdog timeout ({elapsed:.3f}s). Sending stop command.')
                self._send_stop(reason='watchdog_timeout', force=True)
                self.watchdog_triggered = True

    def _read_serial(self):
        try:
            waiting = int(getattr(self.serial_conn, 'in_waiting', 0))
        except Exception:
            waiting = 0

        if waiting <= 0:
            return

        # Drain all currently available lines with non-blocking serial timeout.
        while waiting > 0:
            try:
                raw = self.serial_conn.readline()
            except (SerialException, OSError) as exc:
                self.get_logger().error(f'serial read failed: {exc}')
                return

            if not raw:
                return

            line = raw.decode('utf-8', errors='ignore').strip()
            if line:
                msg = String()
                msg.data = line
                self.status_pub.publish(msg)

            try:
                waiting = int(getattr(self.serial_conn, 'in_waiting', 0))
            except Exception:
                waiting = 0

    def shutdown(self):
        self._send_stop(reason='shutdown', force=True)

        try:
            self.serial_conn.close()
        except Exception as exc:
            self.get_logger().warn(f'serial close failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = McuSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()