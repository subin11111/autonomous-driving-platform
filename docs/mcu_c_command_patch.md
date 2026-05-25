# MCU C Command Patch Guide

현재 저장소에는 MCU .ino 원본 파일이 없으므로, 아래 패치를 MCU 펌웨어의 switch(command) 구문에 적용하세요.

추가 코드:

case 'C':
  mcp.setChannelValue(MCP4728_CHANNEL_A, STEER_CENTER);
  Serial.println(">> 조향: 중앙 (1.6V 인가 완료)");
  break;

주의사항:

- W, S, A, D, Space는 Arduino 단독 수동 테스트용으로만 유지
- `mcu_serial_bridge`는 W/S/A/D/Space를 보내지 않고 `CMD,...`, `STOP`, `ESTOP` 라인만 전송
- 전진/후진 기어 전환 시 기존 딜레이와 안전 로직 유지
- 기어 전환 시 throttle 0 선행 로직 제거 금지
- Space는 기존대로 throttle 0V, brake 0.5V, steering center 수행
- C는 조향만 중앙으로 변경하고 throttle/brake/gear는 변경하지 않음
