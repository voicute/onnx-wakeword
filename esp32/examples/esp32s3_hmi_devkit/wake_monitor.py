#!/usr/bin/env python3
"""Wake Word Monitor — 连接 ESP32-S3，实时显示唤醒状态。用法: python wake_monitor.py [COM端口]"""

import serial, time, sys, re

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM5'
BAUD = 115200

def main():
    print(f'=== Wake Monitor ({PORT}) ===')
    print(f'说关键词触发唤醒 | Ctrl+C 退出\n')

    while True:
        try:
            ser = serial.Serial(PORT, BAUD, timeout=0.3)
            ser.dtr = True
            time.sleep(0.3)
            ser.reset_input_buffer()
            break
        except:
            print(f'等待 {PORT}...')
            time.sleep(2)

    print(f'已连接，监听中...\n')

    last_prob = 0.0
    try:
        while True:
            try:
                n = ser.in_waiting
                if n > 0:
                    data = ser.read(n)
                    text = data.decode('utf-8', errors='replace')
                    lines = text.replace('\r', '').split('\n')
                    for line in lines:
                        if not line.strip():
                            continue
                        # Highlight wake (from callback line)
                        if 'WAKE' in line:
                            print(f'\n*** 🔊 WAKE! 关键词检测! prob={last_prob:.3f} ***\n')
                        # Show prob on log lines
                        elif 'prob=' in line:
                            pm = re.search(r'prob=([\d.]+).*rms=([\d.]+)', line)
                            im = re.search(r'invoke=([\d.]+)ms', line)
                            if pm:
                                p, r = float(pm.group(1)), int(float(pm.group(2)))
                                last_prob = p
                                inv = f' invoke={float(im.group(1)):.1f}ms' if im else ''
                                bar_len = min(40, int(p * 50))
                                bar = '█' * bar_len + '░' * (40 - bar_len)
                                print(f'  prob={p:.4f} [{bar}] rms={r}{inv}', end='\r')
                        elif 'APP: === Ready ===' in line:
                            print('  [系统就绪，开始监听...]')
                        elif 'Buffer ready' in line:
                            print('  [音频缓冲就绪]')
                        elif any(k in line for k in ('CMD', 'MultiNet', 'loop#', '>>>', 'MN:', 'MN ', 'L5 ', 'L5_', 'DetectLogic', 'SKIP', 'JUMP', 'quiet', 'steady')):
                            print(f'  {line.strip()}')
                else:
                    time.sleep(0.05)
            except serial.SerialException:
                print('\n[串口断开，重连中...]')
                ser.close()
                time.sleep(2)
                while True:
                    try:
                        ser = serial.Serial(PORT, BAUD, timeout=0.3)
                        ser.dtr = True
                        ser.reset_input_buffer()
                        print('[已重连]')
                        break
                    except:
                        time.sleep(2)
    except KeyboardInterrupt:
        print('\n退出')
    finally:
        ser.close()

if __name__ == '__main__':
    main()
