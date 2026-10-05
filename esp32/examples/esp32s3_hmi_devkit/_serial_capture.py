import serial, time, sys

port, secs = sys.argv[1], int(sys.argv[2])
out = sys.argv[3]
s = serial.Serial(port, 115200, timeout=1)
end = time.time() + secs
with open(out, "w", encoding="utf-8", errors="replace") as f:
    while time.time() < end:
        line = s.readline().decode("utf-8", errors="replace").rstrip()
        if line:
            print(line, flush=True)
            f.write(line + "\n")
s.close()
