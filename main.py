import serial, threading, time, respiration, numpy as np
from collections import deque
from parse_csi import parse_line

WINDOW_SEC = 60
BUFFER = deque(maxlen=WINDOW_SEC * 200)  # 60sec * 200Hz
LOCK = threading.Lock()


def read_csi(
    stop_event: threading.Event,
    TAG="CSI_DATA",
    PORT="COM3",
    BAUD=921600,
):
    """
    read csi data from port
    """

    ser = serial.Serial(PORT, BAUD, timeout=1)

    while not stop_event.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            data = parse_line(line, TAG)

            if data is None:
                continue

            now = time.time_ns()
            ts = data["local_timestamp"]
            data = data["data"]

            with LOCK:
                BUFFER.append((now, ts, data))

                while BUFFER[0][0] < BUFFER[-1][0] - WINDOW_SEC * 1e9:  # ns
                    BUFFER.popleft()

        except Exception as e:
            print(e)

    ser.close()


if __name__ == "__main__":
    stop_ev = threading.Event()

    read_thread = threading.Thread(
        target=read_csi,
        args=(stop_ev,),
        daemon=True,
    )

    read_thread.start()

    # main thread ============================================================

    while True:
        try:
            time.sleep(1)

            with LOCK:
                if not BUFFER:
                    continue

                ts = np.array([t for _, t, _ in BUFFER])
                data = np.stack([d for _, _, d in BUFFER])

            if data:
                respiration.estimate(data, ts)

        except KeyboardInterrupt:
            break

    stop_ev.set()
    read_thread.join()
