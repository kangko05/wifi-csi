from serial import Serial
from datetime import datetime
import pandas as pd
import numpy as np
import threading
import queue
import os
import time

from parse_csi import parse_line


def read_port(
    q: queue.Queue, stop: threading.Event, DATA_TAG="CSI_DATA", port="COM3", baud=921600
):
    print(f"reading {port}")

    ser = Serial(port, baud, timeout=1)

    while not stop.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            d = parse_line(line, DATA_TAG)
            if d is None:
                continue

            q.put(d)

            # print(d["data"])

        except KeyboardInterrupt:
            break
        except (ValueError, IndexError) as e:
            # print(f"skipping malformed line ({e}): {line!r}")
            continue

    q.task_done()
    ser.close()


if __name__ == "__main__":
    time.sleep(15)

    q = queue.Queue()
    stop_ev = threading.Event()

    read_serial_port = threading.Thread(
        target=read_port,
        args=(
            q,
            stop_ev,
        ),
    )

    read_serial_port.start()

    # main thread

    DURATION_SEC = 60

    start = time.time()
    while time.time() - start < DURATION_SEC:
        time.sleep(1)

    stop_ev.set()

    actual_cnt = input("enter actual respiration count: ")

    os.makedirs("data", exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join("data", f"{now}_{actual_cnt}.csv")
    datapath = os.path.join("data", f"{now}_{actual_cnt}.npy")
    batch_size = q.qsize()
    data = [q.get() for _ in range(batch_size)]

    df = pd.DataFrame(data)
    npy = np.stack(df["data"].to_numpy())
    df["data"] = [datapath] * df.shape[0]

    np.save(datapath, npy)
    df.to_csv(outpath)

    print("ok")

    # ==============================

    read_serial_port.join()

    print("terminating...")
