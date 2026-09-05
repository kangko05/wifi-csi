# claude generated

import os
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from serial import Serial

from parse_csi import parse_line

PORT = "COM3"
BAUD = 921600
WINDOW_SEC = 30
FS = 50.0

N = int(WINDOW_SEC * FS)
buf_t = deque(maxlen=N)
buf_amp = deque(maxlen=N)
buf_pha = deque(maxlen=N)

log_rows = []
log_csi = []
markers = []

lock = threading.Lock()
stop_ev = threading.Event()
t0 = time.time()
n_seen = 0


def reader():
    global n_seen
    try:
        ser = Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print(f"!! serial open failed: {e}")
        return
    print(f"reading {PORT} @ {BAUD}")

    while not stop_ev.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            d = parse_line(line, "CSI_DATA")
            if d is None:
                continue

            csi = d["data"]
            amp = np.abs(csi)
            live = amp > 0
            if not live.any():
                continue

            # CFO/SFO는 패킷 내 모든 서브캐리어에 공통 -> 기준 대비 위상차로 상쇄
            # (한가운데는 널 서브캐리어라 가장 센 서브캐리어를 기준으로 씀)
            ref = csi[np.argmax(amp)]
            rel = (csi * np.conj(ref))[live]

            with lock:
                buf_t.append(time.time() - t0)
                buf_amp.append(amp[live].mean())
                buf_pha.append(np.angle(rel).mean())
                log_csi.append(csi)
                log_rows.append(
                    {k: v for k, v in d.items() if k != "data"} | {"wall": time.time()}
                )
                n_seen += 1
                if n_seen % 200 == 0:
                    print(f"  {n_seen} packets")

        except (ValueError, IndexError):
            continue
        except Exception as e:
            print(f"!! reader error: {e}")
            break

    ser.close()


def on_key(event):
    if event.key is None:
        return
    with lock:
        markers.append((time.time() - t0, event.key))
    print(f"[marker] t={time.time() - t0:6.1f}s  key={event.key}")


def save_log():
    with lock:
        if not log_rows:
            print("no data to save")
            return
        rows, csi, mk = list(log_rows), np.stack(log_csi), list(markers)

    os.makedirs("data", exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csvpath, npypath = f"data/rt_{now}.csv", f"data/rt_{now}.npy"

    df = pd.DataFrame(rows)
    df["data"] = [npypath] * len(df)
    np.save(npypath, csi)
    df.to_csv(csvpath)
    if mk:
        pd.DataFrame(mk, columns=["elapsed_s", "label"]).to_csv(
            f"data/rt_{now}_markers.csv", index=False
        )
    print(f"saved {len(df)} rows -> {csvpath}   markers: {len(mk)}")


fig, (ax_a, ax_p) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fig.suptitle("CSI realtime  (key press = marker)")
fig.canvas.mpl_connect("key_press_event", on_key)

(ln_a,) = ax_a.plot([], [], lw=1.0)
ax_a.set_ylabel("amplitude")
ax_a.grid(alpha=0.3)

(ln_p,) = ax_p.plot([], [], lw=1.0, color="tab:green")
ax_p.set_ylabel("phase (rad)")
ax_p.set_xlabel("t (s)")
ax_p.grid(alpha=0.3)


def update(_):
    with lock:
        t = np.array(buf_t)
        a = np.array(buf_amp)
        p = np.array(buf_pha)

    if len(t) < 2:
        return

    p = np.unwrap(p)

    ln_a.set_data(t, a)
    ln_p.set_data(t, p)

    for ax, y in ((ax_a, a), (ax_p, p)):
        ax.set_xlim(t[0], max(t[-1], t[0] + 1))
        pad = (y.max() - y.min()) * 0.1 + 1e-9
        ax.set_ylim(y.min() - pad, y.max() + pad)


if __name__ == "__main__":
    th = threading.Thread(target=reader, daemon=True)
    th.start()

    ani = FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)
    plt.show()

    stop_ev.set()
    th.join(timeout=2)
    save_log()
