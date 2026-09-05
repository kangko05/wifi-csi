import numpy as np

"""
Serial Data Format
type,
seq,
mac,
rssi,
rate,
noise_floor,
fft_gain,
agc_gain,
channel,
local_timestamp,
sig_len,
rx_format,
len,
first_word,
data
"""

FIELDS = [
    "type",
    "seq",
    "mac",
    "rssi",
    "rate",
    "noise_floor",
    "fft_gain",
    "agc_gain",
    "channel",
    "local_timestamp",
    "sig_len",
    "rx_format",
    "len",
    "first_word",
    "data",
]

INT_FIELDS = (
    "seq",
    "rssi",
    "rate",
    "noise_floor",
    "fft_gain",
    "agc_gain",
    "channel",
    "local_timestamp",
    "sig_len",
    "rx_format",
    "len",
    "first_word",
)


def parse_line(line, data_tag):
    if not line.startswith(data_tag):
        return None

    row = line.split(",")
    if len(row) < len(FIELDS):
        raise ValueError(f"expected {len(FIELDS)} fields, got {len(row)}")

    d = dict(zip(FIELDS, row))

    for key in INT_FIELDS:
        d[key] = int(d[key])

    data_str = ",".join(row[len(FIELDS) - 1 :]).strip('"[]')
    data = np.array(data_str.split(","), dtype=float)
    d["data"] = data[1::2] + 1j * data[0::2]

    return d
