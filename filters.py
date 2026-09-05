import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt


def hampel_filter(series: pd.Series, window_size=5, n_sigmas=3):
    """
    hampel filter
    - 현재 시점 기준, 주위 데이터들의 중앙값과 현재 값의 차이를 비교해 이상치인지 판단 (오차와 오차의 표준편차로 판단)
    - 이상치라고 판단되는 경우 중앙값으로 대체
    - 이상치를 제거할 목적으로 적용
    """

    L = 1.4826  # gaussian dist standard scale factor

    rolling_median = series.rolling(window=window_size, center=True).median()

    def look_window_mad(window):
        median_val = np.median(window)
        return np.median(np.abs(window - median_val))

    rolling_mad = series.rolling(window=window_size, center=True).apply(
        look_window_mad, raw=True
    )

    threshold = n_sigmas * L * rolling_mad

    difference = np.abs(series - rolling_median)

    is_outlier = difference > threshold

    cleaned_series = series.copy()
    cleaned_series[is_outlier] = rolling_median[is_outlier]

    return cleaned_series, series[is_outlier].index.tolist()


def resample_uniform(matrix: np.ndarray, timestamps_us: np.ndarray, fs=None):
    """
    리샘플링 (불균일 샘플링 보정)
    - local_timestamp(패킷이 실제 도착한 시각, us) 기준으로 선형보간해서
      일정한 간격(fs)의 시간축으로 다시 샘플링
    - 패킷 드롭/지터로 샘플 간격이 들쭉날쭉하면 FFT/필터가 가정하는
      "균일 샘플링"이 깨지기 때문에, 분석 전에 맞춰주는 용도
    - fs를 안 넘기면 실측 평균 간격으로 자동 계산해서 사용
    """

    t = (timestamps_us - timestamps_us[0]) / 1e6  # 초 단위, 0부터 시작

    if fs is None:
        fs = 1.0 / np.mean(np.diff(t))

    uniform_t = np.arange(t[0], t[-1], 1.0 / fs)

    resampled = np.column_stack(
        [np.interp(uniform_t, t, matrix[:, i]) for i in range(matrix.shape[1])]
    )

    return resampled, uniform_t, fs


def butterworth_filter(series: pd.Series, lowcut=0.1, highcut=0.5, fs=100.0, order=4):
    """
    butterworth filter (bandpass)
    - lowcut ~ highcut 주파수 대역만 통과시키고 나머지 성분은 제거
    - sosfiltfilt로 정방향/역방향 두 번 적용해서 위상 지연 없이 필터링
    - 호흡처럼 특정 주기 대역의 신호만 뽑아낼 목적으로 적용
    - b,a(전달함수) 대신 SOS(2차 섹션)로 설계 - 저주파 좁은 대역에서 b,a는
      수치 불안정(계수가 발산)해서 필터가 깨짐
    """

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    sos = butter(order, [low, high], btype="band", output="sos")

    filtered = sosfiltfilt(sos, series.to_numpy())

    return pd.Series(filtered, index=series.index)
