import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from filters import resample_uniform
from preprocess import CSIpreprocessor

LO, HI = 0.1, 0.6  # 호흡 대역 (6 ~ 36 bpm)


def clean_phase(csi, timestamps_us):
    """
    raw 복소 CSI -> 보정된 위상 (서브캐리어별, 균일 시간축)

    - CSIpreprocessor로 이득오차/STO/CPE 제거 (Optimal Preprocessing 논문)
    - 널 서브캐리어 제외
    - local_timestamp 기준 균일 간격으로 리샘플링
    """
    live = ~(np.abs(csi) == 0).all(axis=0)
    corrected, _, _, _ = CSIpreprocessor(csi)._remove_phase_error_LoS()
    C = corrected[:, live]

    # re, _, fs = resample_uniform(C.real, timestamps_us)
    # im, _, _ = resample_uniform(C.imag, timestamps_us)
    Z, _, fs = resample_uniform(C, timestamps_us)

    # return np.unwrap(np.angle(re + 1j * im), axis=0), fs
    return np.unwrap(np.angle(Z), axis=0), fs


def bandpass(x, fs, lo=LO, hi=HI, order=4):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x - x.mean(axis=0), axis=0)


def bnr(x, fs):
    """
    Breathing-to-Noise Ratio = 호흡 대역 파워 / 전체 파워.
    단순 대역 파워로 고르면 그 대역 노이즈가 센 서브캐리어까지 뽑히므로 비율을 쓴다.
    """
    fr, pw = welch(x, fs=fs, nperseg=min(1024, len(x)))
    b = (fr >= LO) & (fr <= HI)
    return pw[b].sum() / (pw.sum() + 1e-20)


def fuse_weights(F, fs, thresh=0.7):
    """
    BNR >= thresh * max(BNR)인 서브캐리어만 선택, BNR을 가중치로 반환.
    단순 평균은 서브캐리어마다 호흡 위상 부호가 달라 서로 상쇄된다.
    """
    scores = np.array([bnr(F[:, k], fs) for k in range(F.shape[1])])
    sel = scores >= thresh * scores.max()
    return sel, scores[sel]


def fuse(M, sel, w):
    return (M[:, sel] * w).sum(axis=1) / w.sum()


def rate_autocorr(x, fs):
    """
    자기상관의 첫 주요 피크 lag로 주기를 구한다.
    저SNR에서 스펙트럼 피크보다 강건하다 (주기 성분이 여러 주기에 걸쳐 누적됨).
    """
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1 :]
    ac = ac / (ac[0] + 1e-20)

    lo, hi = int(fs / HI), int(fs / LO)  # lag 범위 = 주기 범위
    seg = ac[lo : min(hi, len(ac))]
    if len(seg) < 3:
        return np.nan, 0.0

    pk, _ = find_peaks(seg)
    if len(pk) == 0:
        return np.nan, 0.0

    best = pk[np.argmax(seg[pk])]
    return 60.0 * fs / (lo + best), seg[best]


def band_snr(x, fs, f_true, half_width=0.02):
    """
    논문 지표: 참 주파수 ±half_width 파워 / 대역 밖 파워.
    잡음만 있어도 (2*half_width)/(HI-LO) 비율만큼은 나오므로 그 바닥값과 비교해야 한다.
    """
    fr, pw = welch(x, fs=fs, nperseg=min(2048, len(x)))
    m = (fr >= LO) & (fr <= HI)
    fr, pw = fr[m], pw[m]

    inb = np.abs(fr - f_true) <= half_width
    snr = pw[inb].sum() / (pw[~inb].sum() + 1e-20)
    floor = (2 * half_width) / (HI - LO - 2 * half_width)
    return snr, floor


def rate_psd(x, fs):
    """대역 내 PSD 최대 지점. 신호가 실제로 있을 때 자기상관보다 정확했다."""
    fr, pw = welch(x, fs=fs, nperseg=min(2048, len(x)))
    b = (fr >= LO) & (fr <= HI)
    return fr[b][np.argmax(pw[b])] * 60, pw[b].max() / np.median(pw[b])


def estimate(csi, timestamps_us, f_true=None):
    """
    전체 파이프라인. f_true(Hz)를 주면 SNR도 같이 계산.

    주의: 신호가 없어도 추정기는 항상 어떤 값을 내놓는다 (빈 방에서도 25~35bpm).
    f_true 없이 단독으로 쓸 때는 acorr/psd_pm 같은 강도 지표를 같이 보고
    임계값 아래면 '감지 없음'으로 처리해야 한다. 임계값은 아직 미정.
    """
    PH, fs = clean_phase(csi, timestamps_us)
    F = bandpass(PH, fs)

    sel, w = fuse_weights(F, fs)
    x_bp = fuse(F, sel, w)  # 대역통과 후
    x_raw = fuse(PH, sel, w)  # 대역통과 전 (스펙트럼/SNR용)

    bpm, psd_pm = rate_psd(x_raw, fs)
    bpm_ac, acorr = rate_autocorr(x_bp, fs)

    out = {
        "fs": fs,
        "n_sel": int(sel.sum()),
        "bpm": bpm,  # 주 추정치 (PSD)
        "bpm_autocorr": bpm_ac,  # 참고용
        "psd_pm": psd_pm,
        "acorr": acorr,
        "waveform": x_bp,
    }
    if f_true is not None:
        # SNR은 대역통과 전 신호로 재야 한다 (필터가 스펙트럼을 깎으면 지표가 왜곡됨)
        out["snr"], out["snr_floor"] = band_snr(x_raw, fs, f_true)
    return out
