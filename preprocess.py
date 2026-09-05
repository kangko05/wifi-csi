import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class CSIpreprocessor:
    """
    input data = 2d np array

    CLASS VARIABLES

    hpk_tilde = raw csi data
    hpk_gr = gain error removed data
    gain_error = calculated gain error
    """

    def __init__(self, data: np.ndarray, Ts=3.2e-6, phase_unwrap_window=3):
        self.P, self.K = data.shape
        self.hpk_tilde = data
        self.phase_unwrap_window = phase_unwrap_window

        k = np.arange(self.K)

        self.TS = Ts
        self.fk = k / self.TS
        self.hpk_gr, self.gain_error = self._remove_gain_error()

    def _remove_gain_error(self):
        g_hat = np.sqrt(np.mean(np.square(np.abs(self.hpk_tilde)), axis=-1))
        h_pk = self.hpk_tilde / g_hat[:, None]

        return h_pk, g_hat

    def _clean_hpk(self, tau, mu, gp):
        """
        remove all errors (phase & gain errors) from input data

        hpk_tilde = raw csi
        tau = timing error, STO
        mu = common phase error, CPE
        gp = gain error
        """

        # 패킷 축 브로드캐스팅 (fk: (K,), tau/mu/gp: (P,))
        hpk_hat = (
            self.hpk_tilde
            * np.exp(1j * 2 * np.pi * np.outer(tau, self.fk))
            * np.exp(1j * mu)[:, None]
        ) / gp[:, None]

        # --- 원본 (루프) ---
        # hpk_hat = np.zeros((self.P, self.K), dtype=complex)
        # for p in range(self.P):
        #     hpk_hat[p] = (
        #         self.hpk_tilde[p]
        #         * np.exp(1j * 2 * np.pi * self.fk * tau[p])
        #         * np.exp(1j * mu[p])
        #     ) / gp[p]

        return hpk_hat

    def _phase_unwrap(self, omega_pk):
        # 서브캐리어 축 이동합(moving sum). 범위 밖은 0으로 잘리므로
        # mode='same' 컨볼루션과 정확히 동일하다.
        w = self.phase_unwrap_window
        S = np.convolve(omega_pk, np.ones(2 * w + 1), mode="same")

        # --- 원본 (루프) ---
        # K = len(omega_pk)
        # S = np.zeros(K, dtype=complex)
        # for k in range(K):
        #     lo = max(0, k - self.phase_unwrap_window)
        #     hi = min(K, k + self.phase_unwrap_window+1)
        #     S[k] = np.sum(omega_pk[lo:hi])

        return np.unwrap(np.angle(S))

    def _phase_unwrap_robust(self, omega_pk):
        unwrapped = self._phase_unwrap(omega_pk)
        angle_raw = np.angle(omega_pk)

        return np.mod(angle_raw - unwrapped + np.pi, 2 * np.pi) - np.pi + unwrapped

    def _phase_unwrap_robust_batch(self, OM):
        """
        _phase_unwrap_robust를 (P, M) 배열에 한 번에 적용.
        이동합은 누적합 차분으로 구한다 (범위 밖은 0 -> 제로패딩과 동일).
        """
        w = self.phase_unwrap_window
        P, M = OM.shape

        pad = np.zeros((P, w), dtype=OM.dtype)
        Xp = np.concatenate([pad, OM, pad], axis=1)
        C = np.concatenate([np.zeros((P, 1), dtype=OM.dtype), np.cumsum(Xp, axis=1)], axis=1)
        S = C[:, 2 * w + 1 :] - C[:, : M]

        unwrapped = np.unwrap(np.angle(S), axis=1)
        angle_raw = np.angle(OM)
        return np.mod(angle_raw - unwrapped + np.pi, 2 * np.pi) - np.pi + unwrapped

    def _remove_phase_error_baseline(self):
        """
        baseline method for extracting tau & mu, which represents STO & CPE
        need gain error removed data
        """
        P, K = self.hpk_gr.shape[0], self.hpk_gr.shape[1]

        # 패킷 축을 한 번에 (axis=1이 서브캐리어)
        tau_hat = np.angle(
            np.sum(self.hpk_gr[:, :-1] * np.conj(self.hpk_gr[:, 1:]), axis=1)
        ) * (self.TS / (2 * np.pi))

        mu_hat = -np.angle(
            np.sum(self.hpk_gr * np.exp(1j * 2 * np.pi * np.outer(tau_hat, self.fk)), axis=1)
        )

        # --- 원본 (루프) ---
        # tau_hat = np.zeros(P); mu_hat = np.zeros(P)
        # for p in range(P):
        #     tau_hat[p] = np.angle(
        #         np.sum(self.hpk_gr[p, :-1] * np.conj(self.hpk_gr[p, 1:]))
        #     ) * (self.TS / (2 * np.pi))
        #     mu_hat[p] = np.angle(
        #         np.sum(self.hpk_gr[p, :] * np.exp(1j*2*np.pi*self.fk*tau_hat[p]))
        #     ) * (-1)

        hpk_hat = self._clean_hpk(tau_hat, mu_hat, self.gain_error)

        return (hpk_hat, tau_hat, mu_hat)

    def _remove_phase_error_LoS(self):
        P, K = self.hpk_gr.shape[0], self.hpk_gr.shape[1]

        tau_bar = np.angle(
            np.sum(self.hpk_gr[:, :-1] * np.conj(self.hpk_gr[:, 1:]), axis=1)
        ) * (self.TS / (2 * np.pi))

        mu_bar = -np.angle(
            np.sum(self.hpk_gr * np.exp(1j * 2 * np.pi * np.outer(tau_bar, self.fk)), axis=1)
        )

        # (25) ============================================================
        # phase[p,k] = 2*pi*fk[k]*tau_bar[p] + mu_bar[p]
        phase = 2 * np.pi * np.outer(tau_bar, self.fk) + mu_bar[:, None]
        bk = (self.hpk_gr * np.exp(1j * phase)).mean(axis=0)

        omega_pk = (
            np.conj(self.hpk_gr)
            * bk[None, :]
            * np.exp(-1j * 2 * np.pi * np.outer(tau_bar, self.fk))
        )

        # --- 원본 (루프) ---
        # tau_bar = np.zeros(P); mu_bar = np.zeros(P)
        # for p in range(P):
        #     tau_bar[p] = np.angle(np.sum(self.hpk_gr[p,:-1]*np.conj(self.hpk_gr[p,1:]))) * (self.TS/(2*np.pi))
        #     mu_bar[p]  = np.angle(np.sum(self.hpk_gr[p,:]*np.exp(1j*2*np.pi*self.fk*tau_bar[p]))) * (-1)
        # bk = np.zeros(K, dtype=complex)
        # for kk in range(K):
        #     bk[kk] = (1/P)*np.sum(self.hpk_gr[:,kk]*np.exp(1j*(2*np.pi*self.fk[kk]*tau_bar + mu_bar)))
        # omega_pk = np.zeros((P,K), dtype=complex)
        # for p in range(P):
        #     omega_pk[p] = np.conj(self.hpk_gr[p,:])*bk*np.exp(-1j*2*np.pi*self.fk*tau_bar[p])

        valid = np.abs(bk) ** 2 > 0.1
        idx = np.where(valid)[0]

        # ---- 여기부터 최소제곱으로 tau_hat, psi_hat 구하기 ----
        x = self.fk[idx]

        OM = omega_pk[:, idx]  # (P, M)
        Y = self._phase_unwrap_robust_batch(OM)  # (P, M)
        W = np.abs(OM)

        Sw = W.sum(axis=1)
        Swx = (W * x).sum(axis=1)
        Swy = (W * Y).sum(axis=1)
        Swxx = (W * x**2).sum(axis=1)
        Swxy = (W * x * Y).sum(axis=1)

        a = (Sw * Swxy - Swx * Swy) / (Sw * Swxx - Swx**2)
        b = (Swy - a * Swx) / Sw

        tau_hat = tau_bar + a / (2 * np.pi)
        psi_hat = b

        # --- 원본 (루프) ---
        # tau_hat = np.zeros(P); psi_hat = np.zeros(P)
        # for p in range(P):
        #     om = omega_pk[p, idx]
        #     y = self._phase_unwrap_robust(om)
        #     w = np.abs(om)
        #     Sw=np.sum(w); Swx=np.sum(w*x); Swy=np.sum(w*y)
        #     Swxx=np.sum(w*x**2); Swxy=np.sum(w*x*y)
        #     a = (Sw*Swxy - Swx*Swy)/(Sw*Swxx - Swx**2)
        #     b = (Swy - a*Swx)/Sw
        #     tau_hat[p] = tau_bar[p] + a/(2*np.pi)
        #     psi_hat[p] = b

        # 마지막 보정된 hpk (3)
        hpk_hat = self._clean_hpk(tau_hat, psi_hat, self.gain_error)

        return (hpk_hat, tau_hat, psi_hat, bk)


if __name__ == "__main__":
    # sample3 - 가까이서 1.78 -> 0.02
    # sample4 - 거리 좀 띄우고 중간에 노트북 1.75 -> 0.074
    # sample5 - 거리 대략 1m 정도 띄우고 중간에 칸막이 1.78 -> 0.064

    samples = ["sample3", "sample4", "sample5"]

    for sample in samples:
        data = np.load(f"{sample}.npy")
        prep = CSIpreprocessor(data)
        result, _, _, _ = prep._remove_phase_error_LoS()

        # h_pk_raw: (n_frames, n_subcarriers) 원본
        # h_pk_corrected: 보정 후

        phase_raw = np.angle(data)
        phase_corrected = np.angle(result)

        std_before = np.std(phase_raw, axis=0)  # 서브캐리어별, 프레임간 std
        std_after = np.std(phase_corrected, axis=0)

        print(f"{sample}: 보정 전 평균 std:", np.mean(std_before))
        print(f"{sample}: 보정 후 평균 std:", np.mean(std_after))
        print()

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "polar"})

        # sc = 10

        # for ax, data, title in zip(
        #     axes, [data[:, sc], result[:, sc]], ["보정 전", "보정 후"]
        # ):
        #     r = np.abs(data)
        #     theta = np.angle(data)
        #     ax.scatter(theta, r, s=10, alpha=0.6)
        #     ax.set_title(title)

        # plt.tight_layout()
        # plt.savefig("phase_alignment.png", dpi=150)
        # plt.show()
