"""Synthetic quality measurements with mixture-conditioned contribution tracing.

Run python -m stereo_surround.quality. No real voice recordings are used.
"""

import json

import numpy as np
from scipy.fft import rfft
from scipy.signal import butter, sosfilt, sosfreqz

from .decoder import SurroundDecoder


def render(x, **settings):
    d = SurroundDecoder(**settings)
    pieces = [d.process(x[i:i+800]) for i in range(0, len(x), 800)]
    pieces.append(d.flush())
    return np.concatenate(pieces)[d.latency_samples:]


class MixtureProbe(SurroundDecoder):
    """Trace known components using statistics computed ONLY from their mixture."""

    def __init__(self, components, *, intervals=(), **settings):
        super().__init__(**settings)
        self.components = components
        self.frame_number = 0
        self.powers = {name: [0.0, 0.0] for name in components}
        self.interval_powers = [(a, b, {name: [0.0, 0.0] for name in components})
                                for a, b in intervals]
        self._component_histories = {name: np.zeros_like(self._history) for name in components}

    def _frame(self):
        super()._frame()
        h, n = self.hop_size, self.fft_size
        start = self.frame_number*h
        self.frame_number += 1
        # 分離器は入力統計で変わるため、成分を単独で処理した結果を足しても
        # 混合音の処理結果にはならない。混合音から得たRを固定し、同じ線形演算を
        # 既知の各成分へ適用して、混合条件での直接音漏れと残響保持を別々に測る。
        # この測定はSTFT上のリア演算の寄与で、後段のHPFや遅延を含まない。
        pl, pr = self._power
        r = self._cross
        gap = np.hypot(pl-pr, 2*np.abs(r))
        lo = np.maximum(0, (pl+pr-gap)*.5)
        hi = np.maximum((pl+pr+gap)*.5, np.float32(1e-30))
        wll, wrr = 1-(pl-lo)/hi, 1-(pr-lo)/hi
        wlr, wrl = -r/hi, -r.conj()/hi
        if self.mode == "matrix":
            rho = r.real/np.maximum(np.sqrt(pl)*np.sqrt(pr), np.float32(1e-30))
            q = np.maximum(-np.clip(rho, -1, 1), 0)**2
            wll, wrr = (1-q)*wll+q, (1-q)*wrr+q
            wlr, wrl = (1-q)*wlr, (1-q)*wrl
        observed = rfft(self._history*self._window, axis=1)
        expected = np.stack((wll*observed[0]+wlr*observed[1],
                             wrl*observed[0]+wrr*observed[1]))
        # 測定用に再構成した演算が本体のリア出力と一致するかも同時に検査する。
        # 本体の分離式を変えたとき、古い測定式だけで良い結果を出すことを防ぐ。
        tolerance = 2e-6*max(1, float(np.max(np.abs(observed))))
        if np.max(np.abs(expected-self._spectra[4:])) > tolerance:
            raise AssertionError("Contribution tracer does not match the actual rear operator")
        for name, component in self.components.items():
            history = self._component_histories[name]
            history[:, :-h] = history[:, h:]
            history[:, -h:] = 0
            part = component[start:start+h]
            history[:, -h:-h+len(part) if len(part) < h else None] = part.T
            if start < 12000 or start+h > len(component):
                continue
            z = rfft(history*self._window, axis=1)
            zl, zr = wll*z[0]+wlr*z[1], wrl*z[0]+wrr*z[1]
            # 片側スペクトルの電力を足すので、DC/Nyquist以外は共役側の分を2倍する。
            # 同じ窓・FFT長の電力比ではParsevalの1/N係数が相殺される。
            # https://numpy.org/doc/stable/reference/routines.fft.html#real-and-hermitian-transforms
            weights = np.full(n//2+1, 2.0)
            weights[[0, -1]] = 1
            original = float(np.sum(weights*(np.abs(z[0])**2+np.abs(z[1])**2)))
            residual = float(np.sum(weights*(np.abs(zl)**2+np.abs(zr)**2)))
            self.powers[name][0] += original
            self.powers[name][1] += residual
            center = start-(n-h)+n//2
            for a, b, measurements in self.interval_powers:
                if a <= center < b:
                    measurements[name][0] += original
                    measurements[name][1] += residual


def measure(statistics_ms=160, onset_adaptation=False):
    rng = np.random.default_rng(20261007)
    rate, length = 48000, 96000
    t = np.arange(length)/rate
    # Harmonic surrogate with time-dependent amplitude, not a speech recording.
    envelope = (.65+.35*np.sin(2*np.pi*3*t))*(.8+.2*np.sin(2*np.pi*.7*t))
    voice = sum(np.sin(2*np.pi*137*h*t)/h for h in range(1, 26))*envelope
    voice = voice.astype(np.float32)
    voice *= np.float32(.1/np.sqrt(np.mean(voice**2)))
    ambient = sosfilt(butter(2, [300, 7000], fs=rate, btype="bandpass", output="sos"),
                      rng.normal(size=(length, 2)), axis=0).astype(np.float32)
    ambient *= np.float32(.1/np.sqrt(np.mean(ambient**2)))
    settings = dict(output_gain=1, rear_gain=1, rear_highpass=None, rear_delay_ms=0,
                    statistics_ms=statistics_ms, onset_adaptation=onset_adaptation)
    delayed = []
    for milliseconds in (.2, 1, 3):
        delay = round(rate*milliseconds/1000)
        right = np.concatenate((np.zeros(delay), voice[:-delay])).astype(np.float32)
        x = np.column_stack((voice, right))
        y = render(x, **settings)
        reference = np.column_stack(((x[:, 0]-x[:, 1])/np.sqrt(2),)*2)
        input_power = np.sum(x[12000:]**2)
        delayed.append({"delay_ms": milliseconds,
                        "rear_leakage_db": float(10*np.log10(np.sum(y[12000:, 4:]**2)/input_power)),
                        "mid_side_leakage_db": float(10*np.log10(np.sum(reference[12000:]**2)/input_power))})
    mixture_results = []
    for db in (-10, 0, 10):
        primary = np.column_stack((voice, .5*voice))
        primary *= np.float32(np.sqrt(np.mean(ambient**2)/np.mean(primary**2))*10**(db/20))
        mixture = primary+ambient
        d = MixtureProbe({"primary": primary, "ambient": ambient}, **settings)
        for start in range(0, length, 800):
            d.process(mixture[start:start+800])
        mixture_results.append({"primary_ambient_db": db,
                                "primary_leakage_db": float(10*np.log10(d.powers["primary"][1]/d.powers["primary"][0])),
                                "ambient_power_retention": d.powers["ambient"][1]/d.powers["ambient"][0]})
    d = SurroundDecoder()
    _, hp = sosfreqz(d._hp, worN=[80, 1000], fs=rate)
    _, lp = sosfreqz(d._lp, worN=[60, 1000], fs=rate)
    primary = np.column_stack((voice, voice))
    primary[:24000] = 0
    probe = MixtureProbe({"primary": primary, "ambient": ambient},
                         intervals=((24000, 26400), (26400, 36000), (36000, 48000)), **settings)
    onset_mixture = primary+ambient
    for start in range(0, length, 800):
        probe.process(onset_mixture[start:start+800])
    transients = [{"after_onset_ms": [(a-24000)/48, (b-24000)/48],
                   "primary_leakage_db": float(10*np.log10(p["primary"][1]/p["primary"][0]))}
                  for a, b, p in probe.interval_powers]
    return {"synthetic_only": True, "rate": rate, "profile": "balanced",
            "statistics_ms": statistics_ms, "delayed_voice": delayed,
            "onset_adaptation": onset_adaptation,
            "mixtures_spectral_before_filters": mixture_results,
            "onset_spectral_frame_center_intervals": transients,
            "rear_hp_80_vs_1000_db": float(20*np.log10(abs(hp[0]/hp[1]))),
            "lfe_lp_60_db": float(20*np.log10(abs(lp[0]))),
            "lfe_lp_1000_db": float(20*np.log10(abs(lp[1])))}


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
