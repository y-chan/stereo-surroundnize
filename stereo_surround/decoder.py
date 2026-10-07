"""Causal STFT upmix using a covariance model and fixed startup latency."""

from collections import deque
import math
import operator

import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import butter, sosfilt

CHANNELS = ("FL", "FR", "FC", "LFE", "SL", "SR")
PROFILES = {"low_latency": 512, "balanced": 1024, "quality": 2048}


class SurroundDecoder:
    """Process normalized float audio in arbitrary sized blocks.

    process returns exactly as many samples as supplied, including N startup
    zeros. flush returns N + tail_samples samples. Remove the first N samples
    from the concatenated output to obtain time-aligned finite-file audio.
    Configuration is fixed until a new instance is created; not thread-safe.
    """

    def __init__(self, sample_rate=48000, *, profile="balanced", mode="ambience",
                 center_gain=0.85, rear_gain=0.7, lfe_gain=0.5,
                 output_gain=0.5, rear_delay_ms=None, rear_highpass=250.0,
                 lfe_lowpass=120.0, statistics_ms=160.0, onset_adaptation=False,
                 listening_mode="standard", center_width=0.0,
                 rear_shelf_db=None, rear_shelf_hz=3000.0):
        if profile not in PROFILES or mode not in ("ambience", "matrix"):
            raise ValueError("Unknown profile or mode")
        if listening_mode not in ("standard", "music"):
            raise ValueError("listening_mode must be standard or music")
        sample_rate = float(sample_rate)
        if not math.isfinite(sample_rate) or not 8000 <= sample_rate <= 192000:
            raise ValueError("sample_rate must be between 8000 and 192000 Hz")
        self.sample_rate = sample_rate
        self.mode = mode
        self.listening_mode = listening_mode
        center_width = float(center_width)
        if not math.isfinite(center_width) or not 0 <= center_width <= 1:
            raise ValueError("center_width must be between 0 and 1")
        self.center_width = center_width
        # 用途による違いはリアの追加遅延と高域の音色に限定する。
        # Noneのときだけ既定値を選び、明示した0 msや0 dBは上書きしない。
        # musicでも下の共分散モデル、低域HPF、STFTの固定遅延は共通。
        if rear_delay_ms is None:
            rear_delay_ms = 0.0 if listening_mode == "music" else 10.0
        if rear_shelf_db is None:
            rear_shelf_db = -3.0 if listening_mode == "music" else 0.0
        rear_shelf_db, rear_shelf_hz = float(rear_shelf_db), float(rear_shelf_hz)
        if not math.isfinite(rear_shelf_db) or not -12 <= rear_shelf_db <= 0:
            raise ValueError("rear_shelf_db must be between -12 and 0 dB")
        if not math.isfinite(rear_shelf_hz) or not 0 < rear_shelf_hz < sample_rate/2:
            raise ValueError("rear_shelf_hz must be between zero and Nyquist")
        self.rear_shelf_db, self.rear_shelf_hz = rear_shelf_db, rear_shelf_hz
        self.fft_size = PROFILES[profile]
        # 外部から渡されるブロック長とは独立に、内部はN/4サンプルで進む。
        # 外部の分割を変えても統計更新と窓の位置を同じにするための固定hop。
        self.hop_size = self.fft_size // 4
        self.latency_samples = self.fft_size
        for name, value in (("center_gain", center_gain), ("rear_gain", rear_gain),
                            ("lfe_gain", lfe_gain), ("output_gain", output_gain)):
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 4:
                raise ValueError(f"{name} must be finite and between 0 and 4")
            setattr(self, "_" + name, value)
        if not math.isfinite(statistics_ms) or statistics_ms <= 0:
            raise ValueError("statistics_ms must be positive and finite")
        if not math.isfinite(rear_delay_ms) or not 0 <= rear_delay_ms <= 100:
            raise ValueError("rear_delay_ms must be between 0 and 100")
        self.rear_delay_ms = float(rear_delay_ms)
        self._delay_size = round(sample_rate * rear_delay_ms / 1000)
        # EMA: R_new = alpha*R_old + (1-alpha)*R_frame。
        # alphaを秒ではなく「1 hopの経過時間」から求め、rateやNを変えても
        # 指定した時定数を保つ。1ブロックにつき1回更新してはいけない。
        self._alpha = np.float32(math.exp(-self.hop_size / (sample_rate * statistics_ms / 1000)))
        self._onset_adaptation = bool(onset_adaptation)
        self._attack_alpha = np.float32(math.exp(-self.hop_size / (sample_rate*.020)))
        frequencies = np.arange(self.fft_size//2+1)*sample_rate/self.fft_size
        self._steering_band = (frequencies >= 100) & (frequencies <= 7000)
        # 解析・合成の両方にsqrt(periodic Hann)を掛ける。
        # 積はHannで、N/4ずつずらした4窓の和は2となるため、合成側で0.5倍する。
        # np.hanning(N)の対称窓では端点の定義が異なるので置き換えない。
        # 周期HannとCOLAの説明（関数自体は呼ばず、窓を直接生成する）：
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_COLA.html
        self._window = np.sqrt(0.5 - 0.5*np.cos(2*np.pi*np.arange(self.fft_size)/self.fft_size)).astype(np.float32)
        self._hp = self._filter(rear_highpass, "highpass")
        self._lp = self._filter(lfe_lowpass, "lowpass")
        self._shelf = self._high_shelf(rear_shelf_db, rear_shelf_hz)
        sections = [s for s in (self._hp, self._shelf) if s is not None]
        # SOS行を連結するとHPFとシェルフの直列接続になる。
        # 周波数ごとのマスクではなく時間領域の因果フィルタなので、窓が短くても
        # 低域の遮断を保てる。IIRの群遅延はlatency_samplesとは別に残る。
        self._rear_filter = np.vstack(sections) if sections else None
        self.reset()

    def _high_shelf(self, gain_db, frequency):
        """First-order high shelf: unity DC, requested Nyquist gain, dB midpoint.

        Bilinear transform of (g*s + w*sqrt(g))/(s + w*sqrt(g)),
        prewarped at frequency. This is our filter, not Dolby's coefficients.
        """
        if gain_db == 0:
            return None
        # gは電力比ではなく振幅比。アナログH(s)の極をw*sqrt(g)へ置くことで、
        # s=j*wでの振幅がsqrt(g)、すなわちgain_db/2 dBとなる。
        g = 10**(gain_db/20)
        # s = 2*fs*(1-z^-1)/(1+z^-1)の双一次変換と周波数プリワープ。
        # 分母の先頭係数1+kで正規化し、1次式をSOSの6係数へ埋め込む。
        # 変換の定義：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear.html
        # この実装のプリワープと係数は上のH(s)から直接計算する。
        k = math.tan(math.pi*frequency/self.sample_rate)*math.sqrt(g)
        return np.array([[(g+k)/(1+k), (k-g)/(1+k), 0,
                          1, (k-1)/(1+k), 0]], dtype=np.float32)

    def _filter(self, cutoff, kind):
        if cutoff is None:
            return None
        if not math.isfinite(cutoff) or not 0 < cutoff < self.sample_rate/2:
            raise ValueError("Filter cutoff must be between zero and Nyquist, or None")
        # cutoffはfsと同じHz単位の-3 dB点。高次多項式のbaではなくSOSを使う。
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        return butter(4, cutoff, btype=kind, fs=self.sample_rate, output="sos").astype(np.float32)

    def reset(self):
        """Clear all audio, statistics, filters, counters, and termination state."""
        n, h = self.fft_size, self.hop_size
        # 公開APIは(samples, channels)、FFT用履歴は(channels, samples)。
        # pendingには次のhopだけを貯め、fillの範囲だけが有効な入力となる。
        # history、OLA、統計、IIR、リア遅延を一緒に初期化し、シーク前の信号を消す。
        self._history = np.zeros((2, n), np.float32)
        self._pending = np.empty((h, 2), np.float32)
        self._fill = 0
        self._ola = np.zeros((6, n), np.float32)
        self._power = np.zeros((2, n//2+1), np.float32)
        self._cross = np.zeros(n//2+1, np.complex64)
        self._spectra = np.empty((6, n//2+1), np.complex64)
        # 初期のSTFTフレームには時刻0より前のゼロがN-hサンプル含まれる。
        # その区間のOLA出力を捨てるskipと、公開出力の先頭へ置くN個のゼロは別物。
        # N個のゼロが固定遅延の契約を作り、hop未満の呼び出しにも出力を供給する。
        self._skip = n-h
        self._queue = deque([np.zeros((n, 6), np.float32)])
        self._queue_offset = 0
        # sosfiltのziは(sections, 2 delay states, channels)。
        # リア2chの状態は独立に持ち、LFEの1ch状態は(sections, 2)となる。
        # zi/zfの形状：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html
        self._rear_state = None if self._rear_filter is None else np.zeros((len(self._rear_filter), 2, 2), np.float32)
        self._lp_state = None if self._lp is None else np.zeros((len(self._lp), 2), np.float32)
        self._delay = np.zeros((self._delay_size, 2), np.float32)
        self._finished = False
        self.input_samples = 0
        self.peak = 0.0
        self.overrange_samples = 0
        self._attack_remaining = 0

    def _frame(self):
        # 以下の推定はFFTの各ビンへ独立に適用する。
        # xは瞬時の複素振幅、Rは平滑化した共分散であり、同じ量ではない。
        h = self.hop_size
        self._history[:, :-h] = self._history[:, h:]
        self._history[:, -h:] = self._pending.T
        x = rfft(self._history*self._window, axis=1, workers=1)
        # pl/prは_powerのビュー。後続のin-place更新後も新しい統計を参照する。
        pl, pr = self._power
        a = self._alpha
        power = x.real*x.real + x.imag*x.imag
        cross = x[0]*x[1].conj()
        if self._onset_adaptation:
            # 100〜7000 Hzの帯域で、相関音の急なエネルギー増加を検出する。
            # coherenceは帯域全体の複素相関で、声の意味や種類を判定する値ではない。
            # 0.3と1.5倍は独自の検出閾値。検出から80 msは時定数を最大20 msへ短縮し、
            # 続けて検出した場合は期間を延長する。既定ではこの適応を無効にしている。
            band = self._steering_band
            energy = np.sum(power[:, band], axis=1)
            previous = float(np.sum(self._power[:, band]))
            coherence = abs(np.sum(cross[band]))/max(float(np.sqrt(energy[0])*np.sqrt(energy[1])), 1e-30)
            if coherence > .3 and float(np.sum(energy)) > 1.5*max(previous, 1e-30):
                self._attack_remaining = round(self.sample_rate*.080)
            if self._attack_remaining > 0:
                a = min(a, self._attack_alpha)
                self._attack_remaining -= h
        self._power *= a
        self._power += (1-a)*power
        self._cross *= a
        self._cross += (1-a)*cross
        # R = [[pl, r], [conj(r), pr]] はHermitianな2×2共分散。
        # 正半定値の条件 |r| <= sqrt(pl*pr) を丸め誤差でも守るため、
        # rの位相を保ったまま振幅だけを制限する。微小な分母で0除算も防ぐ。
        bound = np.sqrt(pl)*np.sqrt(pr)
        magnitude = np.abs(self._cross)
        self._cross *= np.minimum(1, bound / np.maximum(magnitude, np.float32(1e-30)))
        r = self._cross
        # 直接音1成分と左右で等電力・無相関の残響を仮定すると、
        # R = R_direct + lambda_ambient*I、R_directはrank 1になる。
        # 小さい固有値loを残響電力、差hi-loを直接音の電力と推定する。
        # Hermitian 2×2の閉形式なので、ビンごとの固有ベクトル分解は不要。
        # 複数の直接音や左右で相関した残響はこの仮定から外れ、分離誤差を生じる。
        # 方式分類とモデルの背景：He & Gan, Section II。
        # https://sigport.org/sites/default/files/HeG15_Applying%20Primary%20Ambient%20Extraction%20for%20Immersive%20Spatial%20Audio%20Reproduction-IEEE%20approved.pdf
        # 以下の縮小型Wiener式はこのモデルから導出し、レビュー中の全方式を再現しない。
        gap = np.hypot(pl-pr, 2*np.abs(r))
        lo = np.maximum(0, (pl+pr-gap)*0.5)
        hi = np.maximum((pl+pr+gap)*0.5, np.float32(1e-30))
        # 直接音のWiener推定行列はD = (R-lo*I)/hi。
        # Rの主固有方向では(hi-lo)/hi、直交方向では0という利得を持つ。
        # 残響推定a = (I-D)*xを2行に展開したものがal/ar。
        # lo=0なら主方向を除き、Rが等方的ならD=0で入力を残す。
        # 単なるL-Rではないので、中央からパンされた相関音も抑制対象となる。
        al = x[0] - ((pl-lo)*x[0]+r*x[1])/hi
        ar = x[1] - (r.conj()*x[0]+(pr-lo)*x[1])/hi
        rho = np.clip(r.real/np.maximum(bound, np.float32(1e-30)), -1, 1)
        if self.mode == "matrix":
            # ambience方式では逆相も相関音として除かれる。
            # matrix方式は逆相をサラウンドの手掛かりとして残すため、
            # rho<0の強さに応じて残響推定から元入力へ混ぜ戻す。
            # q=0でambienceと一致、rho=-1ではq=1で元の左右をそのまま使う。
            q = np.maximum(-rho, 0)**2
            al = (1-q)*al + q*x[0]
            ar = (1-q)*ar + q*x[1]
        balance = 2*bound/np.maximum(pl+pr, np.float32(1e-30))
        # センターは「同位相かつ左右のレベルが近い」帯域だけから抽出する。
        # rhoの0.6〜1のランプとbalance**4で、片側に寄った音の中央移動を抑える。
        # これらは独自の分配カーブ。center_widthは抽出量を0〜1で減らす。
        # 中央抽出を含む関連研究（ここでの閾値・分配カーブの出典ではない）：
        # Paulus & Torcoli, https://arxiv.org/abs/2206.02125
        center = (self._center_gain*(1-self.center_width)) * np.clip((rho-0.6)/0.4, 0, 1)*balance**4*(x[0]+x[1])*np.float32(1/math.sqrt(2))
        y = self._spectra
        # Cを左右からC/sqrt(2)ずつ引き、FCへ送る。
        # 全体ゲイン適用前はFL+FC/sqrt(2)=L、FR+FC/sqrt(2)=Rを保つので、
        # center_widthを変えてもフロントの再ダウンミックスは変わらない。
        y[0] = x[0] - center*np.float32(1/math.sqrt(2))
        y[1] = x[1] - center*np.float32(1/math.sqrt(2))
        y[2] = center
        # 2chに独立したLFE情報はない。Midを後段でローパスして追加するだけで、
        # フロントの低域を差し引くベースマネジメントは行わない。
        y[3] = (x[0]+x[1])*0.5
        y[4], y[5] = al, ar
        self._ola += irfft(y, n=self.fft_size, axis=1, workers=1)*self._window*0.5
        # 今後のフレームが加算する領域をOLAに残し、確定した先頭hopだけを取り出す。
        # copyが必要なのは、この直後にOLAをシフトして同じ記憶領域を再利用するため。
        block = self._ola[:, :h].T.copy()
        self._ola[:, :-h] = self._ola[:, h:]
        self._ola[:, -h:] = 0
        if self._skip:
            self._skip -= h
            return
        # IIRのzfを次hopのziへ引き継ぎ、外部ブロックの境界でフィルタを切らない。
        if self._rear_filter is not None:
            block[:, 4:], self._rear_state = sosfilt(self._rear_filter, block[:, 4:], axis=0, zi=self._rear_state)
        if self._lp is not None:
            block[:, 3], self._lp_state = sosfilt(self._lp, block[:, 3], zi=self._lp_state)
        if self._delay_size:
            # 保存したDサンプルと新しいhサンプルをつなぎ、先頭h個を出力する。
            # D<hでもD>hでも同じ式で、最後のD個が次hopへ残る。
            delayed = np.concatenate((self._delay, block[:, 4:]))
            block[:, 4:] = delayed[:h]
            self._delay[:] = delayed[h:]
        block[:, 3] *= self._lfe_gain
        block[:, 4:] *= self._rear_gain
        block *= self._output_gain
        self._queue.append(block)

    def process(self, stereo):
        """Accept floating audio (B, 2); return float32 audio (B, 6)."""
        if self._finished:
            raise RuntimeError("Decoder was flushed; call reset before processing")
        x = np.asarray(stereo)
        if x.ndim != 2 or x.shape[1] != 2 or x.dtype.kind != "f":
            raise ValueError("Expected floating audio with shape (samples, 2)")
        # 状態を書き換える前に入力全体を検査する。不正入力の拒否で統計や
        # フィルタが汚れると、次の正常なブロックまで壊れるため。
        if not np.isfinite(x).all() or (x.size and np.max(np.abs(x)) > 1e6):
            raise ValueError("Audio must be finite with absolute amplitude <= 1e6")
        count = len(x)
        for start in range(0, count, self.hop_size):
            # 前回の余りと今回の入力から、hopが満ちたときだけ_frameを呼ぶ。
            # 外部ブロックがhopの整数倍でなくても、サンプル順序は変わらない。
            part = x[start:start+self.hop_size]
            while len(part):
                take = min(len(part), self.hop_size-self._fill)
                self._pending[self._fill:self._fill+take] = part[:take]
                self._fill += take
                part = part[take:]
                if self._fill == self.hop_size:
                    self._frame()
                    self._fill = 0
        output = np.empty((count, 6), np.float32)
        # 内部生成はhop単位、外部出力は入力と同じcount個。
        # queue_offsetで取り出し途中のhopを保持し、必要な数だけ返す。
        # 先頭N個のゼロと内部skipにより、hop未満の入力でもキューは不足しない。
        pos = 0
        while pos < count:
            front = self._queue[0]
            take = min(count-pos, len(front)-self._queue_offset)
            output[pos:pos+take] = front[self._queue_offset:self._queue_offset+take]
            self._queue_offset += take
            pos += take
            if self._queue_offset == len(front):
                self._queue.popleft()
                self._queue_offset = 0
        self.input_samples += count
        if count:
            self.peak = max(self.peak, float(np.max(np.abs(output))))
            self.overrange_samples += int(np.count_nonzero(np.abs(output) > 1))
        return output

    def flush(self, tail_samples=0):
        """Drain exactly N + tail_samples samples and terminate the stream."""
        tail_samples = operator.index(tail_samples)
        if tail_samples < 0:
            raise ValueError("tail_samples must be non-negative")
        if self._finished:
            return np.empty((0, 6), np.float32)
        original_count = self.input_samples
        # ゼロをN個送って未出力の入力を排出し、追加tailでIIRやリア遅延の余韻を出す。
        # tail=0は入力長を保つための契約で、IIRの無限の尾をすべて保存する意味ではない。
        # 追加ゼロは利用者の入力サンプル数に数えず、このセッションは終了させる。
        result = self.process(np.zeros((self.latency_samples+tail_samples, 2), np.float32))
        self.input_samples = original_count
        self._finished = True
        self._queue.clear()
        return result
