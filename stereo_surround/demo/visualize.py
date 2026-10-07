"""Fixed-size waveform and streaming STFT buffers; no Qt dependency."""

import numpy as np
from scipy.fft import rfft


class VisualBuffer:
    def __init__(self, rate, fft_size=2048, hop=512, history_seconds=5):
        if not 0 < hop <= fft_size:
            raise ValueError("Invalid visualization hop")
        self.rate, self.fft_size, self.hop = rate, fft_size, hop
        self.wave = np.zeros((max(1, round(rate*.1)), 6), np.float32)
        self.wave_time = np.arange(1-len(self.wave), 1)/rate*1000
        self.rows = max(1, round(history_seconds*rate/hop))
        self.history_seconds = self.rows*hop/rate
        # 固定長リングは(time frame, frequency bin, logical channel)。
        # hop整数個に丸めた履歴秒数を軸にも使い、画面とバッファ長のずれを避ける。
        # headは次に書く位置で、満杯になった後は最も古いフレームの位置でもある。
        self.spectrum = np.full((self.rows, fft_size//2+1, 6), -100, np.float32)
        self.window = np.hanning(fft_size).astype(np.float32)
        # 表示用FFTは振幅の片側スペクトルで、電力密度のスペクトルではない。
        # DC/Nyquist付近を除くビン中心の正弦波の主ピークは、およそA*sum(window)/2。
        # 2/sum(window)でピーク振幅Aへ補正し、20*log10でdBFSへ変換する。
        # ビン間の周波数や窓の漏れでは振幅がずれるため、厳密な音量計ではない。
        # DC/Nyquistは共役側の対を持たないので、後段で2倍補正から除外する。
        # FFTの片側表現：https://numpy.org/doc/stable/reference/routines.fft.html#real-and-hermitian-transforms
        self.scale = np.float32(2/np.sum(self.window))
        self.pending = np.zeros((fft_size, 6), np.float32)
        self.fill = 0
        self.head = 0
        self.expected_frame = None
        self.frames_analyzed = 0

    def clear(self):
        self.wave.fill(0)
        self.spectrum.fill(-100)
        self.pending.fill(0)
        self.head = self.fill = self.frames_analyzed = 0
        self.expected_frame = None

    def consume(self, start_frame, audio):
        if self.expected_frame is not None and start_frame != self.expected_frame:
            # 描画キューは欠落を許容する。離れたパケットを同じFFT窓へ混ぜたり、
            # 過去を詰めて連続した履歴に見せたりしないよう、欠落時は履歴ごと消す。
            self.clear()
        self.expected_frame = start_frame+len(audio)
        count = min(len(audio), len(self.wave))
        if count:
            # 波形には直近100 msだけを保持し、長いパケットの先頭部分は表示しない。
            self.wave[:-count] = self.wave[count:]
            self.wave[-count:] = audio[-count:]
        position = 0
        # callbackの可変長パケットを固定N/hopのFFTへ組み直す。
        # 6chを同じ窓で一括処理し、描画チャンネル数が増えても窓の管理を重複させない。
        while position < len(audio):
            take = min(len(audio)-position, self.fft_size-self.fill)
            self.pending[self.fill:self.fill+take] = audio[position:position+take]
            self.fill += take
            position += take
            if self.fill == self.fft_size:
                magnitudes = np.abs(rfft(self.pending*self.window[:, None], axis=0, workers=1))*self.scale
                magnitudes[[0, -1]] *= .5
                self.spectrum[self.head] = np.maximum(-100, 20*np.log10(np.maximum(magnitudes, 1e-8)))
                self.head = (self.head+1) % self.rows
                self.frames_analyzed += 1
                # 直前のN-hop個を次の窓へ引き継ぎ、新しいhop個分だけを待つ。
                self.pending[:-self.hop] = self.pending[self.hop:]
                self.fill -= self.hop

    def image(self, channel):
        # headから末尾、先頭からheadの順へ並べると、左が古く右が新しい履歴になる。
        # ImageItemのrow-majorでは行=周波数、列=時間なので転置する。
        # 元の6chリングから切り出したビューは非連続なため、連続配列にして渡す。
        # https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/imageitem.html
        data = self.spectrum[:, :, channel]
        return np.ascontiguousarray(np.concatenate((data[self.head:], data[:self.head]), axis=0).T)
