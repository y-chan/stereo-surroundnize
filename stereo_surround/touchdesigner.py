"""Small host adapter; importing this module does not require TouchDesigner."""

import numpy as np

from .decoder import CHANNELS, SurroundDecoder


class TouchDesignerAdapter:
    """One adapter per Script CHOP; reset explicitly after source/settings changes."""

    def __init__(self, **settings):
        self.settings = dict(settings)
        self.reset()

    def reset(self):
        self.decoder = None
        self._last_key = None
        self._last_end = None
        self._cached = None

    def cook(self, source, destination):
        if source.numChans != 2 or not source.isTimeSlice:
            raise ValueError("Use a two-channel, time-sliced input CHOP")
        rate = float(source.rate)
        start = float(source.start)
        # CHOPは(channels, samples)、DSPは(samples, channels)という逆の軸順。
        # 入力取得の仕様：https://derivative.ca/UserGuide/CHOP_Class#Methods
        x = source.numpyArray()
        if x.shape != (2, source.numSamples):
            raise ValueError("CHOP channel data has unexpected dimensions")
        key = (start, len(x[0]), rate)
        if key == self._last_key:
            # 同じ時間スライスの再cookでDSPを2回進めると、音とIIR状態がずれる。
            # 同じstart/長さ/rateでは内容も同じという前提で直前の結果を使う。
            # 内容や設定だけを差し替えたときは、呼び出し側でResetする必要がある。
            y = self._cached
        else:
            if self.decoder is None or self.decoder.sample_rate != rate:
                self.decoder = SurroundDecoder(rate, **self.settings)
                self._last_end = None
            elif self._last_end is not None and abs(start-self._last_end) > 0.01:
                # 連続でないstartはシークやスライス欠落として扱い、以前の履歴を消す。
                # 0.01はサンプル位置の丸め誤差の許容値で、10 msの時間幅ではない。
                # スライスが切り詰められる条件：https://derivative.ca/UserGuide/Time_Slicing
                self.decoder.reset()
            # 各Channel.copyNumpyArrayに渡す行が連続float32になるよう転置後にコピー。
            # https://derivative.ca/UserGuide/Channel_Class#Methods
            y = np.ascontiguousarray(self.decoder.process(x.T).T, dtype=np.float32)
            self._last_key = key
            self._last_end = start+source.numSamples
            self._cached = y
        # Channel.nameは読み取り専用なので、appendChanで名前付きチャンネルを作る。
        # CHOP全体のcopyNumpyArrayはchan1等の名前を作るため、各Channelへ一括転送。
        # Pythonのサンプル単位ループは使わず、6本の連続配列をそれぞれコピーする。
        # https://derivative.ca/UserGuide/ScriptCHOP_Class#Methods
        # https://derivative.ca/UserGuide/Channel_Class#Members
        # Time SliceモードではnumSamplesへの代入が禁止される。
        # 配列転送で長さを設定し、長さを保持するclear()で仮のチャンネルだけ消す。
        # この最初の転送は名前付きチャンネルへの転送に加えて1回分のコピーを伴う。
        destination.copyNumpyArray(y)
        destination.clear()
        for name, samples in zip(CHANNELS, y):
            destination.appendChan(name).copyNumpyArray(samples)
        # 配列転送・チャンネル生成を終えてから音声の時間軸を確定する。
        # 配列自体にはrate/startの情報がないため、ホストの既定値に任せない。
        # rateを先に、次にサンプル単位のstartを設定する（CHOP APIの指定順）。
        # https://derivative.ca/UserGuide/CHOP_Class#Members
        destination.rate = rate
        # 出力は現在のスライスのまま、値だけがDSPの固定遅延を持つ。
        destination.start = start
        destination.isTimeSlice = True
        return y
