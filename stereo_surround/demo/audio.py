"""Bounded producer/consumer playback, independent of Qt.

Only device discovery/playback imports sounddevice; only file IO imports
soundfile. File reads and upmix run on a producer thread, never the callback.
"""

from dataclasses import dataclass
from pathlib import Path
import queue
import threading

import numpy as np

from ..decoder import CHANNELS, SurroundDecoder


@dataclass(frozen=True)
class SourceInfo:
    path: str
    rate: int
    frames: int
    channels: int

    @property
    def duration(self):
        return self.frames / self.rate


def inspect_source(path):
    import soundfile as sf
    info = sf.info(path)
    if info.channels not in (2, 6):
        raise ValueError("2ch stereo or 6ch surround files are supported")
    if not 8000 <= info.samplerate <= 192000:
        raise ValueError("Supported sample rates: 8000 to 192000 Hz")
    return SourceInfo(str(Path(path).resolve()), info.samplerate, info.frames, info.channels)


def file_blocks(info, settings, start_frame=0, block_size=1024):
    """Yield exactly the remaining source length, removing DSP startup delay."""
    import soundfile as sf
    if not 0 <= start_frame <= info.frames:
        raise ValueError("Seek position is outside the file")
    with sf.SoundFile(info.path) as source:
        if (source.samplerate, source.frames, source.channels) != (info.rate, info.frames, info.channels):
            raise ValueError("Source file changed; open it again")
        source.seek(start_frame)
        # シークは新しいデコーダから開始し、直前の場所の窓・統計・IIR状態を使わない。
        # 履歴を先読みして復元する方式ではないため、シーク直後は統計の立ち上がりがある。
        remaining = info.frames-start_frame
        decoder = SurroundDecoder(info.rate, **settings) if info.channels == 2 else None
        skip = decoder.latency_samples if decoder else 0
        # DSPが返す先頭N個の遅延用ゼロを、複数ブロックにまたがって取り除く。
        # ファイルでは先読みできるので、開始位置から実際の音源長だけを再生する。
        # ここで除くのはSTFTの固定遅延だけで、リアの追加遅延やIIRはそのまま。
        while True:
            x = source.read(block_size, dtype="float32", always_2d=True)
            if not len(x):
                break
            if not np.isfinite(x).all():
                raise ValueError("Source contains non-finite samples")
            y = decoder.process(x) if decoder else x
            drop = min(skip, len(y))
            skip -= drop
            y = y[drop:drop+remaining]
            remaining -= len(y)
            if len(y):
                yield y
        if decoder:
            # 最後の入力も内部にN個分残るためflushが必要。
            # Nより短い音源ではskipが残るので、flush結果にも同じ長さの契約を適用する。
            y = decoder.flush()[skip:skip+remaining]
            remaining -= len(y)
            if len(y):
                yield y
        if remaining:
            raise ValueError("Source ended before its reported frame count")


def test_tone(channel, rate=48000, duration=1.0, block_size=1024):
    """One quiet, faded channel at a time; caller applies the master volume."""
    if channel not in range(6):
        raise ValueError("Unknown speaker")
    count = round(rate*duration)
    frequency = 60 if channel == 3 else 440
    fade = max(1, round(rate*.03))
    for start in range(0, count, block_size):
        n = np.arange(start, min(start+block_size, count))
        envelope = np.minimum(1, np.minimum(n, count-1-n)/fade)
        y = np.zeros((len(n), 6), np.float32)
        y[:, channel] = .05*envelope*np.sin(2*np.pi*frequency*n/rate)
        yield y


def default_mapping(channels):
    if channels == 2:
        return (0, 1, 0, 1, 0, 1)
    if channels == 8:
        # Windows order: FL FR FC LFE BL BR SL SR. Leave BL/BR silent.
        return (0, 1, 2, 3, 6, 7)
    if channels == 6:
        return tuple(range(6))
    raise ValueError("Choose 6 or 8 output channels")


def validate_mapping(mapping, channels):
    if channels == 2:
        if len(mapping) != 6 or any(not isinstance(v, (int, np.integer)) or not 0 <= v < 2 for v in mapping):
            raise ValueError("Stereo output requires six logical speaker routes")
        return tuple(int(v) for v in mapping)
    if channels not in (6, 8) or len(mapping) != 6:
        raise ValueError("Six logical speakers require a 6ch or 8ch output")
    if any(not isinstance(v, (int, np.integer)) or not 0 <= v < channels for v in mapping):
        raise ValueError("Output mapping is outside the device channel range")
    if len(set(mapping)) != 6:
        raise ValueError("Each speaker must use a different output channel")
    return tuple(int(v) for v in mapping)


def output_devices(backend=None):
    if backend is None:
        import sounddevice as backend
    hosts = backend.query_hostapis()
    return [dict(id=i, name=d["name"], api=hosts[d["hostapi"]]["name"],
                 channels=d["max_output_channels"], rate=d["default_samplerate"])
            for i, d in enumerate(backend.query_devices()) if d["max_output_channels"] > 0]


class AudioFIFO:
    """Single audio consumer; producer puts arrays followed by None (EOF)."""

    def __init__(self, channels, mapping, capacity=24):
        self.channels = channels
        self.mapping = validate_mapping(mapping, channels)
        self.blocks = queue.Queue(maxsize=capacity)
        # producerが追加し、callbackだけがcurrent/offsetと再生カウンタを更新する。
        # デバイスの要求サイズと読み込みブロックのサイズは一致する必要がない。
        # 上限はブロック数であり、実際のバッファ時間は各ブロック長とrateで決まる。
        self.current = None
        self.offset = 0
        self.played_frames = 0
        self.starved_frames = 0
        self.clipped_samples = 0
        self.eof = False
        self.visual_output = np.empty((0, 6), np.float32)

    def render(self, output, gain=1.0, muted=(False,)*6):
        """Fill every output sample, including unused slots and underruns."""
        # PortAudioのoutputは未初期化の可能性がある。
        # 未使用の8chスロット、ミュート、EOF後の余り、キュー不足を先に無音にする。
        output.fill(0)
        self.visual_output = np.zeros((len(output), 6), np.float32)
        position = 0
        while position < len(output) and not self.eof:
            if self.current is None:
                try:
                    self.current = self.blocks.get_nowait()
                except queue.Empty:
                    # callbackはproducerを待たない。不足分のゼロをデバイスへ送るが、
                    # played_framesは増やさず、音源位置を勝手に先へ進めない。
                    self.starved_frames += len(output)-position
                    break
                self.offset = 0
                if self.current is None:
                    # Noneはキューが一時的に空なのではなく、producerが送るEOFの印。
                    # Queue.Emptyと区別しないと、処理待ちを曲の終了と誤認してしまう。
                    self.eof = True
                    break
            take = min(len(output)-position, len(self.current)-self.offset)
            source = self.current[self.offset:self.offset+take]
            target = output[position:position+take]
            visual = self.visual_output[position:position+take]
            for channel in range(6):
                if not muted[channel]:
                    np.multiply(source[:, channel], gain, out=visual[:, channel])
            if self.channels == 2:
                # Center and LFE go to both sides; surrounds go to their matching side.
                target[:, 0] = visual[:, 0] + visual[:, 2] * .7071 + visual[:, 3] + visual[:, 4] * .7071
                target[:, 1] = visual[:, 1] + visual[:, 2] * .7071 + visual[:, 3] + visual[:, 5] * .7071
            else:
                for channel, slot in enumerate(self.mapping):
                    target[:, slot] = visual[:, channel]
            self.offset += take
            position += take
            self.played_frames += take
            if self.offset == len(self.current):
                self.current = None
        self.clipped_samples += int(np.count_nonzero(np.abs(output) > 1))
        np.clip(output, -1, 1, out=output)
        return self.eof


class PlaybackEngine:
    """One session per file/seek/test; GUI polls state instead of receiving audio signals."""

    def __init__(self, info, device, *, output_channels=8, mapping=None,
                 settings=None, start_frame=0, exclusive=False, tone_channel=None,
                 backend=None):
        self.info, self.device = info, device
        self.start_frame = int(start_frame)
        self.settings = dict(settings or {})
        self.exclusive, self.tone_channel = exclusive, tone_channel
        self.backend = backend
        self.fifo = AudioFIFO(output_channels, default_mapping(output_channels) if mapping is None else mapping)
        self.visual = queue.Queue(maxsize=24)
        # stop_eventはGUIからの停止要求、doneはデバイスの終了通知。
        # 音源を読み終えた時点では音声キューとデバイスの出力が残っているため、
        # producerのEOFとデバイスの完了は別々に管理する。
        self.stop_event, self.done = threading.Event(), threading.Event()
        self.thread = None
        self.paused = False
        self.gain = 10**(-12/20)
        self.muted = (False,)*6
        self.error = None
        self.state = "idle"
        self.portaudio_underflows = 0
        self.visual_drops = 0
        self.visual_frames = 0
        self.latency = 0.0

    @property
    def position(self):
        # デバイスへ渡した音源フレーム数による位置。
        # 出力バッファの遅延やアンプでの処理を差し引いた聴取時刻ではない。
        return min(self.info.duration, (self.start_frame+self.fifo.played_frames)/self.info.rate)

    def start(self):
        if self.thread is not None:
            raise RuntimeError("This playback session has already started")
        self.state = "loading"
        self.thread = threading.Thread(target=self._run, name="surround-producer", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            # 古いセッションがデバイスを解放してから次の再生・シークを開始する。
            # GUIを永久に待たせず、閉じられないデバイスは呼び出し元へ報告する。
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError("Audio device is still closing; try again shortly")

    def _put(self, block):
        # 待ってよいのはproducerだけ。短いtimeoutで停止要求を確認できるようにする。
        # callbackの異常終了時はdoneでも抜けないと、満杯のキューで永久に待つ。
        # put(timeout)とget_nowaitの仕様：https://docs.python.org/3/library/queue.html
        while not self.stop_event.is_set() and not self.done.is_set():
            try:
                self.fifo.blocks.put(block, timeout=.02)
                return True
            except queue.Full:
                pass
        return False

    def _callback(self, output, frames, timing, status):
        # 音声callbackではファイルIO、DSPのFFT、Qt操作、キュー待ちを行わない。
        # 配列コピーやPythonのキュー操作は残るため、期限保証のある実装ではない。
        # callbackの制約と終了動作：
        # https://python-sounddevice.readthedocs.io/en/latest/api/streams.html
        output.fill(0)
        if self.stop_event.is_set():
            raise self.backend.CallbackAbort
        try:
            if status.output_underflow:
                self.portaudio_underflows += 1
            finished = False
            if not self.paused:
                finished = self.fifo.render(output, self.gain, self.muted)
            # pause中はゼロを送り続け、FIFOを消費しないので音源位置は保持される。
            # 描画用配列はrenderごとに作り、出力チャンネル数から独立させる。
            visual = self.fifo.visual_output if not self.paused else np.zeros((frames, 6), np.float32)
            packet = (self.visual_frames, visual)
            # この番号はデバイスへ送ったフレームの連番で、pause中も増やす。
            # 音源位置とは別で、描画パケットの欠落と時間の連続性を検出するために使う。
            self.visual_frames += frames
            try:
                self.visual.put_nowait(packet)
            except queue.Full:
                # 描画キューだけは新しいパケットを捨てる。音声を待たせないことを優先。
                self.visual_drops += 1
            if finished:
                # 最後のバッファを送出して終了するCallbackStopを使う。
                # 即座に打ち切るCallbackAbortでは終端の音が失われる可能性がある。
                raise self.backend.CallbackStop
        except (self.backend.CallbackStop, self.backend.CallbackAbort):
            raise
        except Exception as exc:
            output.fill(0)
            self.error = str(exc)
            raise self.backend.CallbackAbort

    def _run(self):
        iterator = None
        stream = None
        try:
            if self.backend is None:
                import sounddevice
                self.backend = sounddevice
            device = self.backend.query_devices(self.device)
            host = self.backend.query_hostapis(device["hostapi"])["name"]
            extra = None
            if "WASAPI" in host:
                # 必要なch数やrateを黙って変換せず、要求形式の非対応を報告する。
                extra = self.backend.WasapiSettings(exclusive=self.exclusive, auto_convert=False)
            elif self.exclusive:
                raise ValueError("Exclusive mode is only available for a WASAPI device")
            self.backend.check_output_settings(device=self.device, channels=self.fifo.channels,
                                               samplerate=self.info.rate, dtype="float32", extra_settings=extra)
            iterator = (test_tone(self.tone_channel, self.info.rate) if self.tone_channel is not None
                        else file_blocks(self.info, self.settings, self.start_frame))
            iterator = iter(iterator)
            exhausted = False
            # 最初のcallbackより先に最大4ブロックを用意し、読み込み開始直後の不足を防ぐ。
            # 短いファイルは4ブロック未満でEOFを送り、同じ消費経路で終える。
            for _ in range(4):
                try:
                    if not self._put(next(iterator)):
                        return
                except StopIteration:
                    exhausted = True
                    self._put(None)
                    break
            if self.stop_event.is_set():
                return
            stream = self.backend.OutputStream(
                # blocksize=0でデバイスが要求サイズを選ぶ。FIFOは可変長要求に対応する。
                # latency="high"は安定性のための指定で、アルゴリズムの遅延とは別。
                device=self.device, samplerate=self.info.rate, channels=self.fifo.channels,
                dtype="float32", blocksize=0, latency="high", extra_settings=extra,
                callback=self._callback, finished_callback=self.done.set)
            self.latency = float(stream.latency)
            self.state = "playing"
            stream.start()
            if not exhausted:
                for block in iterator:
                    if not self._put(block):
                        break
                if not self.stop_event.is_set():
                    self._put(None)
            # EOFをキューへ入れた後も、callbackが残りを送り、デバイスの
            # finished_callbackがdoneを立てるまではストリームを閉じない。
            while not self.stop_event.is_set() and not self.done.wait(.02):
                pass
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                iterator.close()
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    self.error = self.error or str(exc)
            self.state = "error" if self.error else ("stopped" if self.stop_event.is_set() else "finished")
            self.done.set()
