"""Bounded-memory file IO and speaker-labelled WAVEFORMATEXTENSIBLE output."""

from contextlib import contextmanager
import operator
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from .decoder import SurroundDecoder

_GUID_SUFFIX = b"\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


class _WaveInput:
    """Seekable RIFF PCM/IEEE float reader, including packed PCM24."""

    def __init__(self, path):
        self.file = open(path, "rb")
        try:
            header = self.file.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                raise ValueError("Expected little-endian RIFF WAV (RF64 is not supported)")
            end = struct.unpack_from("<I", header, 4)[0]+8
            if end > os.fstat(self.file.fileno()).st_size:
                raise ValueError("Truncated RIFF WAV")
            fmt, data = None, None
            while self.file.tell()+8 <= end:
                # RIFFチャンクは順序を固定しない。未知のチャンクもサイズで飛ばし、
                # fmtと最初のdataが両方見つかった時点で音声の読み込み位置を決める。
                # RIFFチャンク構造とパディング：
                # https://learn.microsoft.com/en-us/windows/win32/xaudio2/resource-interchange-file-format--riff-
                chunk, size = struct.unpack("<4sI", self.file.read(8))
                offset = self.file.tell()
                if offset+size > end:
                    raise ValueError("WAV chunk exceeds RIFF size")
                if chunk == b"fmt ":
                    fmt = self.file.read(min(size, 40))
                elif chunk == b"data" and data is None:
                    data = (offset, size)
                # チャンクは2バイト境界。奇数サイズの後のパディングはsizeに含まれない。
                self.file.seek(offset+size+(size & 1))
                if fmt is not None and data is not None:
                    break
            if fmt is None or len(fmt) < 16 or data is None:
                raise ValueError("Missing WAV format or data chunk")
            tag, channels, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", fmt)
            if tag == 0xFFFE:
                # EXTENSIBLEは外側のtagだけではPCM/floatを判別できない。
                # 拡張部のSubFormat GUIDを検査し、その先頭値を通常のtagとして使う。
                if len(fmt) < 40 or struct.unpack_from("<H", fmt, 16)[0] < 22:
                    raise ValueError("Incomplete WAVEFORMATEXTENSIBLE header")
                valid_bits = struct.unpack_from("<H", fmt, 18)[0]
                if not 0 < valid_bits <= bits or fmt[28:40] != _GUID_SUFFIX:
                    raise ValueError("Invalid WAV subtype or valid bit count")
                tag = struct.unpack_from("<I", fmt, 24)[0]
            if channels != 2:
                raise ValueError("Input must contain exactly two channels")
            if not ((tag == 1 and bits in (8, 16, 24, 32)) or
                    (tag == 3 and bits in (32, 64))):
                raise ValueError("Supported WAV inputs: PCM8/16/24/32 or IEEE float32/64")
            if rate <= 0 or align != 2*bits//8 or byte_rate != rate*align or data[1] % align:
                raise ValueError("Invalid WAV sample rate, alignment, or data size")
            self.rate, self.frames = rate, data[1]//align
            self.tag, self.bits, self.align = tag, bits, align
            self.remaining = self.frames
            self.file.seek(data[0])
        except BaseException:
            self.file.close()
            raise

    def read(self, frames):
        count = min(frames, self.remaining)
        raw = self.file.read(count*self.align)
        if len(raw) != count*self.align:
            raise ValueError("Truncated WAV audio")
        self.remaining -= count
        if self.tag == 3:
            x = np.frombuffer(raw, dtype="<f4" if self.bits == 32 else "<f8").astype(np.float32)
        elif self.bits == 8:
            x = (np.frombuffer(raw, np.uint8).astype(np.float32)-128)/128
        elif self.bits == 24:
            # NumPyに3バイト整数型はないためlittle-endianの3バイトをint32へ組む。
            # 24bitの符号ビットを反転して引く式で符号拡張し、2**23で正規化する。
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
            x = ((v ^ 0x800000)-0x800000).astype(np.float32)/8388608
        else:
            # 符号付きPCMの範囲は[-2**(bits-1), 2**(bits-1)-1]。
            # 最小値が-1となり、最大値は1よりわずかに小さい。PCM8だけは符号なし。
            x = np.frombuffer(raw, dtype="<i2" if self.bits == 16 else "<i4").astype(np.float32)
            x /= 2**(self.bits-1)
        return x.reshape(count, 2)

    def close(self):
        self.file.close()


@contextmanager
def _open_stereo(path):
    if Path(path).suffix.lower() == ".wav":
        reader = _WaveInput(path)
    else:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("FLAC and other formats require stereo-surroundnize[files]") from exc
        reader = sf.SoundFile(path)
        if reader.channels != 2:
            reader.close()
            raise ValueError("Input must contain exactly two channels")
        reader.rate = reader.samplerate
    try:
        yield reader
    finally:
        reader.close()


def _read_block(reader, frames):
    if isinstance(reader, _WaveInput):
        x = reader.read(frames)
    else:
        x = reader.read(frames, dtype="float32", always_2d=True)
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite audio")
    return x


def read_stereo(path):
    """Convenience whole-file reader. convert_file uses bounded reads instead."""
    with _open_stereo(path) as reader:
        return reader.rate, _read_block(reader, reader.frames)


class _WaveWriter:
    def __init__(self, stream, rate, frames, subtype, layout):
        if subtype not in ("float32", "pcm16", "pcm24") or layout not in ("side", "back"):
            raise ValueError("Unknown WAV subtype or layout")
        self.bits = {"float32": 32, "pcm16": 16, "pcm24": 24}[subtype]
        self.tag = 3 if subtype == "float32" else 1
        rate, frames = operator.index(rate), operator.index(frames)
        if not 0 < rate < 1000000 or frames < 0:
            raise ValueError("Invalid sample rate or frame count")
        align = 6*self.bits//8
        guid = struct.pack("<I", self.tag)+_GUID_SUFFIX
        # チャンネルマスクの昇順ビットがインターリーブの順序を決める。
        # 0x60F = FL/FR/FC/LFE/SL/SR、0x3F = FL/FR/FC/LFE/BL/BR。
        # 数値配列の最後2列はどちらでも同じだが、WAVヘッダのスピーカー位置が異なる。
        # SubFormat GUIDとdwChannelMaskの定義：
        # https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ksmedia/ns-ksmedia-waveformatextensible
        fmt = struct.pack("<HHIIHHHHI", 0xFFFE, 6, rate, rate*align, align, self.bits,
                          22, self.bits, 0x60F if layout == "side" else 0x3F)+guid
        data_size = frames*align
        riff_size = 4+8+len(fmt)+(12 if self.tag == 3 else 0)+8+data_size
        if riff_size > 0xFFFFFFFF:
            raise ValueError("Output exceeds RIFF 4 GiB limit; RF64 is not supported")
        self.stream, self.remaining = stream, frames
        stream.write(struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE"))
        stream.write(struct.pack("<4sI", b"fmt ", len(fmt))+fmt)
        if self.tag == 3:
            stream.write(struct.pack("<4sII", b"fact", 4, frames))
        stream.write(struct.pack("<4sI", b"data", data_size))

    def write(self, audio):
        x = np.asarray(audio)
        if x.ndim != 2 or x.shape[1] != 6 or x.dtype.kind not in "fi" or not np.isfinite(x).all():
            raise ValueError("Expected finite (samples, 6) audio")
        if len(x) > self.remaining:
            raise ValueError("Too many output frames")
        if self.tag != 3 and x.size and np.max(np.abs(x)) > 1:
            raise ValueError("PCM amplitude exceeds 1; reduce gain or normalize explicitly")
        if self.tag == 3:
            data = x.astype("<f4")
            if not np.isfinite(data).all():
                raise ValueError("Audio exceeds float32 range")
            raw = data.tobytes()
        else:
            scale = 2**(self.bits-1)
            # +1は符号付きPCMの正側に収まらないので、丸めた整数をscale-1で抑える。
            # それ以外の振幅超過は上で拒否し、暗黙の音量調整は行わない。
            integers = np.clip(np.rint(x*scale), -scale, scale-1)
            if self.bits == 16:
                raw = integers.astype("<i2").tobytes()
            else:
                # int32の2の補数表現からlittle-endianの下位3バイトを取り出す。
                # 負の値も同じ操作で正しいPCM24表現になる。
                raw = integers.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].copy().tobytes()
        self.stream.write(raw)
        self.remaining -= len(x)

    def finish(self):
        if self.remaining:
            raise ValueError("Incomplete output stream")


@contextmanager
def _atomic_output(path, overwrite):
    destination = Path(path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    # 完成するまでは同じディレクトリの一時ファイルへ書く。
    # 同じファイルシステム内の公開操作にすることで、途中で失敗しても既存出力を保つ。
    # rename/replace/linkのOSごとの仕様：https://docs.python.org/3/library/os.html#os.rename
    temporary = tempfile.NamedTemporaryFile(prefix=".surround-", suffix=".tmp",
                                            dir=destination.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            yield temporary
        if overwrite:
            os.replace(temporary_path, destination)
        elif os.name == "nt":
            # Windowsのrenameは既存の宛先を上書きしない。
            os.rename(temporary_path, destination)
        else:
            # POSIXのrenameは上書きするため、非上書き時はhard linkを作る。
            # 宛先の存在確認の後に別プロセスが作成しても、linkはEEXISTで失敗する。
            os.link(temporary_path, destination)
    finally:
        temporary.close()
        temporary_path.unlink(missing_ok=True)


def write_wav(path, rate, audio, *, subtype="float32", layout="side", overwrite=False):
    """Write labelled 6ch audio; failure leaves the destination unchanged."""
    x = np.asarray(audio)
    if x.ndim != 2 or x.shape[1] != 6 or x.dtype.kind not in "fi":
        raise ValueError("Expected finite (samples, 6) audio")
    with _atomic_output(path, overwrite) as stream:
        writer = _WaveWriter(stream, rate, len(x), subtype, layout)
        for start in range(0, len(x), 65536):
            writer.write(x[start:start+65536])
        if not len(x):
            writer.write(x)
        writer.finish()


def _decoded_blocks(reader, decoder, block_size, tail_samples):
    # streaming APIのN個の初期ゼロを除き、入力長+明示したtailだけを書き出す。
    # skipとemittedを全ブロックにまたがって管理し、短い入力にも同じ契約を適用する。
    skip = decoder.latency_samples
    total = reader.frames+tail_samples
    emitted = 0

    def trim(y):
        nonlocal skip, emitted
        drop = min(skip, len(y))
        skip -= drop
        result = y[drop:drop+total-emitted]
        emitted += len(result)
        return result

    while True:
        x = _read_block(reader, block_size)
        if not len(x):
            break
        y = trim(decoder.process(x))
        if len(y):
            yield y
    remaining = decoder.latency_samples+tail_samples
    # flush()でN+tailを一括確保する代わりに、小さいゼロブロックで内部を排出する。
    # 長いtailを指定しても、この変換経路のメモリ量はblock_sizeで制限される。
    while remaining:
        count = min(block_size, remaining)
        y = trim(decoder.process(np.zeros((count, 2), np.float32)))
        remaining -= count
        if len(y):
            yield y
    if emitted != total:
        raise ValueError("Input frame count changed or input was truncated")


def convert_file(source, destination, *, subtype="float32", layout="side",
                 overwrite=False, normalize=False, tail_samples=0, block_size=4096, **settings):
    """Bounded-memory conversion. Normalization spools decoded audio to disk."""
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("Input and output must be different files")
    tail_samples, block_size = operator.index(tail_samples), operator.index(block_size)
    if tail_samples < 0 or block_size <= 0:
        raise ValueError("tail_samples must be non-negative and block_size positive")
    peak = 0.0
    with _open_stereo(source) as reader, _atomic_output(destination, overwrite) as stream:
        decoder = SurroundDecoder(reader.rate, **settings)
        frames = reader.frames+tail_samples
        writer = _WaveWriter(stream, reader.rate, frames, subtype, layout)
        blocks = _decoded_blocks(reader, decoder, block_size, tail_samples)
        if normalize:
            # 全体に一定ゲインを掛けるには、最後まで読んだピークが必要になる。
            # DSPは1回だけ実行してfloat32出力を一時ファイルへ保存し、2回目はIOと
            # ゲイン適用だけを行う。ブロックごとの正規化による音量変動を避ける。
            with tempfile.TemporaryFile() as spool:
                for y in blocks:
                    peak = max(peak, float(np.max(np.abs(y))))
                    spool.write(y.astype("<f4").tobytes())
                spool.seek(0)
                while raw := spool.read(block_size*6*4):
                    y = np.frombuffer(raw, dtype="<f4").reshape(-1, 6).copy()
                    if peak:
                        y /= peak
                        y *= np.float32(.99)
                    writer.write(y)
                if peak:
                    peak = float(np.float32(.99))
        else:
            for y in blocks:
                peak = max(peak, float(np.max(np.abs(y))))
                writer.write(y)
        writer.finish()
        return {"sample_rate": reader.rate, "input_frames": reader.frames, "output_frames": frames,
                "channels": 6, "latency_samples": decoder.latency_samples, "peak": peak}
