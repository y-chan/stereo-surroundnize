from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
import soundfile as sf

from stereo_surround.files import convert_file, read_stereo, write_wav
from stereo_surround import SurroundDecoder
from stereo_surround.files import _WaveInput


class FileTests(unittest.TestCase):
    def test_block_conversion_matches_decoder_and_preserves_tail(self):
        x = np.random.default_rng(18).normal(0, .1, (5501, 2)).astype(np.float32)
        d = SurroundDecoder(48000)
        reference = np.concatenate((d.process(x), d.flush(1493)))[d.latency_samples:]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"in.wav"
            wavfile.write(source, 48000, x)
            for block in (127, 4096):
                for normalize in (False, True):
                    output = Path(tmp)/f"{block}_{normalize}.wav"
                    requested = []
                    original_read = _WaveInput.read
                    def tracked_read(reader, count):
                        requested.append(count)
                        return original_read(reader, count)
                    with patch.object(_WaveInput, "read", tracked_read):
                        result = convert_file(source, output, block_size=block,
                                              normalize=normalize, tail_samples=1493)
                    self.assertLessEqual(max(requested), block)
                    y, _ = sf.read(output, dtype="float32", always_2d=True)
                    target = reference.copy()
                    if normalize:
                        target /= np.max(np.abs(target))
                        target *= np.float32(.99)
                    np.testing.assert_allclose(y, target, atol=2e-6, rtol=0)
                    self.assertEqual(result["output_frames"], 6994)

    def test_failed_late_write_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"in.wav", Path(tmp)/"out.wav"
            x = np.zeros((6000, 2), np.float32)
            x[5000:] = 4
            wavfile.write(source, 48000, x)
            sentinel = b"existing output"
            output.write_bytes(sentinel)
            with self.assertRaises(ValueError):
                convert_file(source, output, overwrite=True, subtype="pcm16", output_gain=4)
            self.assertEqual(output.read_bytes(), sentinel)
            self.assertFalse(list(Path(tmp).glob(".surround-*.tmp")))

    def test_extensible_stereo_and_truncated_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"in.wav"
            x = np.array([[-.8, .7], [-.25, .125]], np.float32)
            for subtype in ("PCM_24", "FLOAT"):
                sf.write(source, x, 48000, format="WAVEX", subtype=subtype)
                np.testing.assert_allclose(read_stereo(source)[1], x, atol=2e-7)
            raw = source.read_bytes()
            source.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                read_stereo(source)

    def test_input_pcm_scaling_and_float(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"in.wav"
            bytes8 = np.array([[0, 255], [64, 192], [128, 128]], np.uint8)
            wavfile.write(path, 48000, bytes8)
            np.testing.assert_array_equal(read_stereo(path)[1], (bytes8.astype(np.float32)-128)/128)
            normalized = np.array([[-1, .5], [-.25, .125], [0, 0]], np.float32)
            for subtype in ("PCM_16", "PCM_24", "PCM_32", "FLOAT"):
                sf.write(path, normalized, 48000, subtype=subtype)
                np.testing.assert_allclose(read_stereo(path)[1], normalized, atol=1/32768)
            wavfile.write(path, 48000, np.zeros(40, np.int16))
            with self.assertRaises(ValueError):
                read_stereo(path)

    def test_invalid_output_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"out.wav"
            for x in (np.full((2, 6), np.nan), np.ones((2, 6), complex)):
                with self.assertRaises(ValueError):
                    write_wav(path, 48000, x)
                self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                write_wav(path, 48000, np.ones((2, 6))*2, subtype="pcm16")
            source = Path(tmp)/"tiny.wav"
            wavfile.write(source, 48000, np.full((1500, 2), 1e-39, np.float32))
            convert_file(source, path, normalize=True)
            normalized, _ = sf.read(path, dtype="float32")
            self.assertTrue(np.isfinite(normalized).all())
            self.assertAlmostEqual(float(np.max(np.abs(normalized))), .99, places=5)

    def test_extensible_roundtrip(self):
        x = np.tile(np.array([.1, -.2, .3, -.4, .5, -.6], np.float32), (37, 1))
        with tempfile.TemporaryDirectory() as tmp:
            for subtype in ("pcm16", "pcm24", "float32"):
                for layout, mask in (("side", 0x60F), ("back", 0x3F)):
                    path = Path(tmp)/f"{subtype}_{layout}.wav"
                    write_wav(path, 48000, x, subtype=subtype, layout=layout)
                    raw = path.read_bytes()
                    self.assertEqual(struct.unpack_from("<I", raw, 40)[0], mask)
                    self.assertEqual(struct.unpack_from("<I", raw, 4)[0], len(raw)-8)
                    y, rate = sf.read(path, dtype="float32", always_2d=True)
                    self.assertEqual(rate, 48000)
                    np.testing.assert_allclose(x, y, atol=1/32768)
                    with self.assertRaises(FileExistsError):
                        write_wav(path, 48000, x)

    def test_wav_flac_conversion(self):
        x = np.random.default_rng(42).integers(-15000, 15000, (1503, 2), dtype=np.int16)
        with tempfile.TemporaryDirectory() as tmp:
            wav, flac = Path(tmp)/"in.wav", Path(tmp)/"in.flac"
            wavfile.write(wav, 44100, x)
            sf.write(flac, x, 44100, subtype="PCM_16")
            np.testing.assert_array_equal(read_stereo(wav)[1], read_stereo(flac)[1])
            outputs = []
            for source in (wav, flac):
                out = Path(tmp)/(source.suffix[1:]+"out.wav")
                result = convert_file(source, out)
                self.assertEqual(result["output_frames"], len(x))
                y, rate = sf.read(out, dtype="float32", always_2d=True)
                self.assertEqual(y.shape, (len(x), 6))
                self.assertEqual(rate, 44100)
                outputs.append(y)
            np.testing.assert_array_equal(*outputs)

    def test_wav_conversion_without_soundfile_and_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"in.wav", Path(tmp)/"out.wav"
            sf.write(source, np.zeros((1501, 2), np.float32), 48000, subtype="PCM_24")
            with patch.dict("sys.modules", {"soundfile": None}):
                convert_file(source, output, block_size=129)
            y, _ = sf.read(output, always_2d=True)
            self.assertEqual(y.shape, (1501, 6))
            wavfile.write(source, 48000, np.empty((0, 2), np.float32))
            convert_file(source, output, overwrite=True, normalize=True)
            y, _ = sf.read(output, always_2d=True)
            self.assertEqual(y.shape, (0, 6))


if __name__ == "__main__":
    unittest.main()
