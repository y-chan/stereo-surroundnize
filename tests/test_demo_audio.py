import importlib.util
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from stereo_surround.demo.audio import (
    AudioFIFO, PlaybackEngine, default_mapping, file_blocks, inspect_source,
    test_tone, validate_mapping,
)
from stereo_surround.demo.visualize import VisualBuffer
from stereo_surround import SurroundDecoder


class DemoBufferTests(unittest.TestCase):
    def test_mapping_partial_callback_eof_mute_and_clipping(self):
        fifo = AudioFIFO(8, default_mapping(8))
        x = np.arange(30, dtype=np.float32).reshape(5, 6)/20
        fifo.blocks.put(x[:2])
        fifo.blocks.put(x[2:])
        fifo.blocks.put(None)
        a, b = np.empty((3, 8), np.float32), np.empty((4, 8), np.float32)
        muted = (False, True, False, False, False, False)
        self.assertFalse(fifo.render(a, 2, muted))
        self.assertTrue(fifo.render(b, 2, muted))
        result = np.concatenate((a, b))
        expected = np.zeros((7, 8), np.float32)
        for logical, slot in enumerate(default_mapping(8)):
            if not muted[logical]:
                expected[:5, slot] = np.clip(x[:, logical]*2, -1, 1)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(fifo.played_frames, 5)
        self.assertEqual(fifo.starved_frames, 0)
        self.assertGreater(fifo.clipped_samples, 0)

    def test_underflow_is_silence_and_does_not_advance_media(self):
        fifo = AudioFIFO(6, default_mapping(6))
        out = np.ones((127, 6), np.float32)
        self.assertFalse(fifo.render(out))
        self.assertFalse(out.any())
        self.assertEqual(fifo.played_frames, 0)
        self.assertEqual(fifo.starved_frames, 127)
        fifo.blocks.put(np.full((127, 6), .2, np.float32))
        fifo.render(out)
        np.testing.assert_array_equal(out, np.full_like(out, .2))

    def test_mapping_validation_and_tones(self):
        for mapping in ((0, 0, 2, 3, 4, 5), (0, 1, 2, 3, 4, 8), (0, 1)):
            with self.assertRaises(ValueError):
                validate_mapping(mapping, 8)
        for channel in range(6):
            y = np.concatenate(list(test_tone(channel, duration=.05)))
            self.assertTrue(np.isfinite(y).all())
            self.assertLessEqual(np.max(np.abs(y)), .05)
            self.assertFalse(y[:, [i for i in range(6) if i != channel]].any())
            np.testing.assert_array_equal(y[[0, -1]], np.zeros((2, 6), np.float32))

    def test_visual_spectrum_frequency_level_and_packet_gaps(self):
        v = VisualBuffer(48000)
        n = np.arange(12000)
        x = np.zeros((len(n), 6), np.float32)
        frequency = 48_000/2048*43
        x[:, 4] = .1*np.sin(2*np.pi*frequency*n/48000)
        for start in range(0, len(x), 800):
            v.consume(start, x[start:start+800])
        image = v.image(4)
        self.assertEqual(int(np.argmax(image[:, -1])), 43)
        self.assertAlmostEqual(float(image[43, -1]), -20, places=2)
        self.assertEqual(float(v.image(0).max()), -100)
        self.assertLessEqual(len(v.wave), 4800)
        v.consume(len(x)+1000, np.zeros((1, 6), np.float32))
        self.assertFalse(v.wave.any())
        self.assertEqual(v.frames_analyzed, 0)
        self.assertEqual(float(v.spectrum.max()), -100)

    def test_core_runs_with_demo_dependencies_blocked(self):
        with patch.dict('sys.modules', {name: None for name in ('PySide6', 'PyQt6', 'pyqtgraph', 'sounddevice', 'soundfile')}):
            d = SurroundDecoder()
            self.assertEqual(d.process(np.zeros((800, 2), np.float32)).shape, (800, 6))


@unittest.skipUnless(importlib.util.find_spec('soundfile'), 'Demo file IO needs soundfile')
class DemoFileTests(unittest.TestCase):
    def test_stereo_seek_and_six_channel_passthrough(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'in.wav'
            x = np.random.default_rng(72).normal(0, .1, (7001, 2)).astype(np.float32)
            sf.write(path, x, 44100, subtype='FLOAT')
            info = inspect_source(path)
            settings = dict(listening_mode='music', center_width=.35)
            y = np.concatenate(list(file_blocks(info, settings, start_frame=391, block_size=127)))
            d = SurroundDecoder(44100, **settings)
            expected = np.concatenate((d.process(x[391:]), d.flush()))[d.latency_samples:]
            np.testing.assert_allclose(y, expected, atol=2e-6, rtol=0)
            self.assertEqual(y.shape, (len(x)-391, 6))
            self.assertEqual(list(file_blocks(info, settings, start_frame=info.frames)), [])
            sf.write(path, expected, 44100, subtype='FLOAT')
            info = inspect_source(path)
            y = np.concatenate(list(file_blocks(info, {}, start_frame=91, block_size=255)))
            np.testing.assert_array_equal(y, expected[91:])

    def test_worker_device_failure_and_completion(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'in.wav'
            sf.write(path, np.full((4403, 2), .1, np.float32), 44100, subtype='FLOAT')
            info = inspect_source(path)
            backend = FakeBackend()
            engine = PlaybackEngine(info, 1, backend=backend)
            engine.start()
            self.assertTrue(engine.done.wait(5))
            engine.stop()
            self.assertIsNone(engine.error)
            self.assertEqual(engine.fifo.played_frames, 4403)
            self.assertFalse(backend.stream.thread.is_alive())
            backend.fail = True
            failed = PlaybackEngine(info, 1, backend=backend)
            failed.start()
            self.assertTrue(failed.done.wait(5))
            failed.stop()
            self.assertIn('Unsupported output format', failed.error)
            self.assertEqual(failed.state, 'error')

    def test_pause_stop_and_visual_queue_backpressure(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'in.wav'
            sf.write(path, np.zeros((96000, 2), np.float32), 48000, subtype='FLOAT')
            e = PlaybackEngine(inspect_source(path), 1, backend=FakeBackend())
            e.paused = True
            e.start()
            deadline = time.monotonic()+5
            while e.visual_drops == 0 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertGreater(e.visual_drops, 0)
            self.assertEqual(e.fifo.played_frames, 0)
            self.assertLessEqual(e.visual.qsize(), 24)
            e.stop()
            self.assertFalse(e.thread.is_alive())
            self.assertIsNone(e.error)


class FakeBackend:
    class CallbackStop(Exception):
        pass

    class CallbackAbort(Exception):
        pass

    def __init__(self):
        self.fail = False
        self.stream = None

    def query_devices(self, device):
        return dict(hostapi=0)

    def query_hostapis(self, host):
        return dict(name='test')

    def check_output_settings(self, **settings):
        if self.fail:
            raise ValueError('Unsupported output format')

    def OutputStream(self, **kwargs):
        self.stream = FakeStream(self, **kwargs)
        return self.stream


class FakeStream:
    latency = .02

    def __init__(self, backend, **kwargs):
        self.backend, self.kwargs = backend, kwargs
        self.stop = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        try:
            while not self.stop.is_set():
                for frames in (127, 511):
                    y = np.empty((frames, self.kwargs['channels']), np.float32)
                    self.kwargs['callback'](y, frames, None, SimpleNamespace(output_underflow=False))
                    time.sleep(.001)
        except (self.backend.CallbackStop, self.backend.CallbackAbort):
            pass
        finally:
            self.kwargs['finished_callback']()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
