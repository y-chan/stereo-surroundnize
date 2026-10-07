import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

AVAILABLE = all(importlib.util.find_spec(name) for name in ('PySide6', 'pyqtgraph', 'soundfile'))


@unittest.skipUnless(AVAILABLE, 'Install the demo extra to run Qt tests')
class DemoGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_play_pause_seek_resume_stop(self):
        import soundfile as sf
        from stereo_surround.demo.audio import PlaybackEngine
        from stereo_surround.demo.window import DemoWindow
        from test_demo_audio import FakeBackend
        devices = [dict(id=1, name='Example multichannel output', api='Windows DirectSound', channels=8, rate=48000)]
        backend = FakeBackend()
        def factory(*args, **kwargs):
            return PlaybackEngine(*args, backend=backend, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'in.wav'
            sf.write(path, np.zeros((480000, 2), np.float32), 48000)
            with patch('stereo_surround.demo.window.output_devices', return_value=devices):
                window = DemoWindow(path, engine_factory=factory)
            try:
                window.play_button.click()
                old = window.engine
                deadline = time.monotonic()+3
                while old.state == 'loading' and time.monotonic()<deadline:
                    self.app.processEvents()
                    time.sleep(.005)
                self.assertEqual(old.state, 'playing')
                self.assertFalse(window.device_combo.isEnabled())
                window.pause_button.click()
                self.assertTrue(old.paused)
                time.sleep(.02)
                position = old.fifo.played_frames
                time.sleep(.02)
                self.assertEqual(old.fifo.played_frames, position)
                window.seek_slider.setValue(5000)
                window.seek()
                self.assertIsNot(window.engine, old)
                self.assertFalse(old.thread.is_alive())
                self.assertEqual(window.engine.start_frame, 240000)
                self.assertTrue(window.engine.paused)
                window.pause_button.click()
                self.assertFalse(window.engine.paused)
                window.stop_button.click()
                self.assertIsNone(window.engine)
                self.assertTrue(window.device_combo.isEnabled())
                self.assertFalse(window.pause_button.isEnabled())
            finally:
                window.close()

    def test_file_controls_mapping_and_waveform_spectrogram_render(self):
        import soundfile as sf
        from stereo_surround.demo.window import DemoWindow
        from stereo_surround.demo.visualize import VisualBuffer
        devices = [
            dict(id=0, name='Example stereo output', api='Windows WASAPI', channels=2, rate=44100),
            dict(id=1, name='Example multichannel output', api='Windows DirectSound', channels=8, rate=44100),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'in.wav'
            sf.write(path, np.zeros((4410, 2), np.float32), 44100)
            with patch('stereo_surround.demo.window.output_devices', return_value=devices):
                window = DemoWindow(path)
            try:
                window.show()
                self.app.processEvents()
                self.assertEqual(window.device_combo.currentData()['id'], 1)
                self.assertEqual(tuple(s.value() for s in window.map_spins), (1, 2, 3, 4, 7, 8))
                window.channels_combo.setCurrentIndex(0)
                self.assertEqual(tuple(s.value() for s in window.map_spins), (1, 2, 3, 4, 5, 6))
                self.assertTrue(window.play_button.isEnabled())
                self.assertFalse(window.pause_button.isEnabled())
                self.assertEqual(window.decoder_settings()['listening_mode'], 'music')
                t = np.arange(4410)/44100
                bins = np.arange(1, 7)*40
                frequencies = bins*44100/2048
                x = .1*np.sin(2*np.pi*t[:, None]*frequencies[None, :])
                window.visual = VisualBuffer(44100)
                window.visual.consume(0, x.astype(np.float32))
                window.draw()
                self.app.processEvents()
                self.assertEqual(len(window.wave_curves[0].getData()[0]), 4410)
                self.assertEqual(len(window.images), 6)
                for i, image in enumerate(window.images):
                    self.assertEqual(image.image.shape[0], 1025)
                    self.assertTrue(np.isfinite(image.image).all())
                    self.assertEqual(np.argmax(image.image[:, -1]), bins[i])
                    np.testing.assert_allclose(window.wave_curves[i].getData()[1], x[:, i], atol=1e-8)
                window.spec_plots[0].setYRange(0, 3000, padding=0)
                window.draw()
                self.app.processEvents()
                for spec in window.spec_plots:
                    np.testing.assert_allclose(spec.viewRange()[1], [0, 3000], atol=1)
                self.assertFalse(window.grab().isNull())
                window.device_combo.setCurrentIndex(0)
                self.assertIn('2ch', window.device_hint.text())
                self.assertTrue(window.exclusive.isEnabled())
            finally:
                window.close()


if __name__ == '__main__':
    unittest.main()
