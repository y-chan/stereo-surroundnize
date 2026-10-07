import unittest

import numpy as np
from scipy.signal import sosfreqz, sosfilt, sos2zpk

from stereo_surround import SurroundDecoder


class FilterTests(unittest.TestCase):
    def test_music_shelf_response(self):
        for rate in (8000, 44100, 48000, 96000):
            d = SurroundDecoder(rate, listening_mode="music")
            _, response = sosfreqz(d._shelf, worN=[0, 3000, rate/2], fs=rate)
            np.testing.assert_allclose(20*np.log10(np.abs(response)), [0, -1.5, -3], atol=2e-5)
            _, poles, _ = sos2zpk(d._shelf)
            self.assertTrue(np.all(np.abs(poles) < 1))
            self.assertIsNone(SurroundDecoder(rate, rear_shelf_db=0)._shelf)

    def test_music_filter_state_and_alignment(self):
        v = np.random.default_rng(21).normal(0, .1, 6007).astype(np.float32)
        x = np.column_stack((v, -v))
        settings = dict(mode="matrix", listening_mode="music", output_gain=1,
                        rear_gain=1, rear_highpass=None)
        d = SurroundDecoder(**settings)
        expected = sosfilt(d._shelf, x, axis=0)
        reference = None
        for block in (127, 800, 1024):
            d = SurroundDecoder(**settings)
            parts = [d.process(x[i:i+block]) for i in range(0, len(x), block)]
            y = np.concatenate(parts+[d.flush()])[d.latency_samples:]
            np.testing.assert_allclose(y[:, 4:], expected, atol=2e-6, rtol=0)
            np.testing.assert_allclose(y[:, :2], x, atol=2e-5, rtol=0)
            if reference is None:
                reference = y
            np.testing.assert_array_equal(y, reference)
            d.reset()
            z = np.concatenate((d.process(x), d.flush()))[d.latency_samples:]
            np.testing.assert_array_equal(z, reference)

    def test_low_frequency_rejection_independent_of_separation(self):
        decoder = SurroundDecoder()
        _, hp = sosfreqz(decoder._hp, worN=[80, 1000], fs=48000)
        self.assertLess(20*np.log10(abs(hp[0]/hp[1])), -35)
        _, lp = sosfreqz(decoder._lp, worN=[60, 1000], fs=48000)
        self.assertGreater(20*np.log10(abs(lp[0])), -1)
        self.assertLess(20*np.log10(abs(lp[1])), -60)

    def test_antiphase_bass_is_filtered_in_matrix_mode(self):
        n = np.arange(48000)
        tone = (0.1*np.sin(2*np.pi*80*n/48000)).astype(np.float32)
        x = np.column_stack((tone, -tone))
        d = SurroundDecoder(mode="matrix", rear_gain=1, output_gain=1, rear_delay_ms=0)
        y = np.concatenate((d.process(x), d.flush()))[d.latency_samples:]
        ratio = np.sum(y[12000:, 4:]**2)/np.sum(x[12000:]**2)
        self.assertLess(10*np.log10(ratio), -35)
        np.testing.assert_allclose(y[:, :2], x, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
