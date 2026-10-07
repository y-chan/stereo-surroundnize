import unittest

import numpy as np

from stereo_surround import SurroundDecoder


def decode(x, block, **kwargs):
    decoder = SurroundDecoder(**kwargs)
    pieces = [decoder.process(x[i:i+block]) for i in range(0, len(x), block)]
    pieces.append(decoder.flush())
    return np.concatenate(pieces)[decoder.latency_samples:]


class DecoderTests(unittest.TestCase):
    def test_center_width_preserves_front_downmix_and_rear(self):
        voice = np.random.default_rng(87).normal(0, .1, 9001).astype(np.float32)
        x = np.column_stack((voice, voice))
        reference = decode(x, 800, output_gain=1)
        for width in (0, .35, 1):
            y = decode(x, 257, center_width=width, output_gain=1)
            np.testing.assert_allclose(y[:, 2], reference[:, 2]*(1-width), atol=2e-6, rtol=0)
            np.testing.assert_allclose(y[:, :2]+y[:, 2:3]/np.sqrt(2), x, atol=2e-5, rtol=0)
            np.testing.assert_array_equal(y[:, 3:], reference[:, 3:])
        np.testing.assert_array_equal(y[:, 2], np.zeros(len(x), np.float32))
        np.testing.assert_allclose(y[:, :2], x, atol=2e-5, rtol=0)

    def test_listening_modes_defaults_and_overrides(self):
        x = np.random.default_rng(82).normal(0, .1, (3007, 2)).astype(np.float32)
        standard = decode(x, 800)
        explicit = decode(x, 800, listening_mode="standard", center_width=0,
                          rear_delay_ms=10, rear_shelf_db=0)
        np.testing.assert_array_equal(standard, explicit)
        music = SurroundDecoder(listening_mode="music")
        self.assertEqual(music.rear_delay_ms, 0)
        self.assertEqual(music.rear_shelf_db, -3)
        bypass = decode(x, 800, listening_mode="music", rear_shelf_db=0,
                        rear_delay_ms=10)
        np.testing.assert_array_equal(standard, bypass)
        for settings in (dict(listening_mode="invalid"), dict(center_width=-.1),
                         dict(center_width=1.1), dict(center_width=np.nan),
                         dict(rear_shelf_db=1), dict(rear_shelf_db=-13),
                         dict(rear_shelf_db=np.nan), dict(rear_shelf_hz=24000),
                         dict(rear_shelf_hz=0), dict(rear_shelf_hz=np.inf)):
            with self.assertRaises(ValueError):
                SurroundDecoder(**settings)

    def test_adaptive_onset_and_chunk_independence(self):
        from stereo_surround.quality import MixtureProbe
        length = 48000
        t = np.arange(length)/48000
        v = (.2*np.sin(2*np.pi*1000*t)).astype(np.float32)
        v[:24000] = 0
        primary = np.column_stack((v, v))
        ambient = np.random.default_rng(63).normal(0, .08, (length, 2)).astype(np.float32)
        mixture = primary+ambient
        leakage = []
        for adaptive in (False, True):
            d = MixtureProbe({"primary": primary, "ambient": ambient},
                             intervals=((24000, 26400),), onset_adaptation=adaptive)
            for start in range(0, length, 800):
                d.process(mixture[start:start+800])
            powers = d.interval_powers[0][2]["primary"]
            leakage.append(powers[1]/powers[0])
        self.assertLess(leakage[1], leakage[0]*.85)
        a = decode(mixture, 800, onset_adaptation=True)
        b = decode(mixture, 257, onset_adaptation=True)
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=0)

    def test_reconstruction_and_block_independence(self):
        x = np.random.default_rng(3).normal(0, .1, (4207, 2)).astype(np.float32)
        settings = dict(center_gain=0, output_gain=1, rear_highpass=None,
                        lfe_lowpass=None, rear_delay_ms=0)
        reference = decode(x, 4207, **settings)
        np.testing.assert_allclose(reference[:, :2], x, atol=2e-5, rtol=0)
        for size in (1, 127, 255, 256, 257, 800, 1024):
            np.testing.assert_allclose(decode(x, size, **settings), reference, atol=2e-5, rtol=0)

    def test_coherent_voice_and_independent_ambience(self):
        rng = np.random.default_rng(4)
        voice = rng.normal(0, .1, 24000).astype(np.float32)
        for ratio in (1, .5, .1, 0, -1):
            x = np.column_stack((voice, ratio*voice))
            y = decode(x, 800, rear_delay_ms=0, output_gain=1, rear_gain=1, rear_highpass=None)
            leakage = np.sum(y[12000:, 4:]**2)/np.sum(x[12000:]**2)
            self.assertLess(leakage, 1e-4)
        x = rng.normal(0, .1, (48000, 2)).astype(np.float32)
        y = decode(x, 800, rear_delay_ms=0, output_gain=1, rear_gain=1, rear_highpass=None)
        ratio = np.sum(y[12000:, 4:]**2)/np.sum(x[12000:]**2)
        self.assertGreater(ratio, .6)
        self.assertLess(ratio, 1.1)

    def test_matrix_antiphase_and_downmix(self):
        t = np.arange(12000)/48000
        v = (.1*np.sin(2*np.pi*1000*t)).astype(np.float32)
        x = np.column_stack((v, -v))
        y = decode(x, 257, mode="matrix", output_gain=1, rear_gain=1,
                   rear_highpass=None, rear_delay_ms=0)
        np.testing.assert_allclose(y[:, 4:], x, atol=2e-5)
        x = np.column_stack((v, v))
        y = decode(x, 800, output_gain=1)
        np.testing.assert_allclose(y[:, :2]+y[:, 2:3]/np.sqrt(2), x, atol=2e-5)

    def test_flush_reset_and_short_files(self):
        for length in (0, 1, 127, 256, 1023):
            x = np.ones((length, 2), np.float32)*.1
            d = SurroundDecoder(center_gain=0, output_gain=1)
            first = d.process(x)
            last = d.flush(tail_samples=31)
            self.assertEqual(len(first)+len(last), length+1024+31)
            np.testing.assert_allclose(np.concatenate((first, last))[1024:1024+length, :2], x, atol=2e-5)
            self.assertEqual(len(d.flush()), 0)
            with self.assertRaises(RuntimeError):
                d.process(x)
            d.reset()
            np.testing.assert_array_equal(d.process(x), first)

    def test_invalid_input_does_not_change_state(self):
        d = SurroundDecoder()
        for x in (np.zeros((3, 1)), np.ones((3, 2), int), np.full((3, 2), np.nan)):
            with self.assertRaises(ValueError):
                d.process(x)
        self.assertEqual(d.input_samples, 0)
        self.assertEqual(d.process(np.empty((0, 2), np.float32)).shape, (0, 6))


if __name__ == "__main__":
    unittest.main()
