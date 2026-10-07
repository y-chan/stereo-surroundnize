import unittest
from types import SimpleNamespace

import numpy as np

from stereo_surround.touchdesigner import TouchDesignerAdapter


class Channel:
    def __init__(self, name, owner):
        self._name = name
        self.owner = owner

    @property
    def name(self):
        # td.Channel.name is read-only; allow names only at channel creation.
        return self._name

    def copyNumpyArray(self, x):
        assert x.shape == (self.owner.numSamples,)
        assert x.dtype == np.float32 and x.flags.c_contiguous
        self.x = x.copy()


class Destination:
    def __init__(self):
        self.channels = []
        self._num_samples = 0
        self.isTimeSlice = True

    @property
    def numSamples(self):
        return self._num_samples

    @numSamples.setter
    def numSamples(self, value):
        if self.isTimeSlice:
            raise RuntimeError("Editing numSamples is not supported in Time Slice mode")
        self._num_samples = value

    def copyNumpyArray(self, x):
        assert x.ndim == 2 and x.dtype == np.float32 and x.flags.c_contiguous
        # Array transfer alone need not preserve audio timing metadata.
        self.rate = 60
        self.start = 0
        self.isTimeSlice = False
        self._num_samples = x.shape[1]
        self.clear()
        for i, samples in enumerate(x):
            self.appendChan(f"chan{i+1}").copyNumpyArray(samples)

    def clear(self):
        self.channels.clear()

    def appendChan(self, name):
        channel = Channel(name, self)
        self.channels.append(channel)
        return channel

    @property
    def x(self):
        return np.stack([channel.x for channel in self.channels])

    def __getitem__(self, index):
        return self.channels[index]


def source(start, count, rate=48000):
    x = np.ones((2, count), np.float32)*.1
    return SimpleNamespace(start=start, numSamples=count, numChans=2, rate=rate,
                           isTimeSlice=True, numpyArray=lambda: x)


class AdapterTests(unittest.TestCase):
    def test_variable_slices_recook_seek_rate_change(self):
        adapter, target = TouchDesignerAdapter(), Destination()
        first = adapter.cook(source(0, 800), target).copy()
        adapter.cook(source(0, 800), target)
        self.assertEqual(adapter.decoder.input_samples, 800)
        np.testing.assert_array_equal(target.x, first)
        adapter.cook(source(800, 1600), target)
        self.assertEqual(adapter.decoder.input_samples, 2400)
        self.assertEqual(target.x.shape, (6, 1600))
        self.assertEqual(target.rate, 48000)
        self.assertEqual(target.start, 800)
        self.assertTrue(target.isTimeSlice)
        np.testing.assert_array_equal(target.x, adapter._cached)
        adapter.cook(source(3000, 800), target)
        self.assertEqual(adapter.decoder.input_samples, 800)
        np.testing.assert_array_equal(target.x, first)
        adapter.cook(source(0, 735, 44100), target)
        self.assertEqual(adapter.decoder.sample_rate, 44100)
        self.assertEqual(target.rate, 44100)
        self.assertEqual([c.name for c in target.channels], ["FL", "FR", "FC", "LFE", "SL", "SR"])

    def test_continuous_audio_becomes_nonzero_after_latency(self):
        adapter, target = TouchDesignerAdapter(), Destination()
        samples = np.arange(4800)
        mono = (.1*np.sin(2*np.pi*440*samples/48000)).astype(np.float32)
        outputs = []
        for start in range(0, len(mono), 800):
            block = np.stack([mono[start:start+800]]*2)
            src = source(start, 800)
            src.numpyArray = lambda: block
            adapter.cook(src, target)
            outputs.append(target.x)
            self.assertEqual(target.rate, 48000)
            self.assertEqual(target.numSamples, 800)
            self.assertEqual(target.start, start)
            self.assertTrue(target.isTimeSlice)
        audio = np.concatenate(outputs, axis=1)
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(np.max(np.abs(audio[:, 1600:])), .01)

    def test_variable_slices_preserve_tone_pitch_and_sample_count(self):
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                adapter, target = TouchDesignerAdapter(), Destination()
                mono = (.1*np.sin(2*np.pi*440*np.arange(3*rate)/rate)).astype(np.float32)
                # Include skipped frames and rounding around a nominal 60 FPS slice.
                sizes = (rate//60, 2*(rate//60), rate//60-1, rate//60+1)
                start, index, outputs = 0, 0, []
                while start < len(mono):
                    count = min(sizes[index % len(sizes)], len(mono)-start)
                    block = np.stack([mono[start:start+count]]*2)
                    src = source(start, count, rate)
                    src.numpyArray = lambda: block
                    adapter.cook(src, target)
                    self.assertEqual(target.rate, rate)
                    self.assertEqual(target.start, start)
                    self.assertEqual(target.numSamples, count)
                    outputs.append(target.x)
                    start += count
                    index += 1
                audio = np.concatenate(outputs, axis=1)
                self.assertEqual(audio.shape, (6, len(mono)))
                # The coherent mono tone is routed to FC. Measure a settled one-second span.
                center = audio[2, rate:2*rate]
                spectrum = np.abs(np.fft.rfft(center*np.hanning(rate)))
                peak_hz = np.fft.rfftfreq(rate, 1/rate)[np.argmax(spectrum)]
                self.assertAlmostEqual(peak_hz, 440, delta=1)


if __name__ == "__main__":
    unittest.main()
