# Third-party dependencies and references

Original project code, tests, examples, and documentation are licensed under
the MIT license in [LICENSE](LICENSE). This does not relicense dependencies,
external publications, trademarks, or input/output audio recordings.
The project wheel and source distribution do not vendor these dependencies.

## Runtime dependencies

| Component | License / upstream information | Use |
| --- | --- | --- |
| NumPy | [BSD-3-Clause](https://numpy.org/doc/stable/license.html) | Array operations. |
| SciPy | [BSD-3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | FFT and filters. |
| PySide6-Essentials / Shiboken6 | [LGPLv3 / GPLv3 / commercial](https://doc.qt.io/qtforpython-6/) | Optional demo; the project uses the LGPL option for its bindings. |
| Qt Core, GUI, Widgets | [Qt licensing](https://doc.qt.io/qt-6/licensing.html) | Optional demo; these modules offer LGPLv3. |
| PyQtGraph | [MIT](https://github.com/pyqtgraph/pyqtgraph/blob/master/LICENSE.txt) | Optional demo plots. |
| sounddevice | [MIT](https://python-sounddevice.readthedocs.io/en/latest/) | Optional demo playback. |
| PortAudio | [MIT-style license](https://portaudio.com/license.html) | Native library loaded by sounddevice. |
| SoundFile | [BSD-3-Clause](https://python-soundfile.readthedocs.io/en/latest/) | Optional file and demo I/O. |
| libsndfile | [LGPL, version 2.1 or 3](https://libsndfile.github.io/libsndfile/#licensing) | Native library loaded by SoundFile. |
| CFFI 2.1.1 | [MIT-0](https://github.com/python-cffi/cffi/blob/main/LICENSE) | Transitive dependency of audio I/O; check other versions separately. |
| pycparser | [BSD-3-Clause](https://github.com/eliben/pycparser/blob/main/LICENSE) | Transitive dependency of CFFI. |
| colorama | [BSD-3-Clause](https://github.com/tartley/colorama/blob/master/LICENSE.txt) | Transitive dependency of PyQtGraph. |
| typing-extensions | [PSF-2.0](https://github.com/python/typing_extensions/blob/main/LICENSE) | Transitive dependency of plotting. |

This table summarizes direct dependencies, selected transitive dependencies,
and the main native libraries;
it is not a complete license inventory of an executable bundle.
Transitive dependencies and native wheel components retain their own notices.
For example, NumPy/SciPy wheels can include BLAS and compiler runtimes;
Qt includes third-party code and tools with additional licenses.
Inspect the exact wheels and binaries included in each distribution.

## Redistributing dependencies

When bundling dependencies, preserve their copyright notices and complete
license texts. For LGPL components, provide the required corresponding source
by a license-permitted method, notices, and the required license copies, and
allow replacement/relinking and debugging of modifications to the libraries.
Do not add restrictions on those rights. Additional installation-information
requirements may apply under LGPLv3.
See [Qt's LGPL obligations](https://www.qt.io/development/open-source-lgpl-obligations)
and the [GNU license FAQ](https://www.gnu.org/licenses/gpl-faq.en.html).
The MIT license of this project does not by itself satisfy those obligations.

## Research references

Research references used to inform the design include:

- Dolby Laboratories, [DP563 User's Manual](https://professional.dolby.com/siteassets/pdfs/dolby-surround-and-pro-logic-ii-encoder-dp563-manual.pdf).
- Roger Dressler, Dolby Laboratories, *Dolby Surround Pro Logic II Decoder: Principles of Operation*, 2000, S00/13238.
- James M. Fosgate, [Surround processor, US5172415A](https://patents.google.com/patent/US5172415A/en).
- Jouni Paulus and Matteo Torcoli, [Geometrically-Motivated Primary-Ambient Decomposition With Center-Channel Extraction](https://arxiv.org/abs/2206.02125), 2022.
- Jianjun He and Woon-Seng Gan, [Applying Primary Ambient Extraction for Immersive Spatial Audio Reproduction](https://sigport.org/sites/default/files/HeG15_Applying%20Primary%20Ambient%20Extraction%20for%20Immersive%20Spatial%20Audio%20Reproduction-IEEE%20approved.pdf).
- FFmpeg, [surround filter source](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/af_surround.c).

No external paper PDF, figure image, audio recording,
FFmpeg source file, or third-party implementation is bundled in this project.
The decoder uses its own covariance model and coefficients; FFmpeg's
LGPL-licensed surround source was a comparison reference, not copied code.
External publications retain the rights and terms of their respective owners.

Dolby and Pro Logic are trademarks of Dolby Laboratories. References to them
identify the literature; this project is not a certified Dolby implementation
and does not claim Dolby affiliation or approval.
