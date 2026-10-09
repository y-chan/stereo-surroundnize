"""PySide6/PyQtGraph presentation. All Qt and FFT work stays on the GUI thread."""

import queue
import os
import sys
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
# Keep pyqtgraph on the same binding even when PyQt is also installed.
os.environ["PYQTGRAPH_QT"] = "PySide6"
import pyqtgraph as pg

from ..decoder import CHANNELS, PROFILES
from .audio import PlaybackEngine, SourceInfo, default_mapping, inspect_source, output_devices
from .visualize import VisualBuffer


def clock_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds//60:02d}:{seconds%60:02d}"


def prepare_font():
    # Qt's offscreen platform on Windows does not enumerate system fonts.
    if sys.platform == "win32" and not QtGui.QFontDatabase.families():
        directory = Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"
        for filename in ("meiryo.ttc", "segoeui.ttf"):
            QtGui.QFontDatabase.addApplicationFont(str(directory/filename))
    if "Meiryo UI" in QtGui.QFontDatabase.families():
        QtWidgets.QApplication.instance().setFont(QtGui.QFont("Meiryo UI", 9))


class DemoWindow(QtWidgets.QMainWindow):
    def __init__(self, path=None, device=None, engine_factory=PlaybackEngine):
        super().__init__()
        prepare_font()
        self.engine_factory = engine_factory
        self.engine = None
        self.source = None
        self.visual = VisualBuffer(48000)
        self._reported_error = None
        self.setWindowTitle("Stereo Surround Demo")
        self.resize(1320, 900)
        pg.setConfigOptions(antialias=False, background="#111827", foreground="#cbd5e1")
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        title = QtWidgets.QLabel("Stereo → Surround")
        title.setStyleSheet("font-size: 24px; font-weight: 600; padding: 6px;")
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel("波形・スペクトルはデコード後の6chを表示。2ch出力では再生時に左右へミックスします。"))

        file_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("WAV / FLAC の音源を選択")
        self.open_button = QtWidgets.QPushButton("音源を開く…")
        self.open_button.clicked.connect(self.choose_file)
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(self.open_button)
        layout.addLayout(file_row)
        self.source_label = QtWidgets.QLabel("音源未選択")
        layout.addWidget(self.source_label)

        device_row = QtWidgets.QHBoxLayout()
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setMinimumContentsLength(35)
        self.device_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        refresh = QtWidgets.QPushButton("デバイス再取得")
        refresh.clicked.connect(self.refresh_devices)
        self.channels_combo = QtWidgets.QComboBox()
        self.channels_combo.addItems(["2ch", "6ch", "8ch"])
        self.channels_combo.setCurrentIndex(1)
        self.exclusive = QtWidgets.QCheckBox("WASAPI排他")
        device_row.addWidget(QtWidgets.QLabel("出力先"))
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(refresh)
        device_row.addWidget(self.channels_combo)
        device_row.addWidget(self.exclusive)
        if sys.platform == "win32":
            audio_settings = QtWidgets.QPushButton("Windows音声設定")
            audio_settings.clicked.connect(lambda: QtCore.QProcess.startDetached("control.exe", ["mmsys.cpl"]))
            device_row.addWidget(audio_settings)
        layout.addLayout(device_row)
        self.device_hint = QtWidgets.QLabel()
        self.device_hint.setWordWrap(True)
        layout.addWidget(self.device_hint)

        routing = QtWidgets.QGroupBox("スピーカー割り当て（2ch出力では左右にミックス）")
        grid = QtWidgets.QGridLayout(routing)
        self.map_spins, self.test_buttons, self.mutes = [], [], []
        for i, name in enumerate(CHANNELS):
            label = QtWidgets.QLabel(name)
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 8)
            button = QtWidgets.QPushButton(f"{name} テスト")
            button.clicked.connect(lambda checked=False, channel=i: self.play_test(channel))
            mute = QtWidgets.QCheckBox("ミュート")
            mute.toggled.connect(self.apply_volume)
            grid.addWidget(label, 0, i)
            grid.addWidget(spin, 1, i)
            grid.addWidget(button, 2, i)
            grid.addWidget(mute, 3, i)
            self.map_spins.append(spin)
            self.test_buttons.append(button)
            self.mutes.append(mute)
        layout.addWidget(routing)

        self.settings_box = QtWidgets.QGroupBox("2ch音源の変換設定（再生前に変更）")
        settings_row = QtWidgets.QHBoxLayout(self.settings_box)
        self.listening_combo = QtWidgets.QComboBox()
        self.listening_combo.addItems(["music", "standard"])
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["ambience", "matrix"])
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.addItems(list(PROFILES))
        self.profile_combo.setCurrentText("balanced")
        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.width_spin.setRange(0, 1)
        self.width_spin.setSingleStep(.05)
        self.width_spin.setValue(.35)
        self.rear_spin = QtWidgets.QDoubleSpinBox()
        self.rear_spin.setRange(0, 4)
        self.rear_spin.setSingleStep(.1)
        self.rear_spin.setValue(.7)
        self.onset_check = QtWidgets.QCheckBox("過渡時の追従を速める")
        for label, widget in (("用途", self.listening_combo), ("方式", self.mode_combo),
                              ("窓", self.profile_combo), ("センター幅", self.width_spin),
                              ("リアゲイン", self.rear_spin)):
            settings_row.addWidget(QtWidgets.QLabel(label))
            settings_row.addWidget(widget)
        settings_row.addWidget(self.onset_check)
        layout.addWidget(self.settings_box)

        transport = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("▶ 再生")
        self.pause_button = QtWidgets.QPushButton("一時停止")
        self.stop_button = QtWidgets.QPushButton("停止")
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.stop_button.clicked.connect(self.stop)
        self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 10000)
        self.seek_slider.sliderReleased.connect(self.seek)
        self.time_label = QtWidgets.QLabel("00:00 / 00:00")
        self.volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.volume_slider.setRange(-60, 0)
        self.volume_slider.setValue(-12)
        self.volume_slider.setMaximumWidth(150)
        self.volume_slider.valueChanged.connect(self.apply_volume)
        self.volume_label = QtWidgets.QLabel("−12 dB")
        for widget in (self.play_button, self.pause_button, self.stop_button):
            transport.addWidget(widget)
        transport.addWidget(self.seek_slider, 1)
        transport.addWidget(self.time_label)
        transport.addWidget(QtWidgets.QLabel("音量"))
        transport.addWidget(self.volume_slider)
        transport.addWidget(self.volume_label)
        layout.addLayout(transport)

        wave_controls = QtWidgets.QHBoxLayout()
        wave_controls.addWidget(QtWidgets.QLabel("波形の縦軸"))
        self.wave_range = QtWidgets.QComboBox()
        self.wave_range.addItems(["±1.0", "±0.25", "±0.05"])
        self.wave_range.setCurrentIndex(1)
        self.wave_range.currentIndexChanged.connect(self.draw)
        wave_controls.addWidget(self.wave_range)
        wave_controls.addStretch()
        layout.addLayout(wave_controls)

        plots = pg.GraphicsLayoutWidget()
        plots.addLabel("波形：直近100 ms", row=0, col=0)
        plots.addLabel("スペクトログラム：直近約5秒 / 縦軸 Hz / −100 ～ 0 dBFS", row=0, col=1)
        self.wave_curves, self.wave_plots = [], []
        self.images, self.spec_plots = [], []
        lookup = pg.colormap.get("viridis").getLookupTable(nPts=256)
        colors = ["#60a5fa", "#22d3ee", "#fbbf24", "#fb7185", "#a78bfa", "#34d399"]
        for i, name in enumerate(CHANNELS):
            plot = plots.addPlot(row=i+1, col=0)
            plot.setLabel("left", name)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.setYRange(-1, 1, padding=0)
            plot.setXRange(-100, 0, padding=0)
            plot.setMouseEnabled(x=False, y=False)
            plot.showGrid(x=True, y=True, alpha=.15)
            if i != 5:
                plot.hideAxis("bottom")
            else:
                plot.setLabel("bottom", "直近の時間", units="ms")
            curve = plot.plot(pen=pg.mkPen(colors[i], width=1))
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            self.wave_plots.append(plot)
            self.wave_curves.append(curve)
            spec = plots.addPlot(row=i+1, col=1)
            spec.getAxis("left").enableAutoSIPrefix(False)
            spec.getAxis("left").setWidth(55)
            spec.setMouseEnabled(x=False, y=True)
            if i != 5:
                spec.hideAxis("bottom")
            else:
                spec.setLabel("bottom", "履歴", units="s")
            if self.spec_plots:
                spec.setYLink(self.spec_plots[0])
            image = pg.ImageItem(axisOrder="row-major")
            image.setLookupTable(lookup)
            spec.addItem(image)
            self.spec_plots.append(spec)
            self.images.append(image)
            plots.ci.layout.setRowStretchFactor(i+1, 1)
        plots.ci.layout.setColumnStretchFactor(0, 4)
        plots.ci.layout.setColumnStretchFactor(1, 6)
        layout.addWidget(plots, 1)
        layout.addWidget(QtWidgets.QLabel("各行は同じチャンネルの波形とスペクトログラムです。周波数範囲はマウスで拡大でき、6chで連動します。"))
        self.status = QtWidgets.QLabel("停止中。出力先と割り当てを確認して再生してください。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._locked_widgets = [self.device_combo, refresh, self.channels_combo,
                                self.exclusive, *self.map_spins]
        self.channels_combo.currentIndexChanged.connect(self.reset_mapping)
        self.device_combo.currentIndexChanged.connect(self.device_changed)
        self.refresh_devices()
        if device is not None:
            matches = []
            for i in range(self.device_combo.count()):
                d = self.device_combo.itemData(i)
                if str(device) == str(d["id"]) or str(device).casefold() in (d["name"]+" "+d["api"]).casefold():
                    matches.append(i)
            if not matches:
                raise ValueError(f"Output device not found: {device}. Use --list-devices.")
            if self.device_combo.currentIndex() not in matches:
                self.device_combo.setCurrentIndex(matches[0])
        self.reset_mapping()
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self._set_running(False)
        self.draw()
        if path:
            self.load_file(path)

    def refresh_devices(self):
        try:
            devices = output_devices()
            old = self.device_combo.currentData()
            self.device_combo.blockSignals(True)
            self.device_combo.clear()
            for d in devices:
                self.device_combo.addItem(f"{d['id']}: {d['name']} / {d['api']} / {d['channels']}ch", d)
            self._devices = devices
            def priority(d):
                return (0 if d["channels"] >= 6 else 1,
                        0 if "WASAPI" in d["api"] else 1 if "DirectSound" in d["api"] else 2)
            chosen = old if old and any(d["id"] == old["id"] for d in devices) else min(devices, key=priority, default=None)
            if chosen:
                self.device_combo.setCurrentIndex(next(i for i, d in enumerate(devices) if d["id"] == chosen["id"]))
            self.device_combo.blockSignals(False)
            self.device_changed()
        except Exception as exc:
            self.device_combo.blockSignals(False)
            self.status.setText(f"音声デバイスを取得できません：{exc}")

    def device_changed(self):
        d = self.device_combo.currentData()
        if d:
            self.channels_combo.setCurrentIndex(2 if d["channels"] >= 8 else 1 if d["channels"] >= 6 else 0)
            self.reset_mapping()
            self.exclusive.setChecked(False)
            self.exclusive.setEnabled("WASAPI" in d["api"])
            if d["channels"] < 6:
                self.device_hint.setText(f"選択した出力は{d['channels']}chです。2chモードで左右にミックスして再生します。センターとLFEは左右へ、サラウンドは対応する左右へ振り分けます。")
            else:
                self.device_hint.setText("8chの標準割当はSL/SRが7/8番です。6chでは5/6番になります。テスト音で実スピーカーとの対応を確認できます。")

    def reset_mapping(self):
        count = (2, 6, 8)[self.channels_combo.currentIndex()]
        for spin, slot in zip(self.map_spins, default_mapping(count)):
            spin.setEnabled(count != 2)
            spin.setMaximum(count)
            spin.setValue(slot+1)

    def choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "音源を開く", "", "音声 (*.wav *.flac *.aiff *.ogg);;すべて (*)")
        if path:
            self.load_file(path)

    def load_file(self, path):
        try:
            info = inspect_source(path)
            self.stop()
            self.source = info
            self.path_edit.setText(info.path)
            operation = "リアルタイム変換" if info.channels == 2 else "6chパススルー（FL FR FC LFE SL SRの順）"
            self.source_label.setText(f"{Path(info.path).name}  |  {info.rate:,} Hz  |  {info.channels}ch  |  {clock_text(info.duration)}  |  {operation}")
            self.visual = VisualBuffer(info.rate)
            self.seek_slider.setValue(0)
            self.time_label.setText(f"00:00 / {clock_text(info.duration)}")
            self._set_running(False)
            self.draw()
        except Exception as exc:
            self.show_error(str(exc))

    def decoder_settings(self):
        return dict(listening_mode=self.listening_combo.currentText(), mode=self.mode_combo.currentText(),
                    profile=self.profile_combo.currentText(), center_width=self.width_spin.value(),
                    rear_gain=self.rear_spin.value(), onset_adaptation=self.onset_check.isChecked())

    def begin(self, info, start_frame=0, tone_channel=None, paused=False):
        # 旧producerの終了を待ってから、新しいデバイス出力とDSP状態を作る。
        # シークやテスト音への切り替えでも同じ経路を使い、セッションを重ねない。
        self.stop()
        d = self.device_combo.currentData()
        if d is None:
            raise ValueError("出力デバイスを選択してください")
        channels = (2, 6, 8)[self.channels_combo.currentIndex()]
        self.engine = self.engine_factory(info, d["id"], output_channels=channels,
                                          mapping=tuple(s.value()-1 for s in self.map_spins),
                                          settings=self.decoder_settings(), start_frame=start_frame,
                                          tone_channel=tone_channel, exclusive=self.exclusive.isChecked())
        self.visual = VisualBuffer(info.rate)
        self.apply_volume()
        # pausedはスレッド開始前に設定する。開始後だと、シーク直後に一瞬だけ
        # 再生されてから停止する競合が生じる可能性がある。
        self.engine.paused = paused
        self._reported_error = None
        self.engine.start()
        self._set_running(True)
        self.pause_button.setText("再開" if paused else "一時停止")
        self.status.setText("音声出力を準備しています…")

    def play(self):
        try:
            if self.engine and self.engine.state in ("loading", "playing") and self.engine.tone_channel is None:
                self.engine.paused = False
                self.pause_button.setText("一時停止")
                return
            if self.source:
                frame = round(self.source.frames*self.seek_slider.value()/10000)
                if frame >= self.source.frames:
                    frame = 0
                self.begin(self.source, frame)
        except Exception as exc:
            self.show_error(str(exc))

    def play_test(self, channel):
        try:
            d = self.device_combo.currentData()
            if d is None:
                raise ValueError("出力デバイスを選択してください")
            rate = round(d["rate"])
            self.begin(SourceInfo(f"test:{CHANNELS[channel]}", rate, rate, 6), tone_channel=channel)
        except Exception as exc:
            self.show_error(str(exc))

    def pause(self):
        if self.engine and self.engine.state in ("loading", "playing"):
            self.engine.paused = not self.engine.paused
            self.pause_button.setText("再開" if self.engine.paused else "一時停止")

    def stop(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self._set_running(False)
        self.pause_button.setText("一時停止")
        self.status.setText("停止中")

    def seek(self):
        if self.engine and self.engine.tone_channel is None and self.source:
            paused = self.engine.paused
            try:
                frame = round(self.source.frames*self.seek_slider.value()/10000)
                self.begin(self.source, frame, paused=paused)
            except Exception as exc:
                self.show_error(str(exc))

    def apply_volume(self, *args):
        db = self.volume_slider.value()
        self.volume_label.setText(f"{db} dB")
        if self.engine:
            self.engine.gain = 10**(db/20)
            self.engine.muted = tuple(m.isChecked() for m in self.mutes)

    def _set_running(self, running):
        for widget in self._locked_widgets:
            widget.setEnabled(not running)
        if not running and self.channels_combo.currentIndex() == 0:
            for spin in self.map_spins:
                spin.setEnabled(False)
        d = self.device_combo.currentData()
        self.exclusive.setEnabled(not running and bool(d and "WASAPI" in d["api"]))
        self.settings_box.setEnabled(not running and bool(self.source and self.source.channels == 2))
        self.play_button.setEnabled(self.source is not None)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        self.seek_slider.setEnabled(self.source is not None and not (running and self.engine and self.engine.tone_channel is not None))

    def tick(self):
        # Qt操作と表示用FFTはGUIスレッドだけで行う。
        # https://doc.qt.io/qt-6/threads-qobject.html
        e = self.engine
        if e is None:
            return
        changed = False
        # キュー上限と同じ24個までに制限し、producerが追加し続けてもGUI処理を終える。
        # 履歴のFFTは全パケットへ適用するが、画面の再描画要求は1 tickにつき1回。
        for _ in range(24):
            try:
                start, audio = e.visual.get_nowait()
            except queue.Empty:
                break
            self.visual.consume(start, audio)
            changed = True
        if changed:
            self.draw()
        if e.tone_channel is None:
            self.time_label.setText(f"{clock_text(e.position)} / {clock_text(e.info.duration)}")
            if not self.seek_slider.isSliderDown():
                self.seek_slider.setValue(round(10000*e.position/max(e.info.duration, .001)))
        if e.error and e.error != self._reported_error:
            self._reported_error = e.error
            self._set_running(False)
            self.show_error(e.error)
        elif e.state in ("finished", "stopped"):
            self._set_running(False)
            self.status.setText("再生終了" if e.state == "finished" else "停止中")
        elif e.state == "playing":
            label = f"{CHANNELS[e.tone_channel]} テスト" if e.tone_channel is not None else ("一時停止" if e.paused else "再生中")
            self.status.setText(f"{label}  |  出力バッファ {e.latency*1000:.0f} ms  |  音声不足 {e.fifo.starved_frames} samples  |  PortAudio不足 {e.portaudio_underflows} 回  |  振幅制限 {e.fifo.clipped_samples} samples  |  描画欠落 {e.visual_drops} 回")

    def draw(self, *args):
        v = self.visual
        amplitude = (1.0, .25, .05)[self.wave_range.currentIndex()]
        for i, curve in enumerate(self.wave_curves):
            curve.setData(v.wave_time, v.wave[:, i])
            self.wave_plots[i].setYRange(-amplitude, amplitude, padding=0)
        for i, image in enumerate(self.images):
            image.setImage(v.image(i), autoLevels=False, levels=(-100, 0))
        # 再生中に軸範囲を設定し直すと、利用者の周波数ズームが毎tick解除される。
        # rate変更時だけ初期化し、他の5chは先頭のプロットへのYリンクで追従させる。
        if getattr(self, "_plot_rate", None) != v.rate:
            for image, spec in zip(self.images, self.spec_plots):
                image.setRect(QtCore.QRectF(-v.history_seconds, 0, v.history_seconds, v.rate/2))
                spec.setXRange(-v.history_seconds, 0, padding=0)
            self.spec_plots[0].setYRange(0, min(12000, v.rate/2), padding=0)
            self._plot_rate = v.rate

    def show_error(self, message):
        self.status.setText(f"エラー：{message}")
        QtWidgets.QMessageBox.critical(self, "再生できません", message+"\n\n出力デバイスのスピーカー構成、出力ch数、音源のサンプルレートを確認してください。WindowsでWASAPIが2chの場合は5.1/7.1を設定してデバイスを再取得します。")

    def closeEvent(self, event):
        try:
            self.stop()
            event.accept()
        except RuntimeError as exc:
            self.show_error(str(exc))
            event.ignore()
