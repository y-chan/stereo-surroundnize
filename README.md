# stereo-surroundnize

ステレオ音声を6chへ変換する、NumPyとSciPyによるストリーミングデコーダです。
WAV・FLACのファイル変換、TouchDesignerのScript CHOP、Qtによる再生デモに対応します。

合成信号とファイル入出力の自動テストは通過しています。
TouchDesigner本体と実際の5.1再生環境での検証はまだ行っていません。
Dolby Pro Logic IIの認証実装やビット単位の再現を目的とするものではありません。

相関を持つ直接音を前方に残し、推定した残響をリアへ送る方式を既定にします。
逆相成分をリアへ残すマトリクス向けモードも設け、声の抑制とのトレードオフを明示します。
声の意味を認識する処理は含まないため、ステレオ化された声や残響まで完全に除去できるとは限りません。

## インストールとファイル変換

Python 3.10以降を対象とします。
FLACも読み込む場合は、ファイルIO用の追加依存をインストールします。

```powershell
python -m pip install -e ".[files]"
python -m stereo_surround input.flac output.wav
python -m stereo_surround input.wav output.wav --mode matrix --subtype pcm24
python -m stereo_surround input.wav output.wav --onset-adaptation
python -m stereo_surround input.wav output_music.wav --listening-mode music --center-width 0.35
```

既定の出力はfloat32の6ch WAVで、列順はFL、FR、FC、LFE、SL、SRです。
PCM出力の振幅超過はエラーとなり、`--normalize`を付けると全体に一定ゲインを掛けてピークを0.99へ合わせます。
既存出力は`--overwrite`を指定した場合だけ上書きします。
入力と同じファイルへの出力は拒否します。
`--onset-adaptation`は相関を持つ音の急な増加に追従を速めるオプションです。
合成音の入り際では改善しましたが残響も減るため、既定では無効にしています。

`--listening-mode music`は、PLIIの動作原理資料を参考に追加した設定です。
追加リア遅延を0 msとし、リアの高域を軽いシェルフで減衰させます。
低域を抑える250 HzのHPFと、声に相当する相関音を抑えるambience方式は維持します。
シェルフの−3 dBと中点3000 Hzは独自の比較用初期値です。

`--center-width`は0で従来のセンター分配、1でセンターを使わず元のステレオをフロント左右へ返す設定です。
前後バランスは`--rear-gain`で調整できます（既定0.7）。
追加リア遅延は`--rear-delay-ms`、高域減衰量は`--rear-shelf-db`、中点周波数は`--rear-shelf-hz`で上書きできます。
例えばMusicのシェルフを外して比較するには`--rear-shelf-db 0`を指定します。

## リアルタイム再生デモ

GUIと音声デバイス用の依存はdemo追加依存に分離しています。

```powershell
uv run --extra demo python -m stereo_surround.demo "/path/to/audio.wav"
```

pipの場合は`python -m pip install -e ".[demo]"`で追加依存を入れ、`python -m stereo_surround.demo`で起動します。
音源の選択、再生、一時停止、シーク、6chそれぞれの波形とスペクトログラムの同時表示に対応しています。
波形とスペクトルは、出力チャンネル数にかかわらずデコード後の6ch信号を表示します。
出力先と2ch／6ch／8chの出力を選び、各チャンネルのテストボタンで実スピーカーとの対応を確認できます。
2chデバイスでは6ch音源も再生でき、再生時にセンターとLFEを左右へ、サラウンドを対応する左右へミックスします。
6ch／8ch再生では、出力デバイスと接続先が同じチャンネル数のPCMに対応している必要があります。
Windowsでは、下記の[スピーカー構成の設定](#windowsのスピーカー構成)で対象デバイスを5.1または7.1に設定します。
標準の割り当ては6chがFL・FR・FC・LFE・SL・SR、8chがFL・FR・FC・LFE・無音・無音・SL・SRです。
6ch入力もFL・FR・FC・LFE・SL・SRの列順として読みます。
起動時には自動再生しません。

## Windowsのスピーカー構成

再生デモとTouchDesignerを使う前に、Windows側でも出力先のスピーカー構成を確認します。
チャンネル数は、従来のサウンド画面の「構成」から選びます。

1. `Win + R`を押し、`mmsys.cpl`を実行する。
2. 「再生」タブで、使用するAVアンプやオーディオデバイスの出力を選ぶ。
3. 左下の「構成」を開く。
4. ステレオ（2ch）、5.1サラウンド（6ch）、7.1サラウンド（8ch）から、使用する構成を選ぶ。
5. テスト音で各スピーカーの対応を確認し、ウィザードを完了する。

「設定 → システム → サウンド」の出力形式とは別の設定です。
「構成」が無効、または必要な構成が表示されない場合は、対象デバイスと接続先、ドライバのマルチチャンネルPCM対応を確認します。
スピーカー構成はWindowsのサウンドコントロールパネルで設定されます。[Microsoftの説明](https://learn.microsoft.com/en-us/windows/win32/coreaudio/pkey-audioendpoint-fullrangespeakers)

## 連続した音声ブロックの処理

```python
import numpy as np
from stereo_surround import SurroundDecoder

decoder = SurroundDecoder(48000)
stereo = np.zeros((800, 2), dtype=np.float32)  # 実際のステレオ入力に置き換える
surround = decoder.process(stereo)           # (800, 6)
```

Music用のAPI設定は`SurroundDecoder(48000, listening_mode="music", center_width=0.35)`です。

同じdecoderを次のブロックでも使います。
ブロック長は可変でよく、既定では1024サンプル（48 kHzで21.33 ms）の固定遅延があります。
リアには追加10 msの遅延があり、デバイスのバッファやcook周期は別に加わります。
musicでは追加リア遅延が0 msになりますが、STFTの固定遅延は残ります。
## TouchDesignerから使う（Windows / uv）

TouchDesignerのTextportで、同梱PythonとNumPyの版を確認します。
外部パッケージを同じPythonのメジャー・マイナー版、同じNumPyの版で用意します。

```python
import sys, numpy
print(sys.version_info[:2], numpy.__version__)
```

例えばPython 3.11、NumPy 2.1.2の場合は、プロジェクトのフォルダで次を実行します。
表示された版が異なる場合は、コマンドの版を置き換えます。
[`uv venv`と`uv pip install --python`](https://docs.astral.sh/uv/pip/environments/)で専用環境を指定するため、activateは不要です。
GUIやファイルIOの追加依存はインストールしません。

```powershell
uv venv --python 3.11 .venv-td
uv pip install --python .venv-td/Scripts/python.exe . "numpy==2.1.2"
(Resolve-Path .venv-td/Lib/site-packages).Path
```

[TouchDesignerの外部Python設定](https://derivative.ca/UserGuide/Python#Installing_Custom_Python_Packages)で、Edit → Preferencesの「Add External Python to Search Path」を有効にします。
「Python 64 bit Module Path」に最後のコマンドで表示された絶対パスを指定し、TouchDesignerを再起動します。
既存のパスも残す場合は`;`で区切ります。
TouchDesigner内のPythonでこの環境のパッケージを読み込む方式なので、仮想環境のPythonへ切り替える操作は不要です。

Textportでインポートを確認します。

```python
import numpy, scipy, stereo_surround
print(numpy.__version__, scipy.__version__)
print(stereo_surround.__file__)
```

2chのTime Slice入力をScript CHOPへ接続し、[コールバック例](examples/touchdesigner_callbacks.py)をCallbacks DATへ貼り付けます。
Setup Parametersを押してReset操作を追加し、設定を変えたときはResetを押します。
出力順はFL、FR、FC、LFE、SL、SRです。
Audio Device Out CHOP側でも6chの割り当てを確認します。
Windows側の[スピーカー構成](#windowsのスピーカー構成)と、Audio Device Outの出力先が同じデバイスになっていることも確認します。
DirectSoundの場合はStereo Modeを無効にし、Front Left、Front Right、Front Center、Low Frequencyと、デバイスが対応するリア左右の計6出力を有効にします。
入力チャンネルは、有効にしたスピーカー出力の順に割り当てられます。
ActiveとCook Every Frameを有効にし、Volumeが0でないことも確認します。[Audio Device Out公式設定](https://derivative.ca/UserGuide/Audio_Device_Out_CHOP)

音が出ない場合は、元の2ch入力を同じAudio Device Outへ直接接続して再生できるかを確認します。
Script CHOPの中クリック情報で、Sample Rateが入力の音声レート（例えば48000）、Samplesが入力スライスと同じ、Time Sliceが有効になっているかを確認します。
48 kHz・60 FPSで通常の1フレーム分なら約800サンプルで、1サンプルや60 Hzの出力は意図した音声ブロックではありません。
最初はDSPの固定遅延により無音が出るため、タイムラインを動かし、複数フレームを連続してcookさせます。
波形表示の見た目だけでは、音声データが正しく届いているかは判断できません。

再生中に音の高さが揺れる場合は、Audio Device Outを参照するInfo CHOPの`queue_speed`を確認します。
正常な再生速度は1.0で、Audio Device OutのAdjust Speedによるキュー補正は音の高さを変える場合があります。[公式のキュー情報と速度補正](https://derivative.ca/UserGuide/Audio_Device_Out_CHOP)
入力・Script CHOPのSample Rate、Samples、startが一致することも確認し、キュー速度補正とサンプルレートの不一致を切り分けます。
`queue_speed`が揺れる場合はCook Every Frameを有効にし、Buffer Lengthを例えば0.15秒から0.2秒へ増やして比較します。
Buffer Lengthに応じて再生遅延が増えるため、CHOP Maximum Time Slice Sizeがその値以上であることも確認します。
Adjust Speedを下げる比較はできますが、処理の遅れやキュー不足そのものが解消するわけではありません。

ソースコードを変更した場合は、上の`uv pip install`コマンドを再実行してからTouchDesignerを再起動します。
通常のインストールを使うことで、editable installの`.pth`処理やプロジェクトの`sys.path`追加をコールバックへ書かずに済みます。
TouchDesigner更新後にPythonやNumPyの版が変わった場合も、それに合わせて専用環境を更新します。

## 検証と残っている項目

```powershell
python -m unittest discover -s tests -v
python -m stereo_surround.benchmark --seconds 60 --block 800
python -m stereo_surround.quality
python -m stereo_surround.longrun --seconds 1800
python -m stereo_surround.sweep --output results.json
```

処理時間は上のベンチマークコマンドで利用環境ごとに確認できます。
TouchDesigner同梱PythonでもDSPの動作を確認していますが、本体のcookと5.1デバイスでの聴取は未検証です。
ファイル変換はWAVとFLACをブロック単位で処理します。
正規化を指定した場合は、処理済み音声を一時ファイルへ保存して全体のピークを測り、一定ゲインで書き出します。

## ライセンス

自作コード、テスト、サンプル、文書は[MIT](LICENSE)で公開します。
デモはPySide6とPyQtGraphを使い、GUI依存はdemo追加依存のままです。
依存ライブラリと外部文献にはそれぞれのライセンスが適用されます。
音源と参照PDFは同梱しません。
依存のライセンスと配布条件は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)に記載しています。
