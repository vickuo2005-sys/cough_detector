# Sound Detector App

這是一個 Flutter + Android Kotlin 的聲音偵測 APP 範本。

功能：

- 不需要外接麥克風，先使用手機內建麥克風測試
- 之後接 3.5mm / USB 外接麥克風時，Android 會依系統音訊路由自動使用可用輸入
- 即時計算 RMS 音量能量
- 超過 threshold 時顯示「偵測到聲音」並觸發手機震動
- 可在畫面上調整 threshold

---

## 使用方式

### 1. 建立 Flutter 專案

```bash
flutter create sound_detector_app
cd sound_detector_app
```

### 2. 用本資料夾中的檔案覆蓋新專案

請把 GitHub 這個資料夾中的檔案覆蓋到你剛建立的 Flutter 專案：

```text
lib/main.dart
pubspec.yaml
android/app/src/main/AndroidManifest.xml
android/app/src/main/kotlin/com/example/sound_detector_app/MainActivity.kt
```

如果你的 package name 不是：

```text
com.example.sound_detector_app
```

請同步修改 `MainActivity.kt` 最上面的 package 名稱，以及 Kotlin 檔案所在路徑。

---

## 執行

```bash
flutter pub get
flutter run
```

第一次開啟 APP 時，請允許麥克風權限。

---

## 測試方式

1. 按下「開始偵測」
2. 對手機說話、拍手、敲桌子
3. 如果 RMS 超過 threshold，畫面會顯示「偵測到聲音」並震動
4. 若太敏感，把 threshold 調高；若偵測不到，把 threshold 調低

---

## 核心原理

APP 會從麥克風讀取 16 kHz、mono、PCM 16-bit 音訊，並計算 RMS：

```text
RMS = sqrt(mean(x^2))
```

當 RMS 大於 threshold 時，就判定為有聲音。

---

## 後續可擴充

- 偵測到聲音時播放警報聲
- 偵測到聲音時自動錄音
- 加入 YAMNet 或自訓練模型進行聲音分類
- 分辨飛機聲、玻璃破裂聲、咳嗽聲等
