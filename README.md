# 🫀 AI CUP 2025 - Aortic Valve Object Detection (主動脈瓣物件偵測)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLO12-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

本專案為 **2025 AI CUP 主動脈瓣物件偵測競賽** 的解決方案程式碼。
針對醫療影像數據量大且特徵細微的挑戰，本專案採用 **YOLO12n (Nano)** 模型架構，透過 **遷移學習 (Transfer Learning)** 與 **快速微調 (Rapid Fine-tuning)** 策略，建立高效率的物件偵測流程。

本文件旨在說明如何配置環境、處理資料以及重現訓練與預測結果。

## 📂 專案結構 (Repository Structure)

```text
.
├── AI_CUP_2025_aortic_valve_object_detection_train.ipynb   # [訓練模組] 資料前處理、模型訓練
├── AI_CUP_2025_aortic_valve_object_detection_predict.ipynb # [預測模組] 推論測試集、生成提交檔
├── aortic_valve_colab.yaml                                 # [設定檔] YOLO 資料集路徑配置
└── README.md                                               # 專案說明文件

```

## 🛠️ 環境配置 (Environment Setup)

本專案建議在 **Google Colab (GPU)** 環境下執行，以確保與開發環境一致。

### 1. 硬體需求

* **GPU**: 建議使用 NVIDIA Tesla T4 (16GB VRAM) 或更高規格。
* **RAM**: 12GB+

### 2. 軟體依賴 (Dependencies)

若於本地端執行，請確保安裝以下套件：

```bash
# 安裝 Ultralytics YOLO 框架
pip install ultralytics

# 安裝影像處理與數據分析工具
pip install opencv-python numpy pandas matplotlib

```

## 🧩 模組功能與輸入/輸出 (Modules I/O)

為了方便第三方使用者除錯與重現，以下說明各模組的核心功能與資料流：

### 1. 訓練模組 (`...train.ipynb`)

* **功能**：負責資料下載、格式轉換、資料集分割以及模型訓練。
* **📥 輸入 (Input)**：
* 原始醫療影像資料集 (Images)
* 原始標註檔案 (XML/JSON annotations)
* `aortic_valve_colab.yaml` 設定檔


* **📤 輸出 (Output)**：
* **YOLO 格式資料集**：轉換後的 `.txt` 標註檔 (Normalized coordinates)。
* **權重檔**：`runs/detect/train/weights/best.pt` (驗證集分數最高的模型)。
* **訓練日誌**：Loss 曲線圖、Confusion Matrix、F1-Score 曲線。



### 2. 預測模組 (`...predict.ipynb`)

* **功能**：載入訓練好的權重，對測試集進行推論並格式化輸出。
* **📥 輸入 (Input)**：
* `best.pt` (訓練好的權重檔)
* 測試集影像資料夾 (Test Images)


* **📤 輸出 (Output)**：
* **提交檔**：符合競賽格式的 CSV 檔案 (包含 ImageID, Label, Confidence, BBox)。
* **視覺化結果**：繪製預測框的影像，供人工檢核。



## 🚀 重現步驟 (Reproduction Guide)

### 步驟 1：資料準備與訓練 (Training)

開啟 `AI_CUP_2025_aortic_valve_object_detection_train.ipynb` 並依序執行：

1. **環境初始化**：程式會自動檢查 CUDA 環境並安裝 `ultralytics`。
2. **資料前處理**：
* 自動將資料集分割為 **訓練集 (80%)** 與 **驗證集 (20%)**。
* 將標註座標歸一化為 `(x_center, y_center, width, height)` 格式。


3. **開始訓練**：
程式將執行以下核心指令進行微調：
```python
model = YOLO('yolo12n.pt')  # 載入預訓練權重
results = model.train(
    data="./aortic_valve_colab.yaml",
    epochs=30,      # 設定訓練 30 輪
    batch=16,       # Batch size
    imgsz=640,      # 輸入影像大小
    device=0        # 使用 GPU
)

```



### 步驟 2：推論與評估 (Inference)

訓練完成後，開啟 `AI_CUP_2025_aortic_valve_object_detection_predict.ipynb`：

1. **指定模型路徑**：確保指向訓練產生的 `best.pt`。
2. **執行推論**：程式會對測試集影像進行批量預測。
3. **後處理 (Post-processing)**：
* 將 YOLO 輸出的歸一化座標 `(xywh)` 還原為原始像素座標 `(xyxy)`。
* 輸出最終 CSV 檔案。



## ⚙️ 參數設定說明 (Configuration)

本專案針對主動脈瓣偵測任務進行了參數優化，主要設定如下：

| 參數 | 設定值 | 說明 |
| --- | --- | --- |
| **Model** | `YOLO12n` | 選擇 Nano 版本以平衡速度與精度。 |
| **Epochs** | `30` | 採用快速微調策略，避免過擬合。 |
| **Image Size** | `640` | 標準輸入解析度，保留足夠特徵細節。 |
| **Batch Size** | `16` | 配合 T4 GPU 記憶體限制的最佳設定。 |
| **Optimizer** | `Auto` | 自動選擇 (通常為 SGD + Momentum)。 |

```

```
