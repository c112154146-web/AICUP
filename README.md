# 🫀 AI CUP 2025 - Aortic Valve Object Detection (主動脈瓣物件偵測)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLO12-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

本專案為 **2025 AI CUP 主動脈瓣物件偵測競賽** 的解決方案程式碼。
我們採用最新的 **YOLO12n (Nano)** 模型架構，並透過 **遷移學習 (Transfer Learning)** 與 **快速微調 (Rapid Fine-tuning)** 策略，在極短的訓練週期內（30 Epochs）實現高效能的醫療影像偵測。

## 📂 專案結構 (Project Structure)

```text
.
├── AI_CUP_2025_aortic_valve_object_detection_train.ipynb   # [訓練] 資料前處理、環境建置與模型訓練腳本
├── AI_CUP_2025_aortic_valve_object_detection_predict.ipynb # [預測] 載入權重、推論測試集與生成提交檔
├── aortic_valve_colab.yaml                                 # [設定] 資料集路徑與類別定義檔
└── README.md                                               # 專案說明文件
