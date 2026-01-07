# 🏆 AI CUP 2025 - Aortic Valve Object Detection (主動脈瓣物件偵測)

[![Rank](https://img.shields.io/badge/Private_LB-Rank_1-gold)](https://aidea-web.tw/)
[![Score](https://img.shields.io/badge/Score-0.9326-brightgreen)]()
[![Model](https://img.shields.io/badge/Model-YOLO12n-blue)]()
[![Framework](https://img.shields.io/badge/Framework-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)

> **TEAM_9088** (賴柏瑋)  
> **Private Leaderboard Score:** 0.9326472252652411  
> **Rank:** 1 (基於 Private LB)

## 📖 專案簡介 (Overview)

本專案為 **2025 AI CUP 主動脈瓣物件偵測競賽** 的第一名解決方案 (Rank 1 Solution)。
針對醫療影像數據量大且特徵細微的挑戰，我們採用了極致輕量化的 **YOLO12n (Nano)** 模型架構。透過 **遷移學習 (Transfer Learning)** 結合 **快速微調 (Rapid Fine-tuning)** 策略，僅需 **30 Epochs** 的訓練即可達到極高的偵測精度 (mAP > 0.93)，實現了準確度與運算效率的最佳平衡。

## 📂 檔案結構 (Repository Structure)

```text
AICUP-2025/
├── AI_CUP_2025_aortic_valve_object_detection_train.ipynb   # [核心] 模型訓練程式碼 (含資料處理)
├── AI_CUP_2025_aortic_valve_object_detection_predict.ipynb # [核心] 推論與提交檔生成程式碼
├── aortic_valve_colab.yaml                                 # [設定] YOLO 資料集路徑配置檔
└── README.md                                               # 專案說明文件
