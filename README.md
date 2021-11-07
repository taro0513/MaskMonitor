# MaskMonitor
A program that can analyze videos according to the weights you select

下載 訓練完的 [weight檔案](https://drive.google.com/file/d/11se-EVLSR7vXucge4VBPooYkuD6l9REF/view?usp=sharing)
執行 MaskDetection.py

內部可更改 輸入來源(鏡頭, 影片, 圖片) 以及輸出條件(人臉, 有帶口罩, 沒戴口罩)
<img width="504" alt="pro" src="https://user-images.githubusercontent.com/56122956/140647797-80b6cd5f-80f2-49a4-b831-57d897b7087e.png">

# 摘要
透過Kaggle以及NVIDIA所提供的配戴口罩人臉資料集，以及WIDER FACE的人臉偵測資料集，來訓練影像辨識模型 YOLOv4 ，並且為了補足預訓練模型對於黃種人辨識準確度較低的狀況，我們增加資料集內的黃種人的人臉數量。刪除重複影像的部分，則使用圖像感知算法(perceptual hash)，實現動態追蹤，刪除重複影像，去除相似度過高的圖片。最後，我們使用facenet人臉識別演算法，實行人臉比對，並以Trieplet loss 作為損失函數，以黃種人臉做為訓練資料，微調模型，在測試集上達到83.23%的準確度。本研究除了能讓使用者自行輸入影片得知關於接觸過關鍵人物的訊息，如接觸時間、地點等，也可擴大使用在各場所或是機構內部連接攝影機進行檢測，在偵測到關鍵人物時，即時通報關鍵人物的相關訊息。
