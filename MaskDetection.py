import numpy as np
import argparse
import time
import cv2
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# from MaskDetectionUi import Ui_MainWindow
from b import Ui_MainWindow
import sys
from threading import Thread, Lock
import cv2
import traceback


labelsPath = r'face.names'
weightsPath = r'yolov4_final.weights'
configPath = r'yolov4.cfg'
#顏色設定

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, __Camerastop__:bool = False, 
                 labelsPath:str = r'face.names',
                 weigthsPath:str = r'yolov4_final.weights',
                 configPath:str = r'yolov4.cfg',
                 confidence:float = 0.5,
                 threshold:float = 0.3,
                 delete_threshold:float = 0.765,
                 DynamicUpdateTracking:bool = True,
                 save:str = 'detect/') -> None:
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.choose_file.clicked.connect(self.setFile)
        self.ui.start.clicked.connect(self.run)

        self.ui.tableWidget.horizontalHeader().resizeSection(0, 100)
        self.ui.tableWidget.horizontalHeader().resizeSection(1, 100)
        self.ui.tableWidget.horizontalHeader().resizeSection(2, 80)
        self.ui.tableWidget.horizontalHeader().resizeSection(3, 80)
        self.ui.tableWidget.horizontalHeader().resizeSection(4, 100)
        self.ui.tableWidget.horizontalHeader().resizeSection(5, 100)
        self.ui.tableWidget.horizontalHeader().resizeSection(6, 300)
        # self.ui.start.clicked.connect(self.detect)

        self.__Camerastop__ = False
        self.__Camera__ = 0
        self.cap = None

        self.labelsPath = labelsPath
        self.weightsPath = weightsPath
        self.configPath = configPath

        self.confidence = confidence
        self.threshold = threshold
        self.delete_threshold = delete_threshold
        self.DynamicUpdateTracking = DynamicUpdateTracking

        self.save = save

        self.__PHASH__ = []
        self.with_mask = 0
        self.without_mask = 0
        self.incorrect = 0
        self.danger_count = self.without_mask + self.incorrect
        self.all_people = self.with_mask + self.danger_count

        self.lock = Lock()

        # close
        self.finish = QAction("Quit", self)
        self.finish.triggered.connect(self.__del__)

        self.show()

    def debug(self, e):
        error_class = e.__class__.__name__ #取得錯誤類型
        detail = e.args[0] #取得詳細內容
        cl, exc, tb = sys.exc_info() #取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
        fileName = lastCallStack[0] #取得發生的檔案名稱
        lineNum = lastCallStack[1] #取得發生的行號
        funcName = lastCallStack[2] #取得發生的函數名稱
        errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        # print(errMsg)
        rowPosition = self.ui.tableWidget.rowCount()
        self.ui.tableWidget.insertRow(rowPosition)
        error = (error_class, detail, cl, lastCallStack, fileName, lineNum, funcName, errMsg)
        for i in range(8):
            self.ui.tableWidget.setItem(rowPosition, i, QtWidgets.QTableWidgetItem(str(error[i])))
        print(errMsg)

    def setupNet(self, random_seed:int =42):
        self.LABELS = open(self.labelsPath).read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self._ln = self.net.getLayerNames()
        self.ln = [self._ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        np.random.seed(random_seed) 
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),dtype="uint8")

    def setupCamera(self, __video__):
        __temp_cap = None
        if self.cap is not None: __temp_cap = self.cap
        try:
            self.__video__ = __video__
            self.cap = cv2.VideoCapture(self.__video__)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.ui.Input_label.setText(f'目前輸入為: {self.__video__}')
            ret, image = self.cap.read()
            image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
            image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            self.ui.imageshow.setPixmap(QPixmap.fromImage(image))
        except Exception as e: 
            self.debug(e)

    def updateCamera(self):
        while self.cap.isOpened():
            if self.__CameraStop__: return
            ret, frame = self.cap.read()
            if ret == False: continue
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            show_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.imageshow.setPixmap(QPixmap.fromImage(show_image))

    def setFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath() ,'*.mp4')
        self.setupCamera(filename)

    @classmethod
    def phash(self, image, length:int = 32, width:int =32) -> str:
        try: 
            image = cv2.resize(image, (length, width))
            image = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY)
            #左上角區域
            dct = cv2.dct(np.float32(image))[0:8, 0:8]
            avreage = np.mean((dct))
            phash = (dct > avreage) + 0

            #轉換為 1行 -> 轉換為list
            return ''.join([str(x) for x in phash.reshape(1, -1)[0].tolist()])
        except Exception as e:
            self.debug(e)

    @classmethod
    def hamming_distance(self, hash1, hash2) :
        _return = 0
        for idx1, idx2 in zip(hash1, hash2):
            if idx1 != idx2: _return += 1 
        return _return


    def detect(self, *args, **kwargs) -> None:
        try:
            _all_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1 
        except:
            self.setupCamera(self.__Camera__)
            _all_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1 

        self.ui.progressBar.setMaximum(int(_all_frame))
        self.progress_rate = 0

        while 1:
            self.with_mask = 0; self.without_mask = 0; self.incorrect = 0
            if self.__Camerastop__: break
            ret, image = self.cap.read()
            try: H, W = image.shape[:2]
            except Exception as e:
                self.debug(e)

            #-------------Detect------------#
            self.blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
            self.lock.acquire()
            # image scalefactor size mean swapRB crop ddepth
            
            self.net.setInput(self.blob)
            _start_timestemp = time.time()

            layerOutputs = self.net.forward(self.ln)
            boxes = []
            confidences = []
            classIDs = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > self.confidence:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            
            if idxs := cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold):
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in self.COLORS[classIDs[i]]]            
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                    
                    #human face
                    _face = image[y:y+h, x:x+w]
                    # cv2.imwrite(f'{self.__video__}-{len(self.__PHASH__)}.jpg', _face)


                    text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    _label = self.LABELS[classIDs[i]]
                    if _label == 'WithMask' : self.with_mask += 1
                    elif _label == 'WithoutMask' : self.without_mask += 1
                    else : self.incorrect += 1
                    
                    # print("label: ",text, (x, y), (x + w, y + h))
                    
                    if _phash := self.phash(_face):
                        _temp_similarity = -1
                        _tag = 0
                        for tag, save_phash in enumerate(self.__PHASH__):
                            _distance = self.hamming_distance(_phash, save_phash)
                            similarity = 1 - _distance * 1.0 / 64
                            if similarity > _temp_similarity:
                                _tag = tag; _temp_similarity = similarity
                        

                        if (_temp_similarity < self.delete_threshold) or (not self.__PHASH__):
                            self.__PHASH__.append(_phash)
                            _name = f'{self.__video__.split("/")[-1].split(".")[0]}-Detect_{len(self.__PHASH__)}.jpg'
                            self.ui.listWidget.addItem(_name)
                            # cv2.imwrite(self.save+_name, _face)
                            _, _file = cv2.imencode('.jpg', _face)
                            _file.tofile(_name)
                        elif self.DynamicUpdateTracking:
                            self.__PHASH__[_tag] = _phash
                        else : pass
            self.danger_count = self.without_mask + self.incorrect
            self.all_people = self.with_mask + self.danger_count
            if self.danger_count: 
                # print('[Danger Count] There is {:.2f} % person in a danger situation!'
                    # .format(1-self.danger_count/self.all_people)*100)

                image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
                image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                self.ui.imageshow.setPixmap(QPixmap.fromImage(image))

                self.ui.all_people.setText(f"目前總人數: {len(self.__PHASH__)}")
                self.ui.now_people.setText(f"當前畫面總人數: {self.all_people}")
                self.ui.no_mask_rate.setText("未戴口罩比率: {:.2f} %".format((self.danger_count/self.all_people)*100))
            else:
                self.ui.no_mask_rate.setText('未戴口罩比率: None')
            self.ui.progressBar.setValue(self.progress_rate)
            self.progress_rate += 1
            
            _time = time.time() - _start_timestemp 
            fps = 1/_time
            _remaining = (_all_frame - self.progress_rate) / fps
            remaining_h = _remaining // 3600
            remaining_m = _remaining % 3600 // 60
            remaining_s = _remaining % 3600 % 60

            self.ui.remaining_time.setText(f"預計剩餘時間: {int(remaining_h)} 小時 {int(remaining_m)} 分")
            self.ui.FPS.setText(f"FPS: {round(fps, 3)}")
            self.ui.FPS_2.setText(f"剩餘張數: {int(_remaining*fps)} / {int(_all_frame)}")
            self.lock.release()

    def run(self, *args, **kwargs) -> None:
        try:
            _thread = Thread(target = self.detect)
            _thread.start()
        except Exception as e:
            self.debug(e)
    
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print('kill!')

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.setupNet()
    # window.setupCamera('01.mp4')
    
    sys.exit(app.exec_())
    window.__Camerastop__ = True
    del window


#ui setup !

# self.imageshow.setStyleSheet("border-color: rgb(85, 85, 127);")
# self.imageshow.setAlignment(Qt.AlignTop)
# self.imageshow.setAlignment(Qt.AlignLeft)
# self.imageshow.setScaledContents(True)

# self.listWidget.setCurrentRow(0)
# self.listWidget.setViewMode(QListView.ListMode)
# self.listWidget.setSpacing(3)
# self.listWidget.setItemAlignment(QtCore.Qt.AlignCenter)
# self.listWidget.setEnabled(True)