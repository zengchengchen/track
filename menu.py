import os
import torch
import numpy as np
import cv2

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from PyQt5.QtWidgets import QMainWindow, QFileDialog

from singleTrack import Ui_singleTrack
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer


class menu(QMainWindow, Ui_singleTrack):

    def __init__(self):
        # 初始化
        super(menu, self).__init__()
        self.cap = None
        self.picNow = None
        self.pic = None
        self.tracker = None
        self.startTrackFlag = False
        self.capTimer = None
        self.trackerTimer = None
        self.keyboardAbledFlag = False
        # 创建UI界面
        self.setupUi(self)
        # 视频路径
        self.vPath = ''
        # 按钮初始化
        self.buttonSetAbled(CAREMA=True, VIDEO=True)
        # 按钮动作
        self.setAction()
        # 初始化追踪模型
        self.initTrack()

    def buttonSetAbled(self, CAREMA=False, VIDEO=False, SELECT=False, START=False, END=False):
        # 打开摄像头
        if CAREMA:
            self.openCamera.setEnabled(True)
        else:
            self.openCamera.setEnabled(False)
        # 打开视频
        if VIDEO:
            self.openVideo.setEnabled(True)
        else:
            self.openVideo.setEnabled(False)
        # 选择目标
        if SELECT:
            self.selectObj.setEnabled(True)
        else:
            self.selectObj.setEnabled(False)
        # 开始追踪
        if START:
            self.startTrack.setEnabled(True)
        else:
            self.startTrack.setEnabled(False)
        # 结束追踪
        if END:
            self.endTrack.setEnabled(True)
        else:
            self.endTrack.setEnabled(False)

    # 添加动作
    def setAction(self):
        # 打开相机
        self.openCamera.clicked.connect(self.pressCarema)
        # 打开视频
        self.openVideo.clicked.connect(self.pressVideo)
        # 选择目标
        self.selectObj.clicked.connect(self.pressSelectObj)
        # 开始跟踪
        self.startTrack.clicked.connect(self.pressStartTrack)
        # 结束跟踪
        self.endTrack.clicked.connect(self.pressEndTrack)

    # 载入以及初始化追踪模型
    # 使用pysot目标跟踪库中的SiamRPN算法
    # 使用已经训练好的模型siamrpn_mobilev2_l234_dwxcorr
    def initTrack(self):
        # 配置文件所在路径
        cfgPath = './models/siamrpn_mobilev2_l234_dwxcorr/config.yaml'
        # 预训练模型所在位置 训练权重文件
        shotModelPath = './models/siamrpn_mobilev2_l234_dwxcorr/model.pth'
        # 加载config.yaml配置文件对cfg默认参数进行修改
        cfg.merge_from_file(cfgPath)
        # 系统是否支持CUDA以及CUDA
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        # 初始化设备 要么是CUDA（即GPU显卡）要么是CPU
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        # 加载训练好的模型参数 GPU->CPU 模型是GPU 预加载的类型如果是CPU则需要转换为GPU
        checkpoint = torch.load(shotModelPath, map_location=lambda storage, loc: storage.cpu())
        # 创建模型
        model = ModelBuilder()
        # 加载训练好的模型参数
        model.load_state_dict(checkpoint)
        # 将模型加载到指定设备上
        model.eval().to(device)
        self.processShow.setText('预训练模型加载完成')
        # 追踪器
        self.tracker = build_tracker(model)

    # 打开摄像头
    def pressCarema(self):
        # 清空labelshow的内容
        self.labelShow.clearFlag = False
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        # 定时器定时刷新画面
        self.capTimer = QTimer(self)
        # 显示每帧的图像
        self.capTimer.timeout.connect(self.showImage)
        self.processShow.setText('摄像头运行中')
        self.capTimer.start(40)
        self.buttonSetAbled(SELECT=True, END=True)

    # 打开视频
    def pressVideo(self):
        # 清空labelShow的内容
        self.labelShow.clearFlag = False
        # 获得视频路径
        vPath, vType = QFileDialog.getOpenFileName(self, '选择视频文件', os.path.dirname(__file__), '*.mp4 *.avi')
        # 未读取到视频结束
        if vPath == '':
            return
        self.vPath = vPath
        self.cap = cv2.VideoCapture(self.vPath)
        # 定时器定时刷新画面
        self.capTimer = QTimer(self)
        self.capTimer.timeout.connect(self.showImage)
        # 显示每帧的图像
        self.processShow.setText('视频播放中')
        self.capTimer.start(40)
        self.buttonSetAbled(SELECT=True, END=True)

    # 显示摄像头/视频画面
    def showImage(self):
        # 打开摄像头/视频
        if self.cap.isOpened():
            # 按帧显示
            ret, frame = self.cap.read()
            if ret:
                # 调整每一帧图片的尺寸便于显示
                # 每一帧图像作为全局变量便于用户选择目标时，能够及时获取当前图像
                self.pic = cv2.resize(frame, (self.labelShow.width(), self.labelShow.height()),
                                      interpolation=cv2.INTER_CUBIC)
                # BGR变成RGB
                img = cv2.cvtColor(self.pic, cv2.COLOR_BGR2RGB)
                # pyqt显示图像:->QImage->QPixmap
                img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                self.labelShow.setPixmap(QPixmap.fromImage(img))
        else:
            # 关闭摄像头/视频
            self.capTimer.stop()
            self.cap.release()

    # 选择跟踪目标
    def pressSelectObj(self):
        # 键盘可用
        self.keyboardAbledFlag = True
        if self.cap.isOpened():
            # 当前图片
            self.picNow = self.pic
            # 暂停摄像/播放
            self.capTimer.stop()
        else:
            pass
        self.processShow.setText('框选出跟踪目标: "s"开始框选 "e"结束框选')
        self.buttonSetAbled(START=True, END=True)

    # 重写QWidget中的keyPressEvent
    def keyPressEvent(self, QKeyEvent):
        # 当键盘允许输入时执行
        if self.keyboardAbledFlag:
            # 's'开始框选
            if QKeyEvent.key() == Qt.Key_S:
                # 把鼠标变成十字光标
                self.labelShow.setCursor(Qt.CrossCursor)
                # 使用鼠标
                self.labelShow.useMouseFlag = True
                # 绘制矩形
                self.labelShow.drawRecFlag = True
            # 'e'结束框选
            if QKeyEvent.key() == Qt.Key_E:
                self.processShow.setText('目标选择结束')
                # 取消十字光标
                self.labelShow.unsetCursor()
                # 结束框选后关闭一些功能
                self.labelShow.drawRecFlag = False
                self.labelShow.useMouseFlag = False
                self.keyboardAbledFlag = False

    # 开始追踪
    def pressStartTrack(self):
        if self.keyboardAbledFlag is False:
            # 获得目标矩阵
            targetRec = [self.labelShow.rect.x(), self.labelShow.rect.y(), self.labelShow.rect.width(),
                         self.labelShow.rect.height()]
            # 追踪器的追踪目标初始化
            self.tracker.init(self.picNow, tuple(targetRec))
            # 清空先前所画的目标框
            self.clearLabel()
            # 定时追踪
            self.trackerTimer = QTimer(self)
            self.processShow.setText('开始追踪')
            self.trackerTimer.timeout.connect(self.trackObj)
            self.trackerTimer.start(40)
            self.buttonSetAbled(END=True)

    # 追踪
    def trackObj(self):
        self.startTrackFlag = True
        if self.cap.isOpened():
            # 继续按帧读取
            ret, frame = self.cap.read()
            if ret:
                # 调整图像尺寸
                frame = cv2.resize(frame, (self.labelShow.width(), self.labelShow.height()),
                                   interpolation=cv2.INTER_CUBIC)
                # 获得追踪结果 返回bbox(list):[x, y, width, height]
                result = self.tracker.track(frame)
                # 检测到目标 将分类结果与回归结果融合
                if 'polygon' in result:
                    # 转换数据类型
                    polygon = np.array(result['polygon']).astype(np.int32)
                    # 绘制矩阵
                    # polygon是列表
                    # frame画布矩阵 折线顶点数组 是否为闭合折线 折线颜色 折线粗细
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                    # 预测的mask
                    mask = ((result['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    # 高维数组转化为视图
                    mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                    # 将frame和mask融合 图一 图一的权重 图二 图二的权重 权重和后添加的数值
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                # 直接用bbox
                else:
                    # 矩形
                    bbox = list(map(int, result['bbox']))
                    # bbox(list): [x, y, width, height]
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)
                # 显示图片
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                self.labelShow.setPixmap(QPixmap.fromImage(img))
        else:
            # 关闭
            self.trackerTimer.stop()
            self.cap.release()

    # 停止追踪
    def pressEndTrack(self):
        # 结束追踪就清空画框
        self.clearLabel()
        self.capTimer.stop()
        self.cap.release()
        if self.startTrackFlag:
            self.trackerTimer.stop()
            self.cap.release()
            self.startTrackFlag = False
        self.processShow.clear()
        self.buttonSetAbled(CAREMA=True, VIDEO=True)

    # 清空Label
    def clearLabel(self):
        self.labelShow.clearFlag = True
        self.labelShow.clear()
