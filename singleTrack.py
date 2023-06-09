# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'singleTrack.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QLabel


class Ui_singleTrack(object):
    def setupUi(self, singleTrack):
        singleTrack.setObjectName("singleTrack")
        singleTrack.resize(1000, 800)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(singleTrack.sizePolicy().hasHeightForWidth())
        singleTrack.setSizePolicy(sizePolicy)
        singleTrack.setMinimumSize(QtCore.QSize(1000, 800))
        self.centralwidget = QtWidgets.QWidget(singleTrack)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.widget.setStyleSheet("")
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        # 使用自己编写的Label
        # self.labelShow = QtWidgets.QLabel(self.widget)
        self.labelShow = Label(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelShow.sizePolicy().hasHeightForWidth())
        self.labelShow.setSizePolicy(sizePolicy)
        self.labelShow.setMinimumSize(QtCore.QSize(1000, 600))
        self.labelShow.setSizeIncrement(QtCore.QSize(900, 800))
        self.labelShow.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.labelShow.setText("")
        self.labelShow.setObjectName("labelShow")
        self.gridLayout_2.addWidget(self.labelShow, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.widget, 0, 0, 1, 1)
        self.groupBox_1 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_1.setFont(font)
        self.groupBox_1.setTitle("")
        self.groupBox_1.setObjectName("groupBox_1")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_1)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.processShow = QtWidgets.QTextBrowser(self.groupBox_1)
        self.processShow.setObjectName("processShow")
        self.gridLayout_3.addWidget(self.processShow, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_1, 1, 0, 1, 2)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.openCamera = QtWidgets.QPushButton(self.groupBox_2)
        self.openCamera.setMinimumSize(QtCore.QSize(140, 60))
        self.openCamera.setMaximumSize(QtCore.QSize(140, 60))
        self.openCamera.setObjectName("openCamera")
        self.verticalLayout.addWidget(self.openCamera)
        self.openVideo = QtWidgets.QPushButton(self.groupBox_2)
        self.openVideo.setMinimumSize(QtCore.QSize(140, 60))
        self.openVideo.setMaximumSize(QtCore.QSize(140, 60))
        self.openVideo.setObjectName("openVideo")
        self.verticalLayout.addWidget(self.openVideo)
        self.selectObj = QtWidgets.QPushButton(self.groupBox_2)
        self.selectObj.setMinimumSize(QtCore.QSize(140, 60))
        self.selectObj.setMaximumSize(QtCore.QSize(140, 60))
        self.selectObj.setObjectName("selectObj")
        self.verticalLayout.addWidget(self.selectObj)
        self.startTrack = QtWidgets.QPushButton(self.groupBox_2)
        self.startTrack.setMinimumSize(QtCore.QSize(140, 60))
        self.startTrack.setMaximumSize(QtCore.QSize(140, 60))
        self.startTrack.setObjectName("startTrack")
        self.verticalLayout.addWidget(self.startTrack)
        self.endTrack = QtWidgets.QPushButton(self.groupBox_2)
        self.endTrack.setMinimumSize(QtCore.QSize(140, 60))
        self.endTrack.setMaximumSize(QtCore.QSize(140, 60))
        self.endTrack.setObjectName("endTrack")
        self.verticalLayout.addWidget(self.endTrack)
        self.gridLayout_4.addWidget(self.groupBox_2, 0, 1, 1, 1)
        self.gridLayout_4.setColumnStretch(0, 2)
        singleTrack.setCentralWidget(self.centralwidget)

        self.retranslateUi(singleTrack)
        QtCore.QMetaObject.connectSlotsByName(singleTrack)

    def retranslateUi(self, singleTrack):
        _translate = QtCore.QCoreApplication.translate
        singleTrack.setWindowTitle(_translate("singleTrack", "单目标追踪"))
        self.openCamera.setText(_translate("singleTrack", "打开相机"))
        self.openVideo.setText(_translate("singleTrack", "打开视频"))
        self.selectObj.setText(_translate("singleTrack", "选择目标"))
        self.startTrack.setText(_translate("singleTrack", "开始跟踪"))
        self.endTrack.setText(_translate("singleTrack", "结束跟踪"))


# 重写QLabel类中的一些方法
class Label(QLabel):
    startX = 0
    startY = 0
    endX = 0
    endY = 0
    useMouseFlag = False
    selectObjFlag = False
    drawRecFlag = False
    clearFlag = False
    rect = QRect()

    # 按下鼠标
    def mousePressEvent(self, event):
        if self.useMouseFlag:
            self.selectObjFlag = True
            self.startX = event.x()
            self.startY = event.y()

    # 释放鼠标
    def mouseReleaseEvent(self, event):
        self.selectObjFlag = False

    # 移动鼠标
    def mouseMoveEvent(self, event):
        if self.selectObjFlag:
            self.endX = event.x()
            self.endY = event.y()
            if self.drawRecFlag:
                self.update()

    # 绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 5, Qt.SolidLine))
        if self.clearFlag:
            self.startX = 0
            self.startY = 0
            self.endX = 0
            self.endY = 0
        self.rect = QRect(self.startX, self.startY, abs(self.endX - self.startX), abs(self.endY - self.startY))
        painter.drawRect(self.rect)
        self.update()
