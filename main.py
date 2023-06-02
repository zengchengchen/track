from menu import menu
import sys
from PyQt5.QtWidgets import QApplication


# 运行
def main():
    app = QApplication(sys.argv)
    m = menu()
    m.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
