import sys
import leidenfrost.gui as lfg
import PySide6.QtWidgets as QtWidgets


app = QtWidgets.QApplication(sys.argv)
lfg.LFGui()
app.exec_()
sys.exit()
