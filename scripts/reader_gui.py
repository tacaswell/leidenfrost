import sys
import leidenfrost.gui as lfg
import PySide.QtGui as QtGui


app = QtGui.QApplication(sys.argv)
lfg.LFReaderGui()
sys.exit()
