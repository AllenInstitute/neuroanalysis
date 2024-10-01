import time


class UserTestUi(object):
    def __init__(self, expected_display, current_display):
        import pyqtgraph as pg

        pg.mkQApp()

        self.widget = pg.QtWidgets.QSplitter(pg.QtCore.Qt.Vertical)
        self.widget.resize(1600, 1000)

        self.display_splitter = pg.QtWidgets.QSplitter(pg.QtCore.Qt.Horizontal)
        self.widget.addWidget(self.display_splitter)
        
        self.display1 = expected_display
        self.display2 = current_display
        self.display_splitter.addWidget(self.display1.widget)
        self.display_splitter.addWidget(self.display2.widget)

        self.ctrl = pg.QtWidgets.QWidget()
        self.widget.addWidget(self.ctrl)
        self.ctrl_layout = pg.QtWidgets.QVBoxLayout()
        self.ctrl.setLayout(self.ctrl_layout)
        self.diff_widget = pg.DiffTreeWidget()
        self.ctrl_layout.addWidget(self.diff_widget)

        self.pass_btn = pg.QtWidgets.QPushButton('pass')
        self.fail_btn = pg.QtWidgets.QPushButton('fail')
        self.ctrl_layout.addWidget(self.pass_btn)
        self.ctrl_layout.addWidget(self.fail_btn)

        self.pass_btn.clicked.connect(self.pass_clicked)
        self.fail_btn.clicked.connect(self.fail_clicked)

        self.last_btn_clicked = None
        self.widget.setSizes([750, 250])

    def pass_clicked(self):
        self.last_btn_clicked = 'pass'

    def fail_clicked(self):
        self.last_btn_clicked = 'fail'

    def user_passfail(self):
        import pyqtgraph as pg

        self.widget.show()
        while True:
            pg.QtWidgets.QApplication.processEvents()
            last_btn_clicked = self.last_btn_clicked
            self.last_btn_clicked = None

            if last_btn_clicked == 'fail' or not self.widget.isVisible():
                raise Exception("User rejected test result.")
            elif last_btn_clicked == 'pass':
                break
            
            time.sleep(0.03)

    def show_results(self, expected, current):
        self.diff_widget.setData(expected, current)
        self.display2.show_result(current)
        self.display1.show_result(expected)

    def clear(self):
        self.display1.clear()
        self.display2.clear()
        self.diff_widget.setData(None, None)
