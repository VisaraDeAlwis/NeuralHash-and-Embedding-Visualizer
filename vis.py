import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QGridLayout, QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class MetricCard(QFrame):
    def __init__(self, title, color):
        super().__init__()
        self.value = 0

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 14px;
            }}
            QLabel {{
                color: white;
            }}
            QPushButton {{
                background-color: rgba(255,255,255,0.2);
                color: white;
                border-radius: 6px;
                padding: 4px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: rgba(255,255,255,0.35);
            }}
        """)

        layout = QVBoxLayout()

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 13))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.value_label = QLabel("0")
        self.value_label.setFont(QFont("Segoe UI", 30, QFont.Weight.Bold))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_layout = QHBoxLayout()
        plus_btn = QPushButton("+")
        minus_btn = QPushButton("−")

        plus_btn.setFixedSize(28, 28)
        minus_btn.setFixedSize(28, 28)

        plus_btn.clicked.connect(self.increment)
        minus_btn.clicked.connect(self.decrement)

        btn_layout.addWidget(minus_btn)
        btn_layout.addWidget(plus_btn)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def increment(self):
        self.value += 1
        self.value_label.setText(str(self.value))

    def decrement(self):
        if self.value > 0:
            self.value -= 1
            self.value_label.setText(str(self.value))


class PerformanceDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralHash – Live Model Evaluation")
        self.setMinimumSize(900, 550)
        self.setStyleSheet("background-color: #12121c;")

        main_layout = QVBoxLayout()

        title = QLabel("NeuralHash Face Recognition – Rextro 3rd Day Dashboard")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # ===== Metric Cards Grid =====
        grid = QGridLayout()
        grid.setSpacing(22)

        self.tp = MetricCard("Criminal is Detected as Criminal", "#2ecc71")   # green
        self.fp = MetricCard("Innocent Detected as Criminal", "#e74c3c") # red
        self.tn = MetricCard("Innocent Detected as Innocent", "#27ae60")   # green
        self.fn = MetricCard("Criminal Detected as Innocent", "#f39c12") # orange

        grid.addWidget(self.tp, 0, 0)
        grid.addWidget(self.fp, 0, 1)
        grid.addWidget(self.tn, 1, 0)
        grid.addWidget(self.fn, 1, 1)

        main_layout.addLayout(grid)

        # ===== Metrics Output =====
        self.metrics_label = QLabel()
        self.metrics_label.setFont(QFont("Segoe UI", 14))
        self.metrics_label.setStyleSheet("color: #a9a9ff;")
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.metrics_label)

        update_btn = QPushButton("Update Performance Metrics")
        update_btn.setFont(QFont("Segoe UI", 14))
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #00b894;
                color: white;
                border-radius: 12px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #00997a;
            }
        """)
        update_btn.clicked.connect(self.update_metrics)
        main_layout.addWidget(update_btn)

        self.setLayout(main_layout)

    def update_metrics(self):
        TP, FP, TN, FN = (
            self.tp.value,
            self.fp.value,
            self.tn.value,
            self.fn.value,
        )

        accuracy = (TP + TN) / max((TP + TN + FP + FN), 1)
        precision = TP / max((TP + FP), 1)
        recall = TP / max((TP + FN), 1)
        f1 = (2 * precision * recall) / max((precision + recall), 1)

        self.metrics_label.setText(
            f"Accuracy: {accuracy:.3f}   |   "
            f"Precision: {precision:.3f}   |   "
            f"Recall: {recall:.3f}   |   "
            f"F1 Score: {f1:.3f}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PerformanceDashboard()
    window.show()
    sys.exit(app.exec())
