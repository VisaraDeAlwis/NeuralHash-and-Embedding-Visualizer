import sys
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import csv
from pathlib import Path

# Your logic module with model + helper functions
from model_logic import (
    device,
    get_mtcnn,
    get_embedding,
    cosine_similarity,
    load_pca,
    load_hyperplanes,
    neuralhash_from_embedding,
    bits_to_grouped_binary,
)

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QPushButton,
    QSlider,
    QGroupBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtCore import QTimer, Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =============================================================================
# CONFIG: paths for NeuralHash pipeline
# =============================================================================

PCA_PATH = r"D:\FYP\Hybrid_Face_Recognition\neuralhash\assets\pca_512_to_128.pkl"
NEURALHASH_DAT_PATH = (
    r"D:\FYP\Hybrid_Face_Recognition\neuralhash\assets\neuralhash_128x96_seed1.dat"
)


# =============================================================================
# Similarity sparkline widget
# =============================================================================


class SimilarityPlotWidget(QWidget):
    """
    Tiny sparkline plot showing cosine similarity over time
    (e.g., similarity between consecutive embeddings).
    """

    def __init__(self, parent=None, max_points: int = 100):
        super().__init__(parent)
        self.max_points = max_points
        self.values: list[float] = []

        self.fig = Figure(figsize=(2, 2))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self._style_axes(self.ax)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.fig.tight_layout(pad=0.1)

    def _style_axes(self, ax):
        ax.set_facecolor("#000000")
        self.fig.patch.set_facecolor("#000000")
        ax.tick_params(colors="#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#888888")
        ax.title.set_color("#ffffff")
        ax.xaxis.label.set_color("#ffffff")
        ax.yaxis.label.set_color("#ffffff")

    def add_value(self, value: float):
        value = max(0.0, min(1.0, value))  # clamp
        self.values.append(value)
        if len(self.values) > self.max_points:
            self.values = self.values[-self.max_points :]
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self._style_axes(self.ax)
        self.ax.plot(self.values, linewidth=1.5, color="#ff5555")
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_xlim(0, max(len(self.values), 1))
        self.ax.set_title("Cosine Similarity", fontsize=9, color="#ffffff")
        self.ax.set_xticks([])
        self.ax.set_yticks([0.0, 0.5, 1.0])
        self.fig.tight_layout(pad=0.1)
        self.canvas.draw_idle()


# =============================================================================
# NeuralHash visualization widget (12x8 bit grid)
# =============================================================================


class NeuralHashWidget(QWidget):
    """
    Displays the 96-bit NeuralHash as a 12x8 black/white grid.
    """

    def __init__(self, parent=None, figsize=(2.2, 2.2)):
        super().__init__(parent)

        self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self._style_axes(self.ax)

        layout = QVBoxLayout()
        # remove extra margins so the grid occupies almost all area
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.fig.tight_layout(pad=0.1)

    def _style_axes(self, ax):
        ax.set_facecolor("#000000")
        self.fig.patch.set_facecolor("#000000")
        ax.tick_params(colors="#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#888888")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    def set_bits(self, bits: np.ndarray | None, prev_bits: np.ndarray | None = None):
        """
        bits: (96,) array of 0/1
        prev_bits: optional previous hash for red-box highlight
        """
        self.ax.clear()
        self._style_axes(self.ax)

        if bits is None or getattr(bits, "size", 0) == 0:
            self.ax.text(
                0.5,
                0.5,
                "NeuralHash\nnot available",
                ha="center",
                va="center",
                color="#ffffff",
                fontsize=9,
            )
        else:
            try:
                grid = bits.reshape(12, 8)
            except Exception:
                grid = np.zeros((12, 8), dtype=np.uint8)

            self.ax.imshow(grid, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)

            if prev_bits is not None and getattr(prev_bits, "size", 0) == 96:
                try:
                    prev_grid = prev_bits.reshape(12, 8)
                    import matplotlib.patches as patches

                    for r in range(12):
                        for c in range(8):
                            if grid[r, c] != prev_grid[r, c]:
                                rect = patches.Rectangle(
                                    (c - 0.5, r - 0.5),
                                    1,
                                    1,
                                    linewidth=1.2,
                                    edgecolor="#ff5555",
                                    facecolor="none",
                                )
                                self.ax.add_patch(rect)
                except Exception:
                    pass

            self.ax.set_title("NeuralHash 96-bit", color="#ffffff", fontsize=9)

        # Fill entire figure (minimise borders)
        self.ax.set_position([0.0, 0.0, 1.0, 1.0])
        self.fig.tight_layout(pad=0.1)
        self.canvas.draw_idle()


# =============================================================================
# Embedding plots widget (1D + 2D projection)
# =============================================================================


class EmbeddingPlotWidget(QWidget):
    """
    Top: 1D embedding curve (dim index vs value)
    Bottom: 2D scatter of all collected embeddings
           with selectable projection: PCA / t-SNE / Raw-2D
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)

        self.ax1 = self.fig.add_subplot(211)  # 1D embedding curve
        self.ax2 = self.fig.add_subplot(212)  # 2D projection scatter

        self._style_axes(self.ax1)
        self._style_axes(self.ax2)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.fig.tight_layout(pad=0.2)

        self.embeddings_history: list[np.ndarray] = []
        self.projection_method: str = "PCA"

    def _style_axes(self, ax):
        ax.set_facecolor("#000000")
        self.fig.patch.set_facecolor("#000000")
        ax.tick_params(colors="#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#888888")
        ax.title.set_color("#ffffff")
        ax.xaxis.label.set_color("#ffffff")
        ax.yaxis.label.set_color("#ffffff")

    def set_projection_method(self, method: str):
        self.projection_method = method
        if self.embeddings_history:
            self._update_plots(self.embeddings_history[-1])

    def add_embedding(self, embedding: np.ndarray):
        self.embeddings_history.append(embedding.copy())
        self._update_plots(embedding)

    def _update_plots(self, latest_embedding: np.ndarray):
        # 1D curve
        self.ax1.clear()
        self._style_axes(self.ax1)
        self.ax1.plot(
            np.arange(len(latest_embedding)),
            latest_embedding,
            color="#ff5555",
            linewidth=1.2,
        )
        self.ax1.set_title("Latest Face Embedding (512D)", color="#ffffff", fontsize=11)
        self.ax1.set_xlabel("Dimension index", color="#ffffff")
        self.ax1.set_ylabel("Value", color="#ffffff")

        # 2D projection
        self.ax2.clear()
        self._style_axes(self.ax2)

        n = len(self.embeddings_history)
        if n >= 2:
            X = np.stack(self.embeddings_history, axis=0)

            method = self.projection_method
            try:
                if method == "PCA":
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X)
                    title = f"PCA of Embeddings (N={n})"
                elif method == "t-SNE":
                    max_pts = min(n, 200)
                    X_sub = X[-max_pts:]
                    perplexity = min(30, max_pts - 1) if max_pts > 1 else 1
                    tsne = TSNE(
                        n_components=2,
                        perplexity=perplexity,
                        learning_rate="auto",
                        init="random",
                        random_state=42,
                    )
                    X_2d = tsne.fit_transform(X_sub)
                    title = f"t-SNE of Last {max_pts} Embeddings"
                elif method == "Raw-2D":
                    X_2d = X[:, :2]
                    title = f"Raw 2D (Dims 0 & 1) (N={n})"
                else:
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X)
                    title = f"PCA of Embeddings (N={n}) [fallback]"

                self.ax2.scatter(X_2d[:, 0], X_2d[:, 1], s=20, c="#00d1ff")
                self.ax2.set_title(title, color="#ffffff", fontsize=11)
                self.ax2.set_xlabel("Component 1", color="#ffffff")
                self.ax2.set_ylabel("Component 2", color="#ffffff")
            except Exception as e:
                self.ax2.text(
                    0.5,
                    0.5,
                    f"Projection error:\n{e}",
                    ha="center",
                    va="center",
                    transform=self.ax2.transAxes,
                    color="#ff5555",
                )
                self.ax2.set_title(f"{method} error", color="#ffffff")
        else:
            self.ax2.text(
                0.5,
                0.5,
                "Need at least 2 embeddings\nfor 2D projection",
                ha="center",
                va="center",
                transform=self.ax2.transAxes,
                color="#ffffff",
            )
            self.ax2.set_title("2D Projection", color="#ffffff")

        self.fig.tight_layout(pad=0.2)
        self.canvas.draw_idle()


# =============================================================================
# Main PyQt window
# =============================================================================


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Embedding + NeuralHash Demonstration")

        # Global dark stylesheet
        self.setStyleSheet(
            """
        QMainWindow {
            background-color: #121212;
        }
        QWidget {
            background-color: #121212;
            color: #f0f0f0;
            font-family: Segoe UI, Arial;
            font-size: 10pt;
        }
        QGroupBox {
            border: 1px solid #444444;
            margin-top: 4px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 6px;
            padding: 0 3px 0 3px;
            color: #dddddd;
            font-size: 9pt;
        }
        QPushButton {
            background-color: #333333;
            border: 1px solid #555555;
            border-radius: 5px;
            padding: 3px 6px;
            color: #f0f0f0;
            font-size: 9pt;
            min-height: 18px;
        }
        QPushButton:hover {
            background-color: #555555;
        }
        QPushButton:pressed {
            background-color: #777777;
        }
        QSlider::groove:horizontal {
            background: #333333;
            height: 6px;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #ff5555;
            width: 12px;
            margin: -4px 0;
            border-radius: 6px;
        }
        QComboBox {
            background-color: #333333;
            color: #f0f0f0;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 3px;
            font-size: 9pt;
        }
        QComboBox QAbstractItemView {
            background-color: #222222;
            color: #f0f0f0;
            selection-background-color: #444444;
        }
        """
        )

        # --- state ---
        self.is_capturing = False
        self.is_frozen = False
        self.last_frame_bgr: np.ndarray | None = None
        self.detection_threshold = 0.90
        self.last_embedding: np.ndarray | None = None

        # NeuralHash state
        self.pca_model = None
        self.hyperplanes = None
        self.last_hash_bits: np.ndarray | None = None
        self.hash_history: list[str] = []
        self.hamming_threshold = 20
        self.hash_history_list = QListWidget()

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # MTCNN
        self.mtcnn = get_mtcnn(keep_all=True)

        # Logging files
        self.log_path = Path("embeddings_log.csv")
        self._init_log_file()
        self.hash_log_path = Path("neuralhash_log.csv")
        self._init_hash_log_file()

        # --- UI COMPONENTS -------------------------------------------------

        # Camera view
        self.video_label = QLabel("Camera")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_label.setStyleSheet("background-color: #000000;")

        # Cropped face
        self.cropped_label = QLabel("Cropped Face")
        self.cropped_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cropped_label.setFixedSize(150, 150)
        self.cropped_label.setStyleSheet("background-color: #000000;")

        # Cosine similarity (shortened height)
        self.similarity_widget = SimilarityPlotWidget()
        self.similarity_widget.setMinimumSize(100, 80)
        self.similarity_widget.setMaximumHeight(140)

        # Controls (NOW will live in right-bottom, but create here)
        controls_widget = self._create_controls_panel()

        # ---- LEFT layout: camera + (crop+similarity) ----
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)

        # top: camera
        left_layout.addWidget(self.video_label)

        # bottom: cropped + similarity
        bottom_left_layout = QHBoxLayout()
        bottom_left_layout.addWidget(self.cropped_label, stretch=1)
        bottom_left_layout.addWidget(self.similarity_widget, stretch=2)
        left_layout.addLayout(bottom_left_layout)

        # only two rows now
        left_layout.setStretch(0, 4)  # camera
        left_layout.setStretch(1, 3)  # crop + similarity

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # ---- RIGHT: embeddings (top 50%) + NeuralHash (middle) + controls (bottom) ----

        self.embedding_plot = EmbeddingPlotWidget()
        self.embedding_plot.setMinimumHeight(280)
        self.embedding_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        # NeuralHash block
        self.neuralhash_widget = NeuralHashWidget(figsize=(3, 3))
        self.neuralhash_widget.setMinimumHeight(180)
        self.neuralhash_widget.setMaximumHeight(230)
        self.neuralhash_widget.setMaximumWidth(200)

        self.neuralhash_label = QLabel("NeuralHash (binary): N/A")
        self.neuralhash_label.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size: 8pt;"
        )
        self.neuralhash_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.hash_history_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.hash_history_list.setMaximumHeight(140)
        self.hash_history_list.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size: 9pt;"
        )
        self.hash_history_list.itemDoubleClicked.connect(
            self._on_history_item_double_clicked
        )
        self.hash_history_list.itemClicked.connect(self._on_history_item_clicked)

        nh_group = QGroupBox("NeuralHash")
        nh_layout = QVBoxLayout()
        nh_top_row = QHBoxLayout()
        nh_top_row.addWidget(self.neuralhash_widget, stretch=1)
        nh_top_row.addWidget(self.hash_history_list, stretch=2)
        nh_layout.addLayout(nh_top_row)
        nh_layout.addWidget(self.neuralhash_label)
        nh_group.setLayout(nh_layout)

        # Right vertical layout:
        right_layout = QVBoxLayout()
        # embedding plots: ~50% of right column
        right_layout.addWidget(self.embedding_plot, stretch=5)
        # bottom half: NeuralHash (top part of bottom) + controls (bottom part)
        right_layout.addWidget(nh_group, stretch=3)
        right_layout.addWidget(controls_widget, stretch=2)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # ---- MAIN layout: left vs right ----
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        # Swap the order to show the embedding / plots on the left and the camera on the right.
        # Keep the same width proportions by reversing the widget order but preserving the stretch factors.
        main_layout.addWidget(right_widget, 7)
        main_layout.addWidget(left_widget, 3)
        main_widget.setLayout(main_layout)

        self.setCentralWidget(main_widget)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ------------------------------------------------------------------ #
    # Controls panel                                                     #
    # ------------------------------------------------------------------ #

    def _create_controls_panel(self) -> QWidget:
        group = QGroupBox("Controls")
        layout = QVBoxLayout()

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_freeze = QPushButton("Freeze")
        self.btn_unfreeze = QPushButton("Unfreeze")

        self.btn_start.clicked.connect(self.start_capture)
        self.btn_stop.clicked.connect(self.stop_capture)
        self.btn_freeze.clicked.connect(self.freeze_view)
        self.btn_unfreeze.clicked.connect(self.unfreeze_view)

        row1 = QHBoxLayout()
        row1.addWidget(self.btn_start)
        row1.addWidget(self.btn_stop)

        row2 = QHBoxLayout()
        row2.addWidget(self.btn_freeze)
        row2.addWidget(self.btn_unfreeze)

        self.threshold_label = QLabel("Detection Threshold: 0.90")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(99)
        self.threshold_slider.setValue(90)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)

        self.proj_label = QLabel("Projection: PCA")
        self.proj_combo = QComboBox()
        self.proj_combo.addItems(["PCA", "t-SNE", "Raw-2D"])
        self.proj_combo.currentTextChanged.connect(self._on_projection_changed)

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addSpacing(6)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_slider)
        layout.addSpacing(6)
        layout.addWidget(self.proj_label)
        layout.addWidget(self.proj_combo)

        # Hamming threshold slider
        self.hamming_threshold = 12
        self.hamming_label = QLabel(f"Hamming threshold: {self.hamming_threshold}")
        self.hamming_slider = QSlider(Qt.Orientation.Horizontal)
        self.hamming_slider.setMinimum(0)
        self.hamming_slider.setMaximum(96)
        self.hamming_slider.setValue(self.hamming_threshold)
        self.hamming_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.hamming_slider.setTickInterval(4)
        self.hamming_slider.valueChanged.connect(self._on_hamming_threshold_changed)

        layout.addSpacing(8)
        layout.addWidget(self.hamming_label)
        layout.addWidget(self.hamming_slider)

        self.btn_copy_hash = QPushButton("Copy Last Hash")
        self.btn_copy_hash.clicked.connect(self._copy_last_hash)
        layout.addWidget(self.btn_copy_hash)

        self.btn_clear_hash_history = QPushButton("Clear Hash History")
        self.btn_clear_hash_history.clicked.connect(self._clear_hash_history)
        layout.addWidget(self.btn_clear_hash_history)

        group.setLayout(layout)
        return group

    # simple control handlers
    def start_capture(self):
        self.is_capturing = True
        print("Capture started")

    def stop_capture(self):
        self.is_capturing = False
        print("Capture stopped")

    def freeze_view(self):
        self.is_frozen = True
        print("View frozen")

    def unfreeze_view(self):
        self.is_frozen = False
        print("View unfrozen")

    def _on_threshold_changed(self, value: int):
        self.detection_threshold = value / 100.0
        self.threshold_label.setText(
            f"Detection Threshold: {self.detection_threshold:.2f}"
        )

    def _on_projection_changed(self, text: str):
        self.proj_label.setText(f"Projection: {text}")
        self.embedding_plot.set_projection_method(text)

    # ------------------------------------------------------------------ #
    # Logging embeddings                                                 #
    # ------------------------------------------------------------------ #

    def _init_log_file(self):
        if not self.log_path.exists():
            with self.log_path.open("w", newline="") as f:
                writer = csv.writer(f)
                header = ["timestamp"] + [f"dim_{i}" for i in range(512)]
                writer.writerow(header)
            print(f"Created log file: {self.log_path}")
        else:
            print(f"Appending to existing log file: {self.log_path}")

    def _log_embedding(self, emb: np.ndarray):
        ts = datetime.now().isoformat()
        row = [ts] + emb.tolist()
        with self.log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ------------------------------------------------------------------ #
    # NeuralHash log (CSV + UI)                                          #
    # ------------------------------------------------------------------ #

    def _init_hash_log_file(self):
        if not self.hash_log_path.exists():
            with self.hash_log_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "binary_hash", "hamming_to_prev"])
            print(f"Created NeuralHash log: {self.hash_log_path}")
        else:
            print(f"Appending to existing NeuralHash log: {self.hash_log_path}")

        self.hash_history_list.clear()
        try:
            with self.hash_log_path.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = list(reader)
                for row in rows[-20:]:
                    ts, binary_str, hamming_field = row
                    try:
                        hamming_val = (
                            int(hamming_field) if hamming_field != "" else None
                        )
                    except Exception:
                        hamming_val = None
                    entry = (
                        f"{ts} | d_prev={hamming_val if hamming_val is not None else 'N/A':>3} | "
                        f"{binary_str}"
                    )
                    item = QListWidgetItem(entry)
                    item.setToolTip(binary_str)
                    if hamming_val is None:
                        color = QColor("#ffffff")
                    else:
                        if hamming_val <= self.hamming_threshold:
                            color = QColor("#00c853")
                        elif hamming_val <= max(1, self.hamming_threshold * 2):
                            color = QColor("#ff9800")
                        else:
                            color = QColor("#ff1744")
                    item.setForeground(color)
                    self.hash_history_list.addItem(item)
                while self.hash_history_list.count() > 20:
                    self.hash_history_list.takeItem(self.hash_history_list.count() - 1)
        except Exception:
            pass

    def _log_neuralhash(self, binary_str: str, hamming: int | None):
        ts = datetime.now().isoformat()
        with self.hash_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, binary_str, hamming if hamming is not None else ""])

        entry = (
            f"{ts} | d_prev={hamming if hamming is not None else 'N/A':>3} | "
            f"{binary_str}"
        )
        self.hash_history.append(entry)
        if len(self.hash_history) > 20:
            self.hash_history = self.hash_history[-20:]

        item = QListWidgetItem(entry)
        item.setToolTip(binary_str)
        if hamming is None:
            color = QColor("#ffffff")
        else:
            if hamming <= self.hamming_threshold:
                color = QColor("#00c853")
            elif hamming <= max(1, self.hamming_threshold * 2):
                color = QColor("#ff9800")
            else:
                color = QColor("#ff1744")
        item.setForeground(color)
        self.hash_history_list.insertItem(0, item)
        while self.hash_history_list.count() > 20:
            self.hash_history_list.takeItem(self.hash_history_list.count() - 1)

    def _on_history_item_double_clicked(self, item: QListWidgetItem):
        QApplication.clipboard().setText(item.text())
        print("Copied history entry to clipboard:", item.text())

    def _on_history_item_clicked(self, item: QListWidgetItem):
        text = item.text()
        try:
            parts = text.split("|")
            binary_str = parts[2].strip()
            raw = binary_str.replace(" ", "")
            bits = np.array([int(b) for b in raw.strip()])
            if self.last_hash_bits is not None:
                self.neuralhash_widget.set_bits(self.last_hash_bits, prev_bits=bits)
            else:
                self.neuralhash_widget.set_bits(bits, prev_bits=None)
        except Exception as e:
            print("Error parsing selected hash:", e)

    def _on_hamming_threshold_changed(self, value: int):
        self.hamming_threshold = int(value)
        self.hamming_label.setText(f"Hamming threshold: {self.hamming_threshold}")
        for idx in range(self.hash_history_list.count()):
            item = self.hash_history_list.item(idx)
            try:
                parts = item.text().split("|")
                dp = parts[1].strip()
                h = dp.split("=")[1].strip()
                hamming = None if h == "N/A" or h == "" else int(h)
            except Exception:
                hamming = None
            if hamming is None:
                color = QColor("#ffffff")
            else:
                if hamming <= self.hamming_threshold:
                    color = QColor("#00c853")
                elif hamming <= max(1, self.hamming_threshold * 2):
                    color = QColor("#ff9800")
                else:
                    color = QColor("#ff1744")
            item.setForeground(color)

    def _copy_last_hash(self):
        if self.hash_history_list.count() == 0:
            return
        item = self.hash_history_list.item(0)
        if item:
            QApplication.clipboard().setText(item.text())
            print("Copied last hash entry to clipboard")

    def _clear_hash_history(self):
        try:
            self.hash_history_list.clear()
            with self.hash_log_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "binary_hash", "hamming_to_prev"])
            self.hash_history = []
            print("Cleared neuralhash history and file")
        except Exception as e:
            print("Clear history error:", e)

    # ------------------------------------------------------------------ #
    # NeuralHash computation / display                                   #
    # ------------------------------------------------------------------ #

    def _ensure_neuralhash_models(self):
        if self.pca_model is None or self.hyperplanes is None:
            try:
                print("Loading NeuralHash PCA + hyperplanes...")
                self.pca_model = load_pca(PCA_PATH)
                self.hyperplanes = load_hyperplanes(NEURALHASH_DAT_PATH)
                print("NeuralHash models loaded.")
            except Exception as e:
                print("Error loading NeuralHash models:", e)
                self.pca_model = None
                self.hyperplanes = None

    def _update_neuralhash_display(self, emb512: np.ndarray):
        self._ensure_neuralhash_models()
        if self.pca_model is None or self.hyperplanes is None:
            self.neuralhash_widget.set_bits(None)
            self.neuralhash_label.setText("NeuralHash (binary): models not loaded")
            return

        try:
            bits = neuralhash_from_embedding(emb512, self.pca_model, self.hyperplanes)

            prev_bits = self.last_hash_bits
            hamming = None
            if prev_bits is not None and bits.shape == prev_bits.shape:
                hamming = int(np.sum(bits != prev_bits))

            self.neuralhash_widget.set_bits(bits, prev_bits=prev_bits)
            self.last_hash_bits = bits

            binary_str = bits_to_grouped_binary(bits, group_size=8)

            if hamming is None:
                self.neuralhash_label.setStyleSheet(
                    "color: #ffffff; background: transparent; "
                    "font-family: Consolas, 'Courier New', monospace; font-size: 8pt;"
                )
            else:
                if hamming <= self.hamming_threshold:
                    bg = "#c8facc"
                    fg = "#000000"
                elif hamming <= max(1, self.hamming_threshold * 2):
                    bg = "#fff3cd"
                    fg = "#000000"
                else:
                    bg = "#ffd3d3"
                    fg = "#000000"
                self.neuralhash_label.setStyleSheet(
                    f"background-color: {bg}; color: {fg}; "
                    "font-family: Consolas, 'Courier New', monospace; font-size: 8pt; padding: 4px;"
                )

            self.neuralhash_label.setText(f"NeuralHash (binary): {binary_str}")
            self._log_neuralhash(binary_str, hamming)
        except Exception as e:
            print("NeuralHash computation error:", e)
            self.neuralhash_widget.set_bits(None)
            self.neuralhash_label.setText("NeuralHash (binary): error")

    # ------------------------------------------------------------------ #
    # Frame loop                                                         #
    # ------------------------------------------------------------------ #

    def update_frame(self):
        if self.is_frozen and self.last_frame_bgr is not None:
            self._display_main_frame(self.last_frame_bgr)
            return

        ret, frame_bgr = self.cap.read()
        if not ret:
            return

        self.last_frame_bgr = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn.detect(frame_rgb)
        face_crop = None

        if boxes is not None and len(boxes) > 0:
            boxes = np.array(boxes)
            probs = np.array(probs)

            valid_indices = [
                i
                for i, p in enumerate(probs)
                if p is not None and p >= self.detection_threshold
            ]

            if valid_indices:
                valid_boxes = boxes[valid_indices]
                areas = (valid_boxes[:, 2] - valid_boxes[:, 0]) * (
                    valid_boxes[:, 3] - valid_boxes[:, 1]
                )
                best_local_idx = int(np.argmax(areas))
                best_idx = valid_indices[best_local_idx]

                x1, y1, x2, y2 = boxes[best_idx].astype(int)

                h, w, _ = frame_rgb.shape
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 > x1 and y2 > y1:
                    face_crop = frame_rgb[y1:y2, x1:x2, :]

                    if self.is_capturing:
                        try:
                            emb = get_embedding(face_crop)

                            if self.last_embedding is not None:
                                sim = cosine_similarity(self.last_embedding, emb)
                                self.similarity_widget.add_value(sim)
                            self.last_embedding = emb

                            self.embedding_plot.add_embedding(emb)
                            self._log_embedding(emb)
                            self._update_neuralhash_display(emb)
                        except Exception as e:
                            print("Embedding error:", e)

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if face_crop is not None:
            self._update_cropped_face(face_crop)

        self._display_main_frame(frame_bgr)

    def _display_main_frame(self, frame_bgr: np.ndarray):
        display_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        display_rgb = np.ascontiguousarray(display_rgb, dtype=np.uint8)
        h, w, ch = display_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            display_rgb.tobytes(),
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.width() if self.video_label.width() > 0 else w,
                self.video_label.height() if self.video_label.height() > 0 else h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_cropped_face(self, face_rgb: np.ndarray):
        face_rgb = np.ascontiguousarray(face_rgb, dtype=np.uint8)
        h, w, ch = face_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            face_rgb.tobytes(),
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.cropped_label.width(),
            self.cropped_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.cropped_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


# =============================================================================
# Entry point
# =============================================================================


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1920, 1080)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
