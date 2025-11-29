import sys
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from datetime import datetime
import csv
from pathlib import Path

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
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------- Face embedding model ---------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def get_embedding(rgb_image: np.ndarray) -> np.ndarray:
    """
    Takes aligned RGB numpy image (H,W,3),
    returns a 512D embedding as float32 numpy array.
    """
    import torchvision.transforms as transforms

    if rgb_image.dtype != np.uint8:
        rgb_image = (
            (rgb_image * 255).astype(np.uint8)
            if rgb_image.max() <= 1.0
            else rgb_image.astype(np.uint8)
        )

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got shape {rgb_image.shape}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # (C,H,W) in [0,1]
            transforms.Resize((160, 160)),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),  # -> [-1,1] for each channel
        ]
    )

    pil_img = Image.fromarray(rgb_image)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = resnet(img_tensor).cpu().numpy().flatten()

    return emb.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D numpy arrays."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------- Sparkline widget for cosine similarity ---------------- #


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

        # Dark theme for this figure
        self._style_axes(self.ax)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.fig.tight_layout()

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
        # Clamp to [0,1] for safety
        value = max(0.0, min(1.0, value))
        self.values.append(value)
        if len(self.values) > self.max_points:
            self.values = self.values[-self.max_points:]
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
        self.fig.tight_layout()
        self.canvas.draw_idle()


# ---------------- Matplotlib widget for embedding plots ---------------- #


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
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.fig.tight_layout()

        # Keep history for projections
        self.embeddings_history: list[np.ndarray] = []
        self.projection_method: str = "PCA"  # default

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
        """Change projection method and refresh plot if we have data."""
        self.projection_method = method
        if self.embeddings_history:
            self._update_plots(self.embeddings_history[-1])

    def add_embedding(self, embedding: np.ndarray):
        """Store embedding and update plots."""
        self.embeddings_history.append(embedding.copy())
        self._update_plots(embedding)

    def _update_plots(self, latest_embedding: np.ndarray):
        # --- 1D curve of latest embedding ---
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

        # --- 2D projection of all embeddings ---
        self.ax2.clear()
        self._style_axes(self.ax2)

        n = len(self.embeddings_history)
        if n >= 2:
            X = np.stack(self.embeddings_history, axis=0)  # (N,512)

            method = self.projection_method

            try:
                if method == "PCA":
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X)
                    title = f"PCA of Embeddings (N={n})"

                elif method == "t-SNE":
                    # t-SNE is expensive; limit to last up to 200 points
                    max_pts = min(n, 200)
                    X_sub = X[-max_pts:]
                    # t-SNE needs perplexity < N
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
                    # Just use first 2 dimensions directly
                    X_2d = X[:, :2]
                    title = f"Raw 2D (Dims 0 & 1) (N={n})"

                else:
                    # Fallback to PCA if unknown
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X)
                    title = f"PCA of Embeddings (N={n}) [fallback]"

                self.ax2.scatter(X_2d[:, 0], X_2d[:, 1], s=20, c="#00d1ff")
                self.ax2.set_title(title, color="#ffffff", fontsize=11)
                self.ax2.set_xlabel("Component 1", color="#ffffff")
                self.ax2.set_ylabel("Component 2", color="#ffffff")

            except Exception as e:
                # In case t-SNE or PCA fails for any numerical reason
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

        self.fig.tight_layout()
        self.canvas.draw_idle()


# ---------------- Main PyQt6 window ---------------- #


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Embedding Demo (PyQt6 + MTCNN + InceptionResnetV1)")

        # Global dark stylesheet for UI widgets
        self.setStyleSheet("""
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
            margin-top: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px 0 4px;
            color: #dddddd;
        }
        QPushButton {
            background-color: #333333;
            border: 1px solid #555555;
            border-radius: 6px;
            padding: 6px;
            color: #f0f0f0;
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
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QComboBox {
            background-color: #333333;
            color: #f0f0f0;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 4px;
        }
        QComboBox QAbstractItemView {
            background-color: #222222;
            color: #f0f0f0;
            selection-background-color: #444444;
        }
        """)

        # State flags
        self.is_capturing = False      # controls embedding + logging + plots
        self.is_frozen = False         # controls whether we read new frames
        self.last_frame_bgr = None     # for freeze view
        self.detection_threshold = 0.90  # default
        self.last_embedding: np.ndarray | None = None  # for cosine similarity

        # Camera capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # MTCNN for face detection (keep_all=True -> we choose largest face)
        self.mtcnn = MTCNN(keep_all=True, device=device)

        # Embedding log file
        self.log_path = Path("embeddings_log.csv")
        self._init_log_file()

        # ---- UI Components ---- #

        # Main camera view
        self.video_label = QLabel("Camera")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.video_label.setStyleSheet("background-color: #000000;")

        # Cropped face preview
        self.cropped_label = QLabel("Cropped Face")
        self.cropped_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cropped_label.setFixedSize(200, 200)  # smaller preview
        self.cropped_label.setStyleSheet("background-color: #000000;")

        # Cosine similarity sparkline widget
        self.similarity_widget = SimilarityPlotWidget()
        self.similarity_widget.setMinimumSize(200, 200)

        # Left side: camera + (cropped face + sparkline)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label, stretch=3)

        bottom_left_layout = QHBoxLayout()
        bottom_left_layout.addWidget(self.cropped_label, stretch=1)
        bottom_left_layout.addWidget(self.similarity_widget, stretch=1)

        left_layout.addLayout(bottom_left_layout, stretch=1)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # Embedding plots
        self.embedding_plot = EmbeddingPlotWidget()

        # Control panel
        controls_widget = self._create_controls_panel()

        # Right side: embedding plots + controls
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.embedding_plot, stretch=4)
        right_layout.addWidget(controls_widget, stretch=1)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget, 2)
        main_layout.addWidget(right_widget, 3)
        main_widget.setLayout(main_layout)

        self.setCentralWidget(main_widget)

        # Timer to grab frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 fps

    # ---------- Controls panel ---------- #

    def _create_controls_panel(self) -> QWidget:
        group = QGroupBox("Controls")
        layout = QVBoxLayout()

        # Start / Stop capture buttons
        self.btn_start = QPushButton("Start Capture")
        self.btn_stop = QPushButton("Stop Capture")
        self.btn_start.clicked.connect(self.start_capture)
        self.btn_stop.clicked.connect(self.stop_capture)

        # Freeze / Unfreeze buttons
        self.btn_freeze = QPushButton("Freeze View")
        self.btn_unfreeze = QPushButton("Unfreeze View")
        self.btn_freeze.clicked.connect(self.freeze_view)
        self.btn_unfreeze.clicked.connect(self.unfreeze_view)

        # Detection threshold slider (0.50 to 0.99)
        self.threshold_label = QLabel("Detection Threshold: 0.90")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(50)   # 0.50
        self.threshold_slider.setMaximum(99)   # 0.99
        self.threshold_slider.setValue(90)     # 0.90
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)

        # Projection method selector
        self.proj_label = QLabel("Projection: PCA")
        self.proj_combo = QComboBox()
        self.proj_combo.addItems(["PCA", "t-SNE", "Raw-2D"])
        self.proj_combo.currentTextChanged.connect(self._on_projection_changed)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addSpacing(10)
        layout.addWidget(self.btn_freeze)
        layout.addWidget(self.btn_unfreeze)
        layout.addSpacing(15)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_slider)
        layout.addSpacing(15)
        layout.addWidget(self.proj_label)
        layout.addWidget(self.proj_combo)

        layout.addStretch(1)
        group.setLayout(layout)
        return group

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
        self.threshold_label.setText(f"Detection Threshold: {self.detection_threshold:.2f}")
        print("Detection threshold set to", self.detection_threshold)

    def _on_projection_changed(self, text: str):
        self.proj_label.setText(f"Projection: {text}")
        self.embedding_plot.set_projection_method(text)
        print("Projection method set to", text)

    # ---------- Logging embeddings to CSV ---------- #

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

    # ---------- Frame update loop ---------- #

    def update_frame(self):
        # If frozen, just redraw the last frame (if any) and return
        if self.is_frozen and self.last_frame_bgr is not None:
            self._display_main_frame(self.last_frame_bgr)
            return

        ret, frame_bgr = self.cap.read()
        if not ret:
            return

        # Keep last frame for freeze mode
        self.last_frame_bgr = frame_bgr.copy()

        # Convert to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run face detection
        boxes, probs = self.mtcnn.detect(frame_rgb)

        face_crop = None

        if boxes is not None and len(boxes) > 0:
            boxes = np.array(boxes)
            probs = np.array(probs)

            # Filter by detection threshold
            valid_indices = [
                i for i, p in enumerate(probs) if p is not None and p >= self.detection_threshold
            ]

            if valid_indices:
                valid_boxes = boxes[valid_indices]

                # Compute area of each valid box and pick the largest
                areas = (valid_boxes[:, 2] - valid_boxes[:, 0]) * (
                    valid_boxes[:, 3] - valid_boxes[:, 1]
                )
                best_local_idx = int(np.argmax(areas))
                best_idx = valid_indices[best_local_idx]

                x1, y1, x2, y2 = boxes[best_idx].astype(int)

                # Clamp to frame size
                h, w, _ = frame_rgb.shape
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 > x1 and y2 > y1:
                    face_crop = frame_rgb[y1:y2, x1:x2, :]

                    # Only generate embedding + plots + logging if capturing
                    if self.is_capturing:
                        try:
                            emb = get_embedding(face_crop)

                            # Cosine similarity vs previous embedding
                            if self.last_embedding is not None:
                                sim = cosine_similarity(self.last_embedding, emb)
                                self.similarity_widget.add_value(sim)

                            self.last_embedding = emb

                            # Update embedding plots
                            self.embedding_plot.add_embedding(emb)
                            # Log to CSV
                            self._log_embedding(emb)
                        except Exception as e:
                            print("Embedding error:", e)

                # Draw bounding box for the largest valid face
                cv2.rectangle(
                    frame_bgr,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

        # Update cropped face preview (even if not capturing)
        if face_crop is not None:
            self._update_cropped_face(face_crop)

        # Display the main frame
        self._display_main_frame(frame_bgr)

    def _display_main_frame(self, frame_bgr: np.ndarray):
        display_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        display_rgb = np.ascontiguousarray(display_rgb, dtype=np.uint8)

        h, w, ch = display_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            display_rgb.tobytes(),  # bytes, not memoryview
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
        # Ensure contiguous uint8 array
        face_rgb = np.ascontiguousarray(face_rgb, dtype=np.uint8)

        h, w, ch = face_rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(
            face_rgb.tobytes(),  # bytes, not memoryview
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
        # Release camera on close
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


# ---------------- Entry point ---------------- #


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
