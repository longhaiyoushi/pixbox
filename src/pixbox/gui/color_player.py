import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QImage,
    QIntValidator,
    QKeyEvent,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pixbox.core.color_primaries import (
    BT709Primaries,
    BT2020Primaries,
    ColorPrimaries,
    SrgbPrimaries,
)
from pixbox.core.color_range import ColorRange, FullRange, LimitedRange
from pixbox.core.color_space import ColorSpace
from pixbox.core.color_transfer import (
    BT709Transfer,
    BT2020Transfer,
    ColorTransfer,
    LinearTransfer,
    SrgbTransfer,
)
from pixbox.core.pixel_format import (
    BGR24,
    GRAY,
    GRAYF,
    RGB24,
    RGBF32,
    YUV420_I420,
    YUV420_NV12,
    YUV420_NV21,
    YUV420_YV12,
    YUV422_I422,
    YUV422_NV16,
    YUV422_NV61,
    YUV422_UYVY,
    YUV422_VYUY,
    YUV422_YUYV,
    YUV422_YV16,
    YUV422_YVYU,
    YUV444_I444,
    YUV444_NV24,
    YUV444_NV42,
    YUV444_YV24,
    PixelFormat,
    YUVFormat,
)

PIXEL_FORMATS: dict[str, type[PixelFormat]] = {
    'GRAY': GRAY,
    'GRAYF': GRAYF,
    'YUV420_I420': YUV420_I420,
    'YUV420_YV12': YUV420_YV12,
    'YUV420_NV12': YUV420_NV12,
    'YUV420_NV21': YUV420_NV21,
    'YUV422_I422': YUV422_I422,
    'YUV422_YV16': YUV422_YV16,
    'YUV422_NV16': YUV422_NV16,
    'YUV422_NV61': YUV422_NV61,
    'YUV422_YUYV': YUV422_YUYV,
    'YUV422_YVYU': YUV422_YVYU,
    'YUV422_UYVY': YUV422_UYVY,
    'YUV422_VYUY': YUV422_VYUY,
    'YUV444_I444': YUV444_I444,
    'YUV444_YV24': YUV444_YV24,
    'YUV444_NV24': YUV444_NV24,
    'YUV444_NV42': YUV444_NV42,
    'RGB24': RGB24,
    'BGR24': BGR24,
    'RGBF32': RGBF32,
}

PRIMARY_OPTIONS: dict[str, ColorPrimaries] = {
    'sRGB': SrgbPrimaries(),
    'BT.709': BT709Primaries(),
    'BT.2020': BT2020Primaries(),
}

RANGE_OPTIONS: dict[str, type[ColorRange]] = {
    'Full Range': FullRange,
    'Limited Range': LimitedRange,
}

TRANSFER_OPTIONS: dict[str, ColorTransfer] = {
    'Linear': LinearTransfer(),
    'sRGB': SrgbTransfer(),
    'BT.709': BT709Transfer(),
    'BT.2020': BT2020Transfer(),
}


def build_color_space(
    color_primary: str,
    color_transfer: str,
    color_range: str,
    pixel_format: str,
    height: int,
    width: int,
    stride: int,
    bits: int,
    expanded: bool = True,
) -> ColorSpace:
    if stride is None:
        stride = 0
    kwargs: dict[str, Any] = {
        'height': height,
        'width': width,
        'stride': stride,
    }
    if issubclass(PIXEL_FORMATS[pixel_format], YUVFormat):
        kwargs['bits'] = bits
        kwargs['expanded'] = expanded
    return ColorSpace(
        pixel_format=PIXEL_FORMATS[pixel_format](**kwargs),
        color_range=RANGE_OPTIONS[color_range](bits=bits),
        color_transfer=TRANSFER_OPTIONS[color_transfer],
        color_primaries=PRIMARY_OPTIONS[color_primary],
    )


def decode_frame_to_rgb(
    frame: bytes | bytearray | memoryview | np.ndarray[Any, Any],
    color_space: ColorSpace,
) -> np.ndarray[Any, Any]:
    if isinstance(frame, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(frame, dtype=np.uint8)
    else:
        raw = np.asarray(frame, dtype=np.uint8)
    rgb = color_space.convert2rgb(raw)
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.asarray(rgb, dtype=np.float32)


class SettingsDialog(QDialog):
    def __init__(
        self,
        data_format: dict[str, Any] | None = None,
        fps: int = 30,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('Video Settings')
        self.resize(480, 360)
        self.setMinimumSize(440, 320)
        self.setMaximumSize(520, 420)
        self.setModal(True)

        data_format = data_format if data_format else {}

        self.primary_combo = QComboBox()
        self.primary_combo.addItems(list(PRIMARY_OPTIONS))
        self.primary_combo.setCurrentText(data_format.get('primary', 'sRGB'))

        self.transfer_combo = QComboBox()
        self.transfer_combo.addItems(list(TRANSFER_OPTIONS))
        self.transfer_combo.setCurrentText(data_format.get('transfer', 'sRGB'))

        self.range_combo = QComboBox()
        self.range_combo.addItems(list(RANGE_OPTIONS))
        self.range_combo.setCurrentText(data_format.get('range', 'Full Range'))

        self.format_combo = QComboBox()
        self.format_combo.addItems(list(PIXEL_FORMATS))
        self.format_combo.setCurrentText(data_format.get('format', 'GRAY'))

        self.height_input = QLineEdit(str(data_format.get('height', 720)))
        self.height_input.setPlaceholderText('Height')
        self.height_input.setValidator(QIntValidator(1, 10000))

        self.width_input = QLineEdit(str(data_format.get('width', 1280)))
        self.width_input.setPlaceholderText('Width')
        self.width_input.setValidator(QIntValidator(1, 10000))

        self.stride_input = QLineEdit(str(data_format.get('stride', 0)))
        self.stride_input.setPlaceholderText('Stride')
        self.stride_input.setValidator(QIntValidator(0, 10000))

        self.bits_combo = QComboBox()
        self.bits_combo.addItems(['8', '10', '12', '14', '16', '32'])
        self.bits_combo.setCurrentText(str(data_format.get('bits', 8)))

        self.expanded_check_box = QCheckBox()
        self.expanded_check_box.setChecked(data_format.get('expanded', True))

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(fps)

        header = QLabel('Configure how raw video files are interpreted.')
        header.setWordWrap(True)
        header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        header.setStyleSheet('font-size: 10pt;')

        color_group = QGroupBox('Color pipeline')
        color_layout = QFormLayout(color_group)
        color_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        color_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        color_layout.addRow('Color Primary:', self.primary_combo)
        color_layout.addRow('Color Transfer:', self.transfer_combo)
        color_layout.addRow('Color Range:', self.range_combo)

        pixel_group = QGroupBox('Pixel layout')
        pixel_layout = QGridLayout(pixel_group)
        pixel_layout.setContentsMargins(12, 12, 12, 12)
        pixel_layout.setSpacing(8)
        pixel_layout.setColumnStretch(1, 1)
        pixel_layout.setColumnStretch(3, 1)
        pixel_layout.addWidget(QLabel('Pixel Format:'), 0, 0)
        pixel_layout.addWidget(self.format_combo, 0, 1, 1, 3)
        pixel_layout.addWidget(QLabel('Height:'), 1, 0)
        pixel_layout.addWidget(self.height_input, 1, 1)
        pixel_layout.addWidget(QLabel('Width:'), 1, 2)
        pixel_layout.addWidget(self.width_input, 1, 3)
        pixel_layout.addWidget(QLabel('Stride:'), 2, 0)
        pixel_layout.addWidget(self.stride_input, 2, 1)
        pixel_layout.addWidget(QLabel('Bits:'), 2, 2)
        pixel_layout.addWidget(self.bits_combo, 2, 3)
        pixel_layout.addWidget(QLabel('Expanded Mode:'), 3, 0)
        pixel_layout.addWidget(self.expanded_check_box, 3, 1)

        playback_group = QGroupBox('Playback')
        playback_layout = QFormLayout(playback_group)
        playback_layout.addRow('FPS:', self.fps_spin)

        buttons = QHBoxLayout()
        ok_button = QPushButton('Ok')
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(header)
        layout.addWidget(color_group)
        layout.addWidget(pixel_group)
        layout.addWidget(playback_group)
        layout.addLayout(buttons)

    def data_format(self) -> dict[str, Any]:
        return {
            'primary': self.primary_combo.currentText(),
            'transfer': self.transfer_combo.currentText(),
            'range': self.range_combo.currentText(),
            'format': self.format_combo.currentText(),
            'height': int(self.height_input.text() or 0),
            'width': int(self.width_input.text() or 0),
            'stride': int(self.stride_input.text() or 0),
            'bits': int(self.bits_combo.currentText() or 8),
            'expanded': self.expanded_check_box.isChecked(),
        }

    def fps(self) -> int:
        return self.fps_spin.value()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Pixbox Player')
        self.resize(960, 720)
        self.setAcceptDrops(True)

        self.raw_bytes: np.memmap | None = None
        self.frame_buffers: list[np.ndarray[Any, Any]] = []
        self.current_frame = 0
        self.playing = False
        self.current_file: Path | None = None
        self.current_format: dict[str, Any] | None = None
        self.fps = 30
        self.zoom_level = 1.0

        self.create_menu_bar()
        self.create_content()
        self.create_footer()

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addLayout(self.content)
        layout.addLayout(self.footer)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)
        self.update_timer_interval()

        self.statusBar().showMessage('Select a raw binary file to begin')

    def create_menu_bar(self) -> None:
        self.file_menu = self.menuBar().addMenu('&File')

        self.open_action = QAction('&Open', self)
        self.open_action.setShortcut('Ctrl+O')
        self.open_action.triggered.connect(self.on_open)
        self.file_menu.addAction(self.open_action)

        self.save_action = QAction('&Save Frame', self)
        self.save_action.setShortcut('Ctrl+S')
        self.save_action.triggered.connect(self.on_save)
        self.file_menu.addAction(self.save_action)

        self.file_menu.addSeparator()

        self.setup_action = QAction('Settings', self)
        self.setup_action.setShortcut('Ctrl+,')
        self.setup_action.triggered.connect(self.on_setup)
        self.file_menu.addAction(self.setup_action)

        self.file_menu.addSeparator()

        self.exit_action = QAction('E&xit', self)
        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        self.view_menu = self.menuBar().addMenu('&View')

        self.first_action = QAction('⏮ First Frame', self)
        self.first_action.setShortcut('Ctrl+Left')
        self.first_action.triggered.connect(self.first_frame)
        self.view_menu.addAction(self.first_action)

        self.prev_action = QAction('⏪ Previous Frame', self)
        self.prev_action.setShortcut('Left')
        self.prev_action.triggered.connect(self.previous_frame)
        self.view_menu.addAction(self.prev_action)

        self.play_action = QAction('▶️ Play / Pause', self)
        self.play_action.setShortcut('Space')
        self.play_action.triggered.connect(self.toggle_play)
        self.view_menu.addAction(self.play_action)

        self.next_action = QAction('⏩ Next Frame', self)
        self.next_action.setShortcut('Right')
        self.next_action.triggered.connect(self.next_frame)
        self.view_menu.addAction(self.next_action)

        self.last_action = QAction('⏭ Last Frame', self)
        self.last_action.setShortcut('Ctrl+Right')
        self.last_action.triggered.connect(self.last_frame)
        self.view_menu.addAction(self.last_action)

        self.view_menu.addSeparator()

        self.zoom_in_action = QAction('Zoom &In', self)
        self.zoom_in_action.setShortcut('Ctrl+=')
        self.zoom_in_action.triggered.connect(self.zoom_in)
        self.view_menu.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction('Zoom &Out', self)
        self.zoom_out_action.setShortcut('Ctrl+-')
        self.zoom_out_action.triggered.connect(self.zoom_out)
        self.view_menu.addAction(self.zoom_out_action)

        self.reset_zoom_action = QAction('Reset &Zoom', self)
        self.reset_zoom_action.setShortcut('Ctrl+0')
        self.reset_zoom_action.triggered.connect(self.reset_zoom)
        self.view_menu.addAction(self.reset_zoom_action)

    def create_content(self) -> None:
        self.content = QHBoxLayout()

        self.image_label = QLabel('No file loaded')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.content.addWidget(self.image_label)

    def create_footer(self) -> None:
        self.footer = QHBoxLayout()
        self.footer.setContentsMargins(12, 12, 12, 12)
        self.footer.setSpacing(10)

        self.frame_info_label = QLabel('Frame 0 / 0')
        self.frame_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(True)
        self.slider.setPageStep(1)
        self.slider.setTracking(False)
        self.slider.valueChanged.connect(self.on_slider_changed)

        self.first_button = QPushButton('⏮')
        self.first_button.setToolTip('First frame')
        self.first_button.clicked.connect(self.first_frame)

        self.prev_button = QPushButton('⏪')
        self.prev_button.setToolTip('Previous frame')
        self.prev_button.clicked.connect(self.previous_frame)

        self.play_button = QPushButton('▶️')
        self.play_button.setToolTip('Play / Pause')
        self.play_button.clicked.connect(self.toggle_play)

        self.next_button = QPushButton('⏩')
        self.next_button.setToolTip('Next frame')
        self.next_button.clicked.connect(self.next_frame)

        self.last_button = QPushButton('⏭')
        self.last_button.setToolTip('Last frame')
        self.last_button.clicked.connect(self.last_frame)

        self.footer.addWidget(self.frame_info_label)
        self.footer.addWidget(self.slider)

        self.footer.addWidget(self.first_button)
        self.footer.addWidget(self.prev_button)
        self.footer.addWidget(self.play_button)
        self.footer.addWidget(self.next_button)
        self.footer.addWidget(self.last_button)

    def on_open(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Open raw video file',
            str(Path.home()),
            'Raw video files (*.*)',
        )
        if not filename:
            return
        self.open_file(filename)

    def open_file(self, filename: str) -> None:
        dialog = SettingsDialog(self.current_format, self.fps, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.fps = dialog.fps()
        self.update_timer_interval()

        data_format = dialog.data_format()
        self.current_format = data_format

        if self.current_file != Path(filename):
            self.current_file = Path(filename)
            self.apply_format()

    def on_setup(self) -> None:
        dialog = SettingsDialog(self.current_format, self.fps, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.fps = dialog.fps()
        self.update_timer_interval()

        data_format = dialog.data_format()
        if self.current_format != data_format:
            self.current_format = data_format
            if self.current_file:
                self.apply_format()

    def apply_format(self) -> None:
        if not self.current_format or not self.current_file:
            return

        color_space = build_color_space(
            color_primary=self.current_format['primary'],
            color_transfer=self.current_format['transfer'],
            color_range=self.current_format['range'],
            pixel_format=self.current_format['format'],
            height=self.current_format['height'],
            width=self.current_format['width'],
            stride=self.current_format['stride'],
            bits=self.current_format['bits'],
            expanded=self.current_format['expanded'],
        )

        self.raw_bytes = np.memmap(self.current_file, dtype=np.uint8, mode='r')
        self.frame_buffers = []
        self.current_frame = 0
        self.playing = False
        self.zoom_level = 1.0
        self.play_button.setText('▶️')
        self.timer.stop()

        frame_size = color_space.pixel_format.bytes_per_frame
        frame_count = max(1, math.ceil(self.raw_bytes.size / frame_size))
        for index in range(frame_count):
            start = index * frame_size
            end = start + frame_size
            frame = self.raw_bytes[start:end]
            pad = frame_size - frame.size
            if pad > 0:
                frame = np.pad(frame, (0, pad))
            rgb = decode_frame_to_rgb(frame, color_space)
            self.frame_buffers.append(rgb)

        self.slider.setMaximum(max(0, len(self.frame_buffers) - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(len(self.frame_buffers) > 1)
        self.display_current_frame()
        self.statusBar().showMessage(
            f'{self.current_file.name} • {len(self.frame_buffers)} frame(s)'
        )

    def display_current_frame(self) -> None:
        if not self.frame_buffers:
            self.image_label.setText('No file loaded')
            self.frame_info_label.setText('Frame 0 / 0')
            self.slider.setValue(0)
            return

        frame = self.frame_buffers[self.current_frame]
        height, width = frame.shape[:2]
        rgb8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        self.image = QImage(
            rgb8.tobytes(),
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        )
        scaled_size = QSize(
            max(1, int(width * self.zoom_level)),
            max(1, int(height * self.zoom_level)),
        )
        self.image_label.setPixmap(
            QPixmap.fromImage(self.image).scaled(
                scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.frame_info_label.setText(
            f'Frame {self.current_frame + 1} / {len(self.frame_buffers)}'
        )
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)

    def on_save(self) -> None:
        if not self.frame_buffers:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save current frame',
            str(Path.home() / 'frame.png'),
            'PNG image (*.png)',
        )
        if not filename:
            return

        if not self.image.save(filename):
            self.statusBar().showMessage(f'Failed to save {filename}')
            return
        self.statusBar().showMessage(f'Saved {Path(filename).name}')

    def first_frame(self) -> None:
        if not self.frame_buffers:
            return
        if self.playing:
            self.timer.stop()
        self.current_frame = 0
        self.display_current_frame()

    def previous_frame(self) -> None:
        if not self.frame_buffers:
            return
        if self.playing:
            self.timer.stop()
        self.current_frame = max(0, self.current_frame - 1)
        self.display_current_frame()

    def next_frame(self) -> None:
        if not self.frame_buffers:
            return
        if self.playing:
            self.timer.stop()
        self.current_frame = min(
            len(self.frame_buffers) - 1, self.current_frame + 1
        )
        self.display_current_frame()

    def last_frame(self) -> None:
        if not self.frame_buffers:
            return
        if self.playing:
            self.timer.stop()
        self.current_frame = len(self.frame_buffers) - 1
        self.display_current_frame()

    def advance_frame(self) -> None:
        if not self.frame_buffers:
            return
        if self.current_frame >= len(self.frame_buffers) - 1:
            self.current_frame = 0
        else:
            self.current_frame += 1
        self.display_current_frame()

    def toggle_play(self) -> None:
        if not self.frame_buffers:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_button.setText('⏸️')
            self.play_action.setIconText('⏸️')
            self.play_action.setText('⏸️ Pause')
            self.timer.start()
        else:
            self.play_button.setText('▶️')
            self.play_action.setIconText('▶️')
            self.play_action.setText('▶️ Play / Pause')
            self.timer.stop()

    def on_slider_changed(self, value: int) -> None:
        self.current_frame = value
        self.display_current_frame()

    def update_timer_interval(self) -> None:
        self.timer.setInterval(int(1000 / self.fps))

    def zoom_in(self) -> None:
        self.zoom_level = min(5.0, self.zoom_level + 0.25)
        self.display_current_frame()

    def zoom_out(self) -> None:
        self.zoom_level = max(0.25, self.zoom_level - 0.25)
        self.display_current_frame()

    def reset_zoom(self) -> None:
        self.zoom_level = 1.0
        self.display_current_frame()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.toggle_play()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Left:
            self.previous_frame()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.next_frame()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData() and event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls() if event.mimeData() else []
        if not urls:
            super().dropEvent(event)
            return
        path = urls[0].toLocalFile()
        if path:
            self.open_file(path)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.frame_buffers:
            self.display_current_frame()


def get_icon_path(filename: str) -> Path:
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / 'resource' / filename
    return Path(__file__).parent.parent.parent.parent / 'resource' / filename


def main() -> None:
    app = QApplication.instance() or QApplication([])
    app = cast(QApplication, app)
    app.setWindowIcon(QIcon(str(get_icon_path('pixbox.ico'))))
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
