import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QCursor,
    QDoubleValidator,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QImage,
    QIntValidator,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
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
    RGBFormat,
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


class PixelInfoLabel(QLabel):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setContentsMargins(10, 8, 10, 8)

        font = self.font()
        font.setFamily('Consolas')
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.setFont(font)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(244, 246, 248))
        self.setPalette(palette)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255, 48), 1))
        painter.setBrush(QColor(24, 28, 33, 224))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)
        painter.end()
        super().paintEvent(event)


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
    valid_min: float = 0.0,
    valid_max: float = 0.0,
) -> ColorSpace:
    if stride is None:
        stride = 0
    kwargs: dict[str, Any] = {
        'height': height,
        'width': width,
        'stride': stride,
        'min': valid_min,
        'max': valid_max,
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


class SettingsDialog(QDialog):
    def __init__(
        self,
        data_format: dict[str, Any] | None = None,
        fps: int = 30,
        loop: bool = False,
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

        self.bits_spin = QSpinBox()
        self.bits_spin.setRange(8, 32)
        self.bits_spin.setValue(data_format.get('bits', 8))

        self.min_input = QLineEdit()
        self.min_input.setPlaceholderText('Min')
        self.min_input.setValidator(QDoubleValidator())
        self.min_input.setText(str(data_format.get('valid_min', 0)))

        self.max_input = QLineEdit()
        self.max_input.setPlaceholderText('Max')
        self.max_input.setValidator(QDoubleValidator())
        self.max_input.setText(str(data_format.get('valid_max', 0)))

        self.expanded_check_box = QCheckBox()
        self.expanded_check_box.setChecked(data_format.get('expanded', True))

        self.endian_check_box = QCheckBox()
        self.endian_check_box.setChecked(data_format.get('big_endian', False))

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(fps)

        self.loop_check_box = QCheckBox()
        self.loop_check_box.setChecked(loop)

        header = QLabel('Configure how raw file is interpreted.')
        header.setWordWrap(True)
        header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        header_font = header.font()
        header_font.setPointSize(10)
        header.setFont(header_font)

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
        pixel_layout.setColumnMinimumWidth(0, 100)
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
        pixel_layout.addWidget(self.bits_spin, 2, 3)
        pixel_layout.addWidget(QLabel('Min:'), 3, 0)
        pixel_layout.addWidget(self.min_input, 3, 1)
        pixel_layout.addWidget(QLabel('Max:'), 3, 2)
        pixel_layout.addWidget(self.max_input, 3, 3)
        pixel_layout.addWidget(QLabel('Expanded:'), 4, 0)
        pixel_layout.addWidget(self.expanded_check_box, 4, 1)
        pixel_layout.addWidget(QLabel('Big endian:'), 4, 2)
        pixel_layout.addWidget(self.endian_check_box, 4, 3)

        playback_group = QGroupBox('Playback')
        playback_layout = QGridLayout(playback_group)
        playback_layout.setContentsMargins(12, 12, 12, 12)
        playback_layout.setSpacing(8)
        playback_layout.setColumnMinimumWidth(0, 100)
        playback_layout.setColumnStretch(1, 1)
        playback_layout.setColumnStretch(3, 1)
        playback_layout.addWidget(QLabel('FPS:'), 0, 0)
        playback_layout.addWidget(self.fps_spin, 0, 1)
        playback_layout.addWidget(QLabel('Loop:'), 0, 2)
        playback_layout.addWidget(self.loop_check_box, 0, 3)

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
            'bits': self.bits_spin.value(),
            'valid_min': float(self.min_input.text() or 0.0),
            'valid_max': float(self.max_input.text() or 0.0),
            'expanded': self.expanded_check_box.isChecked(),
            'big_endian': self.endian_check_box.isChecked(),
        }

    def fps(self) -> int:
        return self.fps_spin.value()

    def loop(self) -> bool:
        return self.loop_check_box.isChecked()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Pixbox Player')
        self.resize(960, 720)
        self.setAcceptDrops(True)

        self.source_file: Path | None = None
        self.source_bytes: np.memmap | None = None
        self.data_format: dict[str, Any] | None = None
        self.color_space: ColorSpace | None = None
        self.frame_size = 0
        self.frame_count = 0

        self.frame_index: int | None = None
        self.source_value: np.ndarray[Any, Any] | None = None
        self.source_image: QImage | None = None
        self.zoom_level = 1.0

        self.playing = False
        self.fps = 30
        self.loop = False

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

        self.image_info_label = QLabel('No file loaded')
        self.image_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_info_label.setMinimumSize(320, 240)
        self.image_info_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_info_label.setMouseTracking(True)
        self.image_info_label.installEventFilter(self)

        self.pixel_info_label = PixelInfoLabel(self.image_info_label)
        self.pixel_info_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.pixel_info_label.setWordWrap(True)
        self.pixel_info_label.hide()

        self.content.addWidget(self.image_info_label)

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
            'Open file',
            str(Path.home()),
            'All files (*.*)',
        )
        if not filename:
            return
        self.open_file(filename)

    def open_file(self, filename: str) -> None:
        dialog = SettingsDialog(self.data_format, self.fps, self.loop, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.fps = dialog.fps()
        self.update_timer_interval()
        self.loop = dialog.loop()

        changed = False

        data_format = dialog.data_format()
        if self.data_format != data_format:
            self.data_format = data_format
            changed = True

        if self.source_file != Path(filename):
            self.source_file = Path(filename)
            changed = True

        if changed:
            self.apply_format()

    def on_setup(self) -> None:
        dialog = SettingsDialog(self.data_format, self.fps, self.loop, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.fps = dialog.fps()
        self.update_timer_interval()
        self.loop = dialog.loop()

        data_format = dialog.data_format()
        if self.data_format != data_format:
            self.data_format = data_format
            if self.source_file:
                self.apply_format()

    def apply_format(self) -> None:
        if not self.data_format or not self.source_file:
            return

        color_space = build_color_space(
            color_primary=self.data_format['primary'],
            color_transfer=self.data_format['transfer'],
            color_range=self.data_format['range'],
            pixel_format=self.data_format['format'],
            height=self.data_format['height'],
            width=self.data_format['width'],
            stride=self.data_format['stride'],
            bits=self.data_format['bits'],
            expanded=self.data_format['expanded'],
            valid_min=self.data_format['valid_min'],
            valid_max=self.data_format['valid_max'],
        )

        self.source_bytes = np.memmap(
            self.source_file, dtype=np.uint8, mode='r'
        )
        self.color_space = color_space
        self.frame_size = color_space.pixel_format.bytes_per_frame
        self.frame_count = math.ceil(self.source_bytes.size / self.frame_size)

        self.playing = False
        self.display_current_frame(0)

        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, self.frame_count - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(self.frame_count > 1)
        self.slider.blockSignals(False)
        self.play_button.setText('▶️')
        self.timer.stop()
        self.statusBar().showMessage(self.source_file.name)

    def display_current_frame(self, frame_index: int) -> None:
        if self.source_bytes is None or self.color_space is None:
            self.image_info_label.setText('No file loaded')
            self.pixel_info_label.hide()
            self.frame_info_label.setText('Frame 0 / 0')
            return

        if self.frame_count <= 0 or self.frame_size <= 0:
            self.image_info_label.setText('Empty file')
            self.pixel_info_label.hide()
            self.frame_info_label.setText('Frame 0 / 0')
            return

        if self.frame_index != frame_index:
            self.frame_index = frame_index
            self.source_value = self._load_frame(frame_index)
            self.source_image = self._value2image(self.source_value)

        if self.source_image is None:
            self.image_info_label.setText('No file loaded')
            self.pixel_info_label.hide()
            self.frame_info_label.setText('Frame 0 / 0')
            return

        scaled_size = QSize(
            max(1, int(self.source_image.width() * self.zoom_level)),
            max(1, int(self.source_image.height() * self.zoom_level)),
        )
        self.image_info_label.setPixmap(
            QPixmap.fromImage(self.source_image).scaled(
                scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.frame_info_label.setText(
            f'Frame {frame_index + 1} / {self.frame_count}'
        )
        self.refresh_hovered_pixel()

    def refresh_hovered_pixel(self) -> None:
        position = self.image_info_label.mapFromGlobal(QCursor.pos())
        if not self.image_info_label.rect().contains(position):
            self.pixel_info_label.hide()
            return
        self._update_pixel_info(position)

    def _load_frame(self, index: int) -> np.ndarray[Any, Any] | None:
        if self.source_bytes is None or self.color_space is None:
            return None
        start = index * self.frame_size
        end = start + self.frame_size
        frame = self.source_bytes[start:end]
        pad = self.frame_size - frame.size
        if pad > 0:
            frame = np.pad(frame, (0, pad))
        values = self.color_space.data2value(frame)
        return values

    def _update_pixel_info(self, position: QPoint) -> None:
        image = self.source_image
        pixmap = self.image_info_label.pixmap()
        if (
            image is None
            or pixmap is None
            or pixmap.isNull()
            or self.color_space is None
        ):
            self.pixel_info_label.hide()
            return

        left = (self.image_info_label.width() - pixmap.width()) // 2
        top = (self.image_info_label.height() - pixmap.height()) // 2
        image_x = position.x() - left
        image_y = position.y() - top
        if not (
            0 <= image_x < pixmap.width() and 0 <= image_y < pixmap.height()
        ):
            self.pixel_info_label.hide()
            return

        x = min(
            image.width() - 1, int(image_x * image.width() / pixmap.width())
        )
        y = min(
            image.height() - 1, int(image_y * image.height() / pixmap.height())
        )
        pixel_format = self.color_space.pixel_format
        channel_names: tuple[str, ...]
        if isinstance(pixel_format, (GRAY, GRAYF)):
            channel_names = ('Y',)
        elif isinstance(pixel_format, YUVFormat):
            channel_names = ('Y', 'U', 'V')
        elif isinstance(pixel_format, RGBFormat):
            channel_names = ('R', 'G', 'B')
        else:
            self.pixel_info_label.hide()
            return

        if self.source_value is None:
            self.pixel_info_label.hide()
            return
        values = self.source_value[y, x]
        channel_values = ' '.join(
            f'{name}:{value:g}'
            for name, value in zip(
                channel_names, values[: len(channel_names)], strict=False
            )
        )
        self.pixel_info_label.setText(channel_values)
        self._position_pixel_info(position)
        self.pixel_info_label.show()

    def _position_pixel_info(self, position: QPoint) -> None:
        margin = 12
        self.pixel_info_label.setMaximumWidth(
            max(margin, self.image_info_label.width() - 2 * margin)
        )
        self.pixel_info_label.adjustSize()
        max_x = (
            self.image_info_label.width()
            - self.pixel_info_label.width()
            - margin
        )
        max_y = (
            self.image_info_label.height()
            - self.pixel_info_label.height()
            - margin
        )
        x = position.x() + margin
        y = position.y() + margin
        if x > max_x:
            x = position.x() - self.pixel_info_label.width() - margin
        if y > max_y:
            y = position.y() - self.pixel_info_label.height() - margin
        if x < margin:
            x = margin
        elif x > max_x:
            x = max_x
        if y < margin:
            y = margin
        elif y > max_y:
            y = max_y
        self.pixel_info_label.move(x, y)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.image_info_label:
            if event.type() == QEvent.Type.MouseMove:
                if isinstance(event, QMouseEvent):
                    position = event.position().toPoint()
                    self._update_pixel_info(position)
            elif event.type() == QEvent.Type.Leave:
                self.pixel_info_label.hide()
            elif event.type() == QEvent.Type.Resize:
                self.refresh_hovered_pixel()
        return super().eventFilter(watched, event)

    def _value2image(self, value: np.ndarray[Any, Any] | None) -> QImage | None:
        if self.color_space is None or value is None:
            return None
        rgb = self.color_space.key2rgb(self.color_space.value2key(value))
        rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        height, width = rgb8.shape[:2]
        return QImage(
            rgb8.tobytes(),
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        )

    def on_save(self) -> None:
        if self.source_image is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save current frame',
            str(Path.home() / 'frame.png'),
            'PNG image (*.png)',
        )
        if not filename:
            return

        if not self.source_image.save(filename):
            self.statusBar().showMessage(f'Failed to save {filename}')
            return
        self.statusBar().showMessage(f'Saved {Path(filename).name}')

    def first_frame(self) -> None:
        if self.frame_count <= 0:
            return
        if self.playing:
            self.timer.stop()
        self.slider.setValue(0)

    def previous_frame(self) -> None:
        if self.frame_count <= 0:
            return
        if self.playing:
            self.timer.stop()
        frame_index = max(0, self.slider.value() - 1)
        self.slider.setValue(frame_index)

    def next_frame(self) -> None:
        if self.frame_count <= 0:
            return
        if self.playing:
            self.timer.stop()
        frame_index = min(self.frame_count - 1, self.slider.value() + 1)
        self.slider.setValue(frame_index)

    def last_frame(self) -> None:
        if self.frame_count <= 0:
            return
        if self.playing:
            self.timer.stop()
        frame_index = self.frame_count - 1
        self.slider.setValue(frame_index)

    def advance_frame(self) -> None:
        if self.frame_count <= 0:
            return
        frame_index = self.slider.value()
        if frame_index < self.frame_count - 1:
            frame_index += 1
            self.slider.setValue(frame_index)
        else:
            if self.loop:
                self.slider.setValue(0)
            else:
                self.playing = False
                self.play_button.setText('▶️')
                self.play_action.setIconText('▶️')
                self.play_action.setText('▶️ Play / Pause')
                self.timer.stop()

    def toggle_play(self) -> None:
        if self.frame_count <= 0:
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
        self.display_current_frame(value)

    def update_timer_interval(self) -> None:
        self.timer.setInterval(int(1000 / self.fps))

    def zoom_in(self) -> None:
        self.zoom_level = min(5.0, self.zoom_level + 0.25)
        frame_index = self.slider.value()
        self.display_current_frame(frame_index)

    def zoom_out(self) -> None:
        self.zoom_level = max(0.25, self.zoom_level - 0.25)
        frame_index = self.slider.value()
        self.display_current_frame(frame_index)

    def reset_zoom(self) -> None:
        self.zoom_level = 1.0
        frame_index = self.slider.value()
        self.display_current_frame(frame_index)

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
        if self.frame_count > 0:
            frame_index = self.slider.value()
            self.display_current_frame(frame_index)


def main() -> None:
    app = QApplication.instance() or QApplication([])
    app = cast(QApplication, app)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
