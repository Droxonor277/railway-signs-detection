import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QHBoxLayout,
                             QVBoxLayout, QWidget, QFileDialog, QSlider, QStyle, QLabel)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QUrl, QTime


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Railway signs detection player")
        self.resize(1280, 720)

        # 1. Multimedia Components
        self.mediaPlayer = QMediaPlayer()
        self.videoWidget = QVideoWidget()
        self.audioOutput = QAudioOutput()

        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.setAudioOutput(self.audioOutput)

        # 2. UI Elements
        self.openButton = QPushButton("Open Video")
        self.openButton.clicked.connect(self.open_file)

        self.playButton = QPushButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_video)

        self.rewindButton = QPushButton()
        self.rewindButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward))
        self.rewindButton.clicked.connect(self.rewind)

        self.forwardButton = QPushButton()
        self.forwardButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward))
        self.forwardButton.clicked.connect(self.fast_forward)

        # --- New: Time Label ---
        self.timeLabel = QLabel("00:00 / 00:00")
        self.timeLabel.setMinimumWidth(100)

        self.positionSlider = QSlider(Qt.Orientation.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.set_position)

        # 3. Layout management
        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)

        controls = QHBoxLayout()
        controls.addWidget(self.openButton)
        controls.addWidget(self.rewindButton)
        controls.addWidget(self.playButton)
        controls.addWidget(self.forwardButton)
        controls.addWidget(self.positionSlider)
        controls.addWidget(self.timeLabel) # Added to the end of the bar

        container = QWidget()
        container.setLayout(layout)
        layout.addLayout(controls)
        self.setCentralWidget(container)

        # 4. Signals
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    # --- Feature Methods ---

    def format_time(self, ms):
        """Converts milliseconds to a MM:SS or HH:MM:SS string."""
        time = QTime(0, 0).addMSecs(ms)
        # If video is longer than an hour, show hours
        if ms >= 3600000:
            return time.toString("hh:mm:ss")
        return time.toString("mm:ss")

    def update_duration_label(self):
        """Updates the text showing current position vs total duration."""
        current = self.format_time(self.mediaPlayer.position())
        total = self.format_time(self.mediaPlayer.duration())
        self.timeLabel.setText(f"{current} / {total}")

    def open_file(self):
        file_dialog = QFileDialog(self)

        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
        )
        if file_path:
            self.mediaPlayer.setSource(QUrl.fromLocalFile(file_path))
            # It's often safer to play() only after the source is set
            self.mediaPlayer.play()
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def play_video(self):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mediaPlayer.pause()
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self.mediaPlayer.play()
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def rewind(self):
        self.mediaPlayer.setPosition(max(0, self.mediaPlayer.position() - 10000))

    def fast_forward(self):
        self.mediaPlayer.setPosition(min(self.mediaPlayer.duration(), self.mediaPlayer.position() + 10000))

    def position_changed(self, position):
        self.positionSlider.setValue(position)
        self.update_duration_label() # Update text as video plays

    def duration_changed(self, duration):
        self.positionSlider.setRange(0, duration)
        self.update_duration_label() # Update total time when file loads

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())