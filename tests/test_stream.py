import tempfile
import unittest
from pathlib import Path

from defectdock.stream import CameraStream, stream_camera


class CameraStreamTests(unittest.TestCase):
    def test_construct_and_close_without_open(self):
        stream = CameraStream("0", "best.pt", ["defect"])
        stream.close()  # 未 open 时 close 应无副作用
        self.assertEqual(stream.source, "0")

    def test_invalid_source_reports_open_failure(self):
        result = stream_camera("definitely-missing-video.mp4", "best.pt", ["defect"], max_frames=3)
        self.assertEqual(result.reason, "source_open_failed")
        self.assertEqual(result.frames, 0)

    def test_detect_requires_model_on_frame(self):
        # detect() 需要真实模型，这里只验证类接口存在且未加载模型时能构造
        stream = CameraStream("0", "best.pt", ["a", "b"])
        self.assertIsNone(stream._model)
        stream.close()

    def test_network_source_read_failures_abort_with_reason(self):
        # M5 回归：网络源持续读失败应在阈值后明确中止，而不是无限空转。
        stream = CameraStream(
            "rtsp://fake/stream", "best.pt", ["a"], max_read_failures=3
        )
        stream.open = lambda: True
        stream.read_frame = lambda: None
        result = stream.run()
        self.assertEqual(result.reason, "stream_read_failed")
        self.assertEqual(result.frames, 0)

    def test_local_file_eof_is_clean_end_of_stream(self):
        # M5 回归：本地视频正常播完仍立即 end_of_stream，不因网络重试逻辑延迟。
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "clip.mp4"
            video.write_bytes(b"fake-video-bytes")
            stream = CameraStream(str(video), "best.pt", ["a"])
            stream.open = lambda: True
            stream.read_frame = lambda: None
            result = stream.run()
            self.assertEqual(result.reason, "end_of_stream")
            self.assertEqual(result.frames, 0)

    def test_transient_read_failures_recover(self):
        # M5 回归：阈值内的瞬断应恢复并继续处理帧。
        import numpy as np

        stream = CameraStream("rtsp://fake/stream", "best.pt", ["a"], max_read_failures=5)
        stream.open = lambda: True
        reads = iter([None, None, np.zeros((4, 4, 3), dtype="uint8")])
        stream.read_frame = lambda: next(reads, None)
        stream.detect = lambda frame: {"detections": [], "count": 0}
        result = stream.run(max_frames=1)
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.frames, 1)


    def test_open_timeout_aborts_dead_network_source(self):
        # M6 回归：FFmpeg 后端的 CAP_PROP_OPEN_TIMEOUT_MSEC 不生效时，死 RTSP 源
        # 的 cap.open() 会阻塞到 FFmpeg 自身默认超时。守护线程 join 硬超时必须兜底，
        # 让 open() 在 open_timeout_ms 附近快速返回失败，而不是挂死检测器。
        import time as _time

        import cv2

        stream = CameraStream("rtsp://dead/stream", "best.pt", ["a"], open_timeout_ms=300)

        class _BlockingCap:
            def __init__(self, *a, **k):
                pass

            def set(self, *a):
                return True

            def open(self, *a):
                _time.sleep(3)
                return False

            def release(self):
                pass

        original = cv2.VideoCapture
        cv2.VideoCapture = _BlockingCap
        try:
            start = _time.monotonic()
            ok = stream.open()
            elapsed = _time.monotonic() - start
        finally:
            cv2.VideoCapture = original
        self.assertFalse(ok)
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
