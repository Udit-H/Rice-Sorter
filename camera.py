from picamera2 import Picamera2
import time
from libcamera import controls

class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        # main={"size": (1600, 1100)}
        video_config = self.picam2.create_video_configuration()
        self.picam2.configure(video_config)
        # Enable autofocus
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.picam2.start()
        time.sleep(2)  # Allow time for the camera to warm up

    def get_frame(self):
        frame = self.picam2.capture_image("main")
        # If the frame is a dict (with JPEG data), return the data.
        if isinstance(frame, dict):
            frame = frame.get("data", None)
            if frame is not None:
                return frame

        # Try to encode the frame to JPEG.
        try:
            from PIL import Image
            import io
            if hasattr(frame, 'save'):
                buf = io.BytesIO()
                frame.save(buf, format="JPEG")
                return buf.getvalue()
        except ImportError:
            pass

        import cv2
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            return jpeg.tobytes()
        else:
            return b""
