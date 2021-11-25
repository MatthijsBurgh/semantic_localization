import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

_bridge = None


def _get_cv_bridge() -> CvBridge:
    global _bridge
    if not _bridge:
        _bridge = CvBridge()
    return _bridge


def color_imgmsg_to_bgr_np_ndarray(msg: Image) -> np.ndarray:
    bridge = _get_cv_bridge()
    return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def uint_np_ndarray_to_mono_imgmsg(arr: np.ndarray, encoding: str = "mono8") -> Image:
    assert encoding in ["mono8", "mono16"], "Encoding should be mono8 or mono16"
    bridge = _get_cv_bridge()
    return bridge.cv2_to_imgmsg(arr, encoding)
