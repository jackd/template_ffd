"""
Hacked mobilenet version implementation was changed at tensorflow v1.8

Both other files in this directory are minor changes of
tf.keras.applications.mobilenet
with error checks on input sizes removed.

Tensorflow version prior to 1.8 use the old version.

Note: keras doesn't play well with native tensorflow. If re-implementing,
users are strongly encouraged to use `models.research.slim.nets.mobilenet`
from [here](https://github.com/tensorflow/models).
"""

try:
    from mobilenet_1p8 import MobileNet
except ImportError:
    from mobilenet_old import MobileNet

__all__ = [MobileNet]
