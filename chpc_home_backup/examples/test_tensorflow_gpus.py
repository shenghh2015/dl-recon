
# Import Tensorflow and check that it can detect the GPU(s)

import tensorflow as tf

tf.Session(config=tf.ConfigProto(log_device_placement=True))


