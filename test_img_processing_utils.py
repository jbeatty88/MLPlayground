from unittest import TestCase

import tensorflow

import img_processing_utils as IU


class TestImageUtils(TestCase):
    def test_decode_image_gets_valid_response(self):
        self.assertIsNotNone(IU.decode_image('images/b88convert.png', 'png', 4))

    def test_decode_image_no_file(self):
        self.assertRaises(tensorflow.python.framework.errors_impl.NotFoundError)