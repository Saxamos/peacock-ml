import os
import unittest
from unittest.mock import MagicMock

from freezegun import freeze_time

from peacock_ml import ROOT_PATH
from peacock_ml.raspberry.car_detector import CarDetector


class TestCarDetector(unittest.TestCase):
    def setUp(self):
        pi_camera = MagicMock()
        model = MagicMock()
        self.car_detector = CarDetector(pi_camera, model, saved_folder=os.path.join(ROOT_PATH, 'tests/raspberry'))

    def test_display_response_with_car_prediction(self):
        # Given
        prediction = 0

        # When
        self.car_detector._display_response(prediction)

        # Then
        # A car should display

    def test_display_response_with_no_car_prediction(self):
        # Given
        prediction = 1

        # When
        self.car_detector._display_response(prediction)

        # Then
        # Homer simpsons should talk to you

    def test_predict_if_car_return_right_inference(self):
        # Given
        image_path = os.path.join(ROOT_PATH, 'tests/raspberry/picamera_1554336000.jpg')

        # When
        prediction = self.car_detector._predict(image_path)

        # Then
        self.assertEqual(prediction, 0)

    @unittest.skip('need keyboard interaction')
    @freeze_time('2019-04-04')
    def test_with_mocked_camera_and_mocked_model(self):
        self.car_detector.run()
