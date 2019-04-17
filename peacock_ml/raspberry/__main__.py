import argparse

from keras.models import load_model
from picamera import PiCamera

from peacock_ml.raspberry.car_detector import CarDetector

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-m', '--model', help='Path to the model', default='model_qui_dechire.h5')
argument_parser.add_argument('-p', '--processing', help='Use processing', action='store_true')
args = argument_parser.parse_args()

print('Using {} in Keras'.format(args.model))
model = load_model(args.model)
pi_camera = PiCamera()

car_detector = CarDetector(pi_camera, model)
car_detector.run()
