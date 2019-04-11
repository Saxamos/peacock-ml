import argparse

from keras.models import load_model
from picamera import PiCamera

from peacock_ml.raspberry.car_detector import CarDetector

MODEL_PATH = 'model_qui_dechire.h5'

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-m', '--model', help='Path of the model', default=MODEL_PATH)
argument_parser.add_argument('-p', '--processing', help='Use processing', action='store_true')
args = argument_parser.parse_args()

print('Using {} in Keras'.format(args.model))
model = load_model(args.model)

car_detector = CarDetector(PiCamera(), model)
car_detector.run()
