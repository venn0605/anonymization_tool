from bz2 import compress
import os, sys
import argparse
import tifffile as tiff
import numpy as np
import cv2

cur_dir  = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)

if root_dir not in sys.path:
    sys.path.append(root_dir)

from anonymizer import *
from anonymizer.anonymization.anonymizer import Anonymizer
from anonymizer.detection.detector import Detector
from anonymizer.detection.weights import download_weights, get_weights_path
from anonymizer.obfuscation.obfuscator import Obfuscator
from anonymizer.utils import tools


def parse_args():
    parser = argparse.ArgumentParser(
        description='Anonymize faces and license plates in a series of images.')
    parser.add_argument('--source-dir', type=str, default='', dest='srcDir', help='The path of input images')
    parser.add_argument('--target-dir', type=str, default='', dest='tgtDir', help='The path of output blurred images')
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights_foler',
                        help='Path to the folder where the weights are stored. If no weights with the '
                             'appropriate names are found they will be downloaded automatically.')
    parser.add_argument('--image-extensions', required=False, default='jpg,png',
                        metavar='"jpg,png"',
                        help='Comma-separated list of file types that will be anonymized')
    parser.add_argument('--face-threshold', type=float, required=False, default=0.3,
                        metavar='0.3',
                        help='Detection confidence needed to anonymize a detected face. '
                             'Must be in [0.001, 1.0]')
    parser.add_argument('--plate-threshold', type=float, required=False, default=0.3,
                        metavar='0.3',
                        help='Detection confidence needed to anonymize a detected license plate. '
                             'Must be in [0.001, 1.0]')
    parser.add_argument('--write-detections', dest='write_detections', action='store_true')
    parser.add_argument('--no-write-detections', dest='write_detections', action='store_false')
    parser.set_defaults(write_detections=True)
    parser.add_argument('--obfuscation-kernel', required=False, default='21,2,9',
                        metavar='kernel_size,sigma,box_kernel_size',
                        help='This parameter is used to change the way the blurring is done. '
                             'For blurring a gaussian kernel is used. The default size of the kernel is 21 pixels '
                             'and the default value for the standard deviation of the distribution is 2. '
                             'Higher values of the first parameter lead to slower transitions while blurring and '
                             'larger values of the second parameter lead to sharper edges and less blurring. '
                             'To make the transition from blurred areas to the non-blurred image smoother another '
                             'kernel is used which has a default size of 9. Larger values lead to a smoother '
                             'transition. Both kernel sizes must be odd numbers.')
    args = parser.parse_args()

    print(f'input_path: {args.srcDir}')
    print(f'output_path: {args.tgtDir}')
    print(f'weights: {args.weights}')
    print(f'image-extensions: {args.image_extensions}')
    print(f'face-threshold: {args.face_threshold}')
    print(f'plate-threshold: {args.plate_threshold}')
    print(f'write-detections: {args.write_detections}')
    print(f'obfuscation-kernel: {args.obfuscation_kernel}')

    return args

if __name__ =="__main__":

    args = parse_args()
    srcDir = args.srcDir
    tgtDir = args.tgtDir

    fileList = []
    for dirPath, dirNames, fileNames in os.walk(srcDir):
        for fileName in fileNames:
            if fileName.lower().endswith('.tiff'):
                measFile = os.path.join(dirPath, fileName)
                fileList.append(measFile)
    
    download_weights(download_directory=args.weights)
    kernel_size, sigma, box_kernel_size = args.obfuscation_kernel.split(',')
    obfuscator = Obfuscator(kernel_size=int(kernel_size), sigma=float(sigma), box_kernel_size=int(box_kernel_size))
    detectors = {
        'face': Detector(kind='face', weights_path=get_weights_path(args.weights, kind='face')),
        'plate': Detector(kind='plate', weights_path=get_weights_path(args.weights, kind='plate'))
    }
    detection_thresholds = {
        'face': args.face_threshold,
        'plate': args.plate_threshold
    }
    anonymizer = Anonymizer(obfuscator=obfuscator, detectors=detectors)

    for imageFile in fileList:
    
        tFile = os.path.basename(imageFile)[:-5] + '_Blurred.tiff'
        tFile = os.path.join(tgtDir, tFile)

        luv_image = tiff.imread(imageFile)

        luv_Y = np.squeeze(luv_image[0,:,:,:])
        luv_U = np.squeeze(luv_image[1,:,:,:])
        luv_V = np.squeeze(luv_image[2,:,:,:])

        luv_uint8 = tools.YUV16toYUV8(luv_Y, luv_U, luv_V)
        rgb_uint8 = cv2.cvtColor(luv_uint8, cv2.COLOR_YUV2RGB)

        anonymized_image, detections = anonymizer.anonymize_image(image=rgb_uint8, detection_thresholds=detection_thresholds)

        luv_image = tools.boxLocal(detections, luv_image)

        tiff.imwrite(tFile, luv_image, compress=9, byteorder = '>')




