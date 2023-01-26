import argparse
import cv2
import math
import numpy as np
import os.path
import pathlib
import random
from os import listdir
from os.path import isfile, join

script_path = pathlib.Path(__file__).parent.absolute()


def display(img, wait=False):
    cv2.imshow('c', img)

    if wait:
        cv2.waitKey(0)


def get_faces(img, cascade):
    face_cascade = cv2.CascadeClassifier(cascade)

    return face_cascade.detectMultiScale(img, 1.3, 5)


def find_faces(img):
    root = os.path.join(script_path, 'haar')

    cascades = [join(root, f) for f in listdir(root) if isfile(join(root, f)) and 'face' in f]

    for cascade in cascades:
        faces = get_faces(img, cascade)

        if len(faces) > 0:
            return faces

    return ()


def get_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eye_cascade = cv2.CascadeClassifier(os.path.join(script_path, 'haar/haarcascade_eye.xml'))

    faces = find_faces(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        sorted_eyes = sorted(eyes, key=lambda x: float(x[0]))

        yield [roi_color, sorted_eyes, (x, y, w, h)]


# stolen from: ttps://stackoverflow.com/a/71701023/310196
def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite


def apply_googlyzation(img, single_eye, eye_size_ratio=20, choose=None):
    eye_index = 1

    # scale the eyes to a size about this proprotion of the face height/width
    # tunable to adjust the googley eye size
    ratio = eye_size_ratio

    for [_, eyes, (face_x, face_y, face_width, face_height)] in get_eyes(img):
        face_x_ratio = math.floor((face_width - face_x) / ratio)
        face_y_ratio = math.floor((face_height - face_y) / ratio)

        for (ex, ey, eye_distance, eye_height) in eyes:
            if choose and eye_index not in choose:
                eye_index += 1
                continue

            resized = cv2.resize(single_eye, (eye_distance + face_x_ratio, eye_height + face_y_ratio))

            rotation = random.randint(0, 3)

            if rotation > 0:
                resized = cv2.rotate(resized, [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180][rotation - 1])

            add_transparent_image(img, resized, ex + face_x, ey + face_y)

            eye_index += 1

    return img


def image_googley(source, eye_source=os.path.join(script_path, 'eyes/default.png'), debug=False, eye_size_ratio=20, choose=None):
    img = cv2.imread(source)

    single_eye = cv2.imread(eye_source, cv2.IMREAD_UNCHANGED)

    googly = apply_googlyzation(img, single_eye, eye_size_ratio, choose)

    if debug:
        display(googly, wait=True)
    else:
        target = os.path.dirname(source) + '/' + pathlib.Path(source).stem + '_googled' + os.path.splitext(source)[1]

        cv2.imwrite(target, googly)


def video_googley(source, eye_source=os.path.join(script_path, 'eyes/default.png'), eye_size_ratio=20, debug=False, choose=None):
    cap = cv2.VideoCapture(source)

    single_eye = cv2.imread(eye_source, cv2.IMREAD_UNCHANGED)

    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))

    out = None

    if not debug:
        target = os.path.dirname(source) + '/' + pathlib.Path(source).stem + '_googled' + os.path.splitext(source)[1]

        out = cv2.VideoWriter(target,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while cap.isOpened():
        ret, frame = cap.read()

        small = cv2.resize(frame, (math.floor(frame_width/2), math.floor(frame_height/2)))

        googly = apply_googlyzation(small, single_eye, eye_size_ratio, choose)

        if debug:
            display(googly)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            out.write(googly)

    cap.release()

    if not debug:
        out.release()


parser = argparse.ArgumentParser(description='Make people googley.')

parser.add_argument('--path', type=str, help='The path of the image to make googley', required=False)

parser.add_argument('--video', dest='video', action='store_true',  help='If this is a video or not',
                    required=False)

parser.add_argument('--webcam', dest='webcam', type=int,  help='If this is a webcam or not',
                    required=False)

parser.add_argument('--eye', dest='eyes', action='append', type=int, help='Chooses which set of eyes to use. Each eye is given a value from 1 to N',
                    required=False)

parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Doesnt write the file, just pops open a window with it')

parser.add_argument('--type', dest='type', type=str, choices={"default", "shape1", "shape2"}, default='default',
                    help='Choose your eye type!')

parser.add_argument('--scale', dest='scale', type=int,
                    default=20,
                    help='Scale the eyes. Smaller number is larger')

args = parser.parse_args()

if args.video:
    video_googley(
        source=args.path,
        eye_size_ratio=args.scale,
        choose=args.eyes,
        debug=args.debug,
        eye_source=os.path.join(script_path, "eyes", args.type + ".png")
    )
if args.webcam:
    video_googley(
        source=args.webcam,
        eye_size_ratio=args.scale,
        debug=True,
        choose=args.eyes,
        eye_source=os.path.join(script_path, "eyes", args.type + ".png")
    )
else:
    image_googley(
        source=args.path,
        debug=args.debug,
        eye_size_ratio=args.scale,
        choose=args.eyes,
        eye_source=os.path.join(script_path, "eyes", args.type + ".png")
    )
