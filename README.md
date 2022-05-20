Auto Derpifier
---


This is just some crappy python from https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/ adjusted to try to auto add derp emoji eyes to a picture. It tries to detect a face with a haar cascade, then find eyes within that face.  For each eye it tries to randomly rotate the googley eye and then merges the images together.

It can work (poorly) on images, videos, even webcams.

Use different shapes! Derp all your friends!

<img width="1559" alt="image" src="https://user-images.githubusercontent.com/1799346/168400525-363600a6-aaac-4f75-9420-4543a3c0a35e.png">

```
$ googly --help
usage: derp.py [-h] [--path PATH] [--video] [--webcam WEBCAM] [--eye EYES] [--debug] [--type {shape1,default,shape2}] [--scale SCALE]

Make people googley.

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           The path of the image to make googley
  --video               If this is a video or not
  --webcam WEBCAM       If this is a webcam or not
  --eye EYES            Chooses which set of eyes to use. Each eye is given a value from 1 to N
  --debug               Doesnt write the file, just pops open a window with it
  --type {shape1,default,shape2}
                        Choose your eye type!
  --scale SCALE         Scale the eyes. Smaller number is larger
```

Installation:

```
pip3 install opencv-python
```
