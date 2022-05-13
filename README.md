Auto Derpifier
---


This is just some crappy python from https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/ adjusted to try to auto add derp emoji eyes to a picture. It tries to detect a face with a haar cascade, then find eyes within that face.  For each eye it tries to randomly rotate the googley eye and then merges the images together.

<img width="1559" alt="image" src="https://user-images.githubusercontent.com/1799346/168400525-363600a6-aaac-4f75-9420-4543a3c0a35e.png">

```
usage: derp.py [-h] [--debug] [--scale SCALE] path

Make people googley.

positional arguments:
  path           the path of the image to google

optional arguments:
  -h, --help     show this help message and exit
  --debug        doesnt write the file, just pops open a window with it
  --scale SCALE  scale the eyes
```

Installation:

```
pip3 install opencv-python
```
