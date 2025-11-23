PyGlobe
-------
3D Globe in Python / Qt / OpenGL

![Screenshot](globe.png?raw=true "Example Screenshot")


Status
------
- Basic functionality tested by exactly one user on one computer

Installation
------------

- NOTE: This has been tested with python 3.13.9
- Optional CONDA setup:

```
conda create --name pyglobe python=3.13
conda activate pyglove
```

- Install:

```
pip install -e .
```

- NOTE: Will install pyside6, pyopengl, numpy, requests, pillow

Usage
-----

```
cd example/
python example.py
```

Code Overview
-------------

- globe.py - Primary Widget
- tile_fetcher.py - threaded qt clss for fetching tiles
- scene.py - load and manage scene objects
- coord_utils.py - Coordinate transform utilities

