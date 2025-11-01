# PyGlobe
3D Globe in Python / Qt / OpenGL

![Screenshot](globe.png?raw=true "Example Screenshot")


# Status
This is just an idea, so far

# Code Index
- dynamic.py - main app, displays an opengl globe with map tiles
- tile_fetcher.py - threaded qt clss for fetching tiles
- test_object_loading  - test code for loading and displaying models
    - triangle : simple rotating triangle
    - satellite : load and rotate a satellite model
    - sample_object_project : models integrated into a project, satellites orbit earth
