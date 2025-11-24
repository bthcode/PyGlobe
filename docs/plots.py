import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_camera_view():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # Axis arrows
    ax.arrow(0, 0, 1.2, 0, width=0.01, head_width=0.06)   # +X
    ax.arrow(0, 0, 0, 1.2, width=0.01, head_width=0.06)   # +Z
    ax.arrow(0, 0, 0.8, -0.8, width=0.01, head_width=0.06)  # +Y

    # Labels
    ax.text(1.25, 0.05, "X", fontsize=14, color="red")
    ax.text(0.05, 1.25, "Z (North)", fontsize=14, color="green")
    ax.text(0.82, -0.85, "Y", fontsize=14, color="blue")

    # Origin text
    ax.text(0.25, 0.25, "Origin\n(Earth center)", fontsize=10)


    # Earth circle
    earth = plt.Circle((0, 0), 1.0, color='lightblue', alpha=0.3)
    ax.add_patch(earth)

    # Camera position
    cam_x, cam_y = -1.75, 0
    ax.plot(cam_x, cam_y, 'o', color='purple')
    ax.text(cam_x, cam_y + 0.5, "Camera in \nSperical Coords", fontsize=12, color="purple")

    ax.text( -1.05, 0.1, "Look At", fontsize=10, color='red')

    # FOV cone
    angle = np.radians(22.5)
    cone_left  = np.array([cam_x + np.cos(angle)+0.25,  np.sin(angle)])
    cone_right = np.array([cam_x + np.cos(angle)+0.25, -np.sin(angle)])

    ax.plot([cam_x, cone_left[0]], [cam_y, cone_left[1]], color='orange', alpha=0.6)
    ax.plot([cam_x, cone_right[0]], [cam_y, cone_right[1]], color='orange', alpha=0.6)
    ax.plot([cone_left[0], cone_right[0]], [cone_left[1], cone_right[1]], color='orange')

    ax.text(-1.5, -0.5, "View Frustum (FOV 45°)", fontsize=9, color='orange')

    # Camera → Earth line
    ax.arrow(cam_x, cam_y, 1, 0, width=0.01, head_width=0.06, color='red')

    # Labels
    ax.text(-0.1, -1.1, "Earth", fontsize=12, color="blue")

    # Formatting
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    plt.title("Camera View Geometry")
    plt.savefig('view_geometry.png')

plot_camera_view()

