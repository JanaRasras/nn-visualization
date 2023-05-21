"""
Generate a autoencoder neural network visualization
"""
## Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

# choose a color Palette
BLUE = "#04253a"
GREEN = "#4c837a"
TAN = "#e1ddbf"
DPI = 300  # increase the resolution 

# These are the size and dimentions of the layout of the visualization.(Its like a dictionary of parameters for the visualization).
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 9
RIGHT_BORDER = 0.7
LEFT_BORDER = 0.7
TOP_BORDER = 0.8
BOTTOM_BORDER = 0.6

N_IMAGE_PIXEL_COLS = 64
N_IMAGE_PIXEL_ROWS = 48
N_NODES_BY_LAYER = [10, 7, 5, 8]

INPUT_IMAGE_BOTTOM = 5
INPUT_IMAGE_HEIGHT = 0.25 * FIGURE_HEIGHT
ERROR_IMAGE_SCALE = 0.7
ERROR_GAP_SCALE = 0.3
BETWEEN_LAYER_SCALE = 0.8
BETWEEN_NODE_SCALE = 0.4


def main():
    p = construct_parameters()
    fig,ax_boss = create_background(p)
    p = find_node_image_size(p)
    print("node image dim", p["node_image"])


def construct_parameters():

    aspect_ratio = N_IMAGE_PIXEL_COLS / N_IMAGE_PIXEL_ROWS

    parameters = {}  # empty dict.
    # This figure as a whole
    parameters["figure"] = {
        "height": FIGURE_HEIGHT,
        "width":FIGURE_WIDTH,
    }

    
    parameters["inputs"] = {
        "n_cols": N_IMAGE_PIXEL_COLS,
        "n_rows": N_IMAGE_PIXEL_ROWS,
        "aspect_ratio": aspect_ratio,
        "image": {
            "bottom": INPUT_IMAGE_BOTTOM,
            "height": INPUT_IMAGE_HEIGHT,
            "width": INPUT_IMAGE_HEIGHT * aspect_ratio,
        }
    }

    parameters["network"] = {
        "n_nodes": N_NODES_BY_LAYER,
        "n_layers": len(N_NODES_BY_LAYER),
        "max_nodes": np.max(N_NODES_BY_LAYER),
    }

    # Individual node image
    parameters["node_image"] = {
        "height": 0,
        "width": 0,
    }
    
    parameters["error_image"] = {
        "left" : 0,
        "bottom": 0,
        "width" : parameters["inputs"]["image"]["width"] * ERROR_IMAGE_SCALE,
        "height" : parameters["inputs"]["image"]["height"]* ERROR_IMAGE_SCALE,
    }

    parameters["gap"] = {
        "right_border":RIGHT_BORDER,
        "left_border":LEFT_BORDER,
        "bottom_border":BOTTOM_BORDER,
        "top_border": TOP_BORDER,
        "between_layer":0,
        "between_layer_scale":BETWEEN_LAYER_SCALE,
        "between_node":0,
        "between_node_scale": BETWEEN_NODE_SCALE,
        "error_gap_scale":ERROR_GAP_SCALE,
    }

    return parameters

def find_node_image_size(p):
    # First assume height is the limiting factor
    total_space_to_fill = (
        p["figure"]["height"]
        - p["gap"]["bottom_border"]
        - p["gap"]["top_border"]
    )
    height_constrained_by_height = (
        total_space_to_fill/(
        p["network"]["max_nodes"]
        + (p["network"]["max_nodes"] - 1)
        * p["gap"]["between_node_scale"]
        )
    )
        # Second assume width is the limiting factor.
    total_space_to_fill = (
        p["figure"]["width"]
        - p["gap"]["left_border"]
        - p["gap"]["right_border"]
        - 2 * p["inputs"]["image"]["width"]
    )

    width_constrained_by_width = (
        total_space_to_fill / (
           p["network"]["n_layers"]
           + (p["network"]["n_layers"] + 1)
           * p["gap"]["between_layer_scale"]
        )
    )

    # Figure out what the height would be for this width.
    height_constrained_by_width = (
        width_constrained_by_width
        / p["inputs"]["aspect_ratio"]
    )
    
    # see which constrain is more restrictive
    p["node_image"]["height"] = np.minimum(
        height_constrained_by_width,
        height_constrained_by_height)
    p["node_image"]["width"] = (
        p["node_image"]["height"]
        * p["inputs"]["aspect_ratio"]
    )
    return p



def create_background(p):
    fig = plt.figure(
        edgecolor=TAN,
        facecolor=GREEN,
        figsize = (p["figure"]["width"], p["figure"]["height"]),
        linewidth=4,
    )
    ax_boss = fig.add_axes((0, 0, 1, 1), facecolor="none")
    ax_boss.set_xlim(0, 1)
    ax_boss.set_ylim(0, 1)

    return fig, ax_boss



def save_nn_viz(fig, postfix = "0"):
    base_name = "nn_viz"
    filename = base_name + postfix + ".png"
    fig.savefig(filename,
                edge_color = fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                )
    dpi = DPI,







if __name__ == "__main__":
    main()