
"""
Generate a autoencoder neural network visualization
"""
## Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from PIL import Image

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
    p = find_between_layer_gap(p)
    p = find_between_node_gap(p)
    p = find_error_image_position(p)


    add_input_image(fig, p)
    for i_layer in range(p["network"]["n_layers"]):
        add_node_images(fig, i_layer, p)
    add_output_image(fig, p)
    add_error_image(fig, p)
    save_nn_viz(fig, postfix="24_error_image")

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

def find_between_layer_gap(p):
    horizontal_gap_total = (
        p["figure"]["width"]
        - 2 * p["inputs"]["image"]["width"]
        - p["network"]["n_layers"] * p["node_image"]["width"]
        - p["gap"]["left_border"]
        - p["gap"]["right_border"]
    )
    n_horizontal_gaps = p["network"]["n_layers"] + 1
    p["gap"]["between_layer"] = horizontal_gap_total / n_horizontal_gaps


    # vertical
    vertical_gap_total = (
        p["figure"]["height"]
        - p["gap"]["top_border"]
        - p["gap"]["bottom_border"]
        - p["network"]["max_nodes"]
        * p["node_image"]["height"]
    )
    n_vertical_gaps = p["network"]["n_layers"] - 1
    p["gap"]["between_nodes"] = vertical_gap_total / n_vertical_gaps



    return p

def save_nn_viz(fig, postfix = "0"):
    base_name = "nn_viz"
    filename = base_name + postfix + ".png"
    fig.savefig(filename,
                edge_color = fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                )
    dpi = DPI,

def find_error_image_position(p):
    """
    Where exactly should the error image be positioned?
    """
    p["error_image"]["bottom"] = (
        p["inputs"]["image"]["bottom"]
        - p["inputs"]["image"]["height"]
        * p["gap"]["error_gap_scale"]
        - p["error_image"]["height"]
    )
    error_image_center = (
        p["figure"]["width"]
        - p["gap"]["right_border"]
        - p["inputs"]["image"]["width"] / 2
    )
    p["error_image"]["left"] = (
        error_image_center
        - p["error_image"]["width"] / 2
    )
    return p

def add_input_image(fig, p):
    absolute_pos = (
        p["gap"]["left_border"],
        p["inputs"]["image"]["bottom"],
        p["inputs"]["image"]["width"],
        p["inputs"]["image"]["height"]
    )
    ax_input = add_image_axes(fig, p, absolute_pos)
    add_filler_image(
        ax_input,
        p["inputs"]["n_rows"],
        p["inputs"]["n_cols"]
    )
        
def add_filler_image(ax, n_im_rows, n_im_cols):
    fill_patch = np.random.sample(size=(n_im_rows, n_im_cols))
    ax.imshow(fill_patch, cmap="inferno")

def add_image_axes(fig, p, absolute_pos):

    scaled_pos = (
        absolute_pos[0] / p["figure"]["width"],
        absolute_pos[1] / p["figure"]["height"],
        absolute_pos[2] / p["figure"]["width"],
        absolute_pos[3] / p["figure"]["height"])
    ax = fig.add_axes(scaled_pos)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(
        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.spines["top"].set_color(TAN)
    ax.spines["bottom"].set_color(TAN)
    ax.spines["left"].set_color(TAN)
    ax.spines["right"].set_color(TAN)
    return ax


def find_between_node_gap(p):
   
    vertical_gap_total = (
        p["figure"]["height"]
        - p["gap"]["top_border"]
        - p["gap"]["bottom_border"]
        - p["network"]["max_nodes"]
        * p["node_image"]["height"]
    )
    n_vertical_gaps = p["network"]["max_nodes"] - 1
    p["gap"]["between_node"] = vertical_gap_total / n_vertical_gaps
    return p
   
def add_node_images(fig, i_layer, p):
   
    node_image_left = (
        p["gap"]["left_border"]
        + p["inputs"]["image"]["width"]
        + i_layer * p["node_image"]["width"]
        + (i_layer + 1) * p["gap"]["between_layer"]
    )
    n_nodes = p["network"]["n_nodes"][i_layer]
    total_layer_height = (
        n_nodes * p["node_image"]["height"]
        + (n_nodes - 1) * p["gap"]["between_node"]
    )
    layer_bottom = (p["figure"]["height"] - total_layer_height) / 2
    for i_node in range(n_nodes):
        node_image_bottom = (
            layer_bottom + i_node * (
                p["node_image"]["height"] + p["gap"]["between_node"]))

        absolute_pos = (
            node_image_left,
            node_image_bottom,
            p["node_image"]["width"],
            p["node_image"]["height"])
        ax = add_image_axes(fig, p, absolute_pos)
        add_filler_image(
            ax,
            p["inputs"]["n_rows"],
            p["inputs"]["n_cols"],
        )

def add_output_image(fig, p):
    output_image_left = (
        p["figure"]["width"]
        - p["inputs"]["image"]["width"]
        - p["gap"]["right_border"]
    )
    absolute_pos = (
        output_image_left,
        p["inputs"]["image"]["bottom"],
        p["inputs"]["image"]["width"],
        p["inputs"]["image"]["height"])
    ax_output = add_image_axes(fig, p, absolute_pos)
    add_filler_image(
       ax_output,
       p["inputs"]["n_rows"],
       p["inputs"]["n_cols"],
    )

def add_error_image(fig, p):
    absolute_pos = (
        p["error_image"]["left"],
        p["error_image"]["bottom"],
        p["error_image"]["width"],
        p["error_image"]["height"])
    ax_error = add_image_axes(fig, p, absolute_pos)
    add_filler_image(
       ax_error,
       p["inputs"]["n_rows"],
       p["inputs"]["n_cols"],
    )

if __name__ == "__main__":
    main()



