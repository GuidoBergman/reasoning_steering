DEFAULT_PROMPT_SHORT_PREFIX = "Below are pairs of matrices. There is a mapping which operates on each input to give the output, only one mapping applies to all matrices. Review the matrices to learn that mapping and then estimate the missing output for the final input matrix.\n"

DEFAULT_PROMPT_SHORT_SUFFIX = "\nYour anwser must contain ONLY your predicted output in np.array format, and no preamble, no prefix, and no punctuation."


DEFAULT_PROMPT_LONG_PREFIX = """
# PROJECT INSTRUCTIONS

The training data consists of pairs of input and output grids, presented as numpy arrays of varying shapes.
Your task is to discover the single mapping which converts each input grid to its corresponding output grid and apply that to the test input, arriving at a test output.

## 1. OBSERVE AND HYPOTHESISE THE MAPPING LOGIC FOR ALL TRAINING PAIRS

When building your hypotheses on the mappings, be aware of the following common transformations:

    Grid Expansion and Repetition (Tiling):
    - Simply expand the grid and repeat (tile) the input grid into the output grid
    Symmetry and Mirroring (flipping):
    - Horizontally or vertically
    Propagation of patterns:
    - Identify non-zero clusters or shapes in the input grid and propagating them in the output. Proceeding horizontally, vertically or diagonally.
    Mathematical Operations:
    - Incrementing values, taking modulo, or performing addition.
    Color/Value Substitution:
    - Values in the input grid replaced with different values in the output grid, often changing all instances of one number to another
    Shape Detection and Transformation:
    - Identifying geometric shapes in the input grid and applying transformations such as rotation, scaling, flipping, translation and/or overlapping.
    Grid Segmentation:
    - Divide the input grid into sections and apply transformations to each section.
    Boundary Detection and Fill:
    - Identify the boundaries of shapes or patterns and fill them with specific values. This sometimes involved propagating values from the edges inward.
    Connectivity-based Transformations:
    - Using connected component analysis to identify and transform groups of connected cells.
    Rule-based Transformations:
    - Applying specific rules based on the arrangement of values in the input grid. These rules often considered the neighboring cells of each position.
    Coordinate-based Transformations:
    - Using the coordinates of cells to determine how they should be transformed or moved in the output grid.
    When the pattern is more complex than originally assumed:
    - Review all training pairs again and try to describe the transformation in plain language

Use these patterns to guide your own hypotheses on the training data.


## 2. THE DATA

"""

DEFAULT_PROMPT_LONG_SUFFIX = """
## 3. PREDICT THE OUTPUT GRID FOR THE TEST INPUT GRID

Your anwser must contain ONLY your predicted output in np.array format, and no preamble, no prefix, and no punctuation."
"""