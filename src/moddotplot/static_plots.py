from plotnine import (
    ggsave,
    ggplot,
    aes,
    geom_histogram,
    scale_color_discrete,
    element_blank,
    theme,
    xlab,
    scale_fill_manual,
    scale_color_cmap,
    coord_cartesian,
    ylab,
    scale_x_continuous,
    scale_y_continuous,
    geom_tile,
    geom_segment,
    geom_col,
    geom_rect,
    coord_fixed,
    facet_grid,
    labs,
    element_line,
    element_text,
    theme_light,
    geom_blank,
    annotate,
    element_rect,
    coord_flip,
    theme_minimal,
    geom_raster,
    geom_area
)
import svgutils.transform as sg
import cairosvg
import pyranges as pr
import pandas as pd
import numpy as np
import glob
from PIL import Image
import patchworklib as pw
import math
import os
import xml.etree.ElementTree as ET
import sys
import re
from moddotplot.parse_fasta import printProgressBar
from lxml import etree
from pygenometracks.utilities import get_region
import matplotlib.pyplot as plt
from moddotplot.const import (
    DIVERGING_PALETTES,
    QUALITATIVE_PALETTES,
    SEQUENTIAL_PALETTES,
)
from typing import List
from palettable.colorbrewer import qualitative, sequential, diverging
import logging
import pyBigWig

# Set log level BEFORE importing pygenometracks
for name in logging.root.manager.loggerDict:
    if name.startswith("pygenometracks"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False  # Don't pass to root logger

# Also make sure the root logger isn’t outputting debug messages
logging.basicConfig(level=logging.CRITICAL)

from pygenometracks.tracksClass import PlotTracks

def is_plot_empty(p):
    # Check if the plot has data or any layers
    return len(p.layers) == 0 and p.data.empty


def check_pascal(single_val, double_val):
    try:
        if len(single_val) == 2:
            assert len(double_val) == 1
        elif len(single_val) == 3:
            assert len(double_val) == 3
        elif len(single_val) == 4:
            assert len(double_val) == 6
        elif len(single_val) == 5:
            assert len(double_val) == 10
        elif len(single_val) == 6:
            assert len(double_val) == 15
        elif len(single_val) == 0:
            assert len(double_val) == (1 or 3 or 6 or 10 or 15)
    except AssertionError as e:
        print(
            f"Missing bed files required to create grid. Please verify all bed files are included."
        )
        sys.exit(8)


def generate_ini_file(
    bedfile, ininame, chrom, color_value="bed_rgb", x_axis=True, display="collapsed"
):
    try:
        thing = chrom.split(":")[0]
        sections = [
            "[spacer]",
            "# height of space in cm (optional)",
            "height = 0.5",
            "",
            f"[{thing}]",
            f"file = {bedfile}",
            f"Title=",
            "height = 1",
            f"display = {display}",
            f"color = {color_value}",
            "labels = false",
            "fontsize = 10",
            "file_type = bed",
        ]

        if x_axis:
            sections.insert(0, "[x-axis]")

        ini_content = "\n".join(sections)
        with open(f"{ininame}.ini", "w") as file:
            file.write(ini_content)
        print(f"Successfully generated {ininame}.ini\n")
        return f"{ininame}.ini"
    except Exception as err:
        print(f"Error producing ini file: {err}\n")
        return None

def make_svg_background_transparent(svg_path, output_path=None):
    """
    Makes the background of an SVG file transparent by removing/modifying background fills.

    Args:
        svg_path: Path to input SVG file
        output_path: Path to output SVG file (if None, overwrites input)
    """
    import xml.etree.ElementTree as ET
    import re

    if output_path is None:
        output_path = svg_path

    # Parse the SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Define SVG namespace
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Remove background rectangles/paths that cover the entire canvas
    # Get SVG dimensions for comparison
    width = root.get("width", "0")
    height = root.get("height", "0")

    # Extract numeric values
    width_num = float(re.sub(r"[a-zA-Z%]+", "", width)) if width != "0" else 0
    height_num = float(re.sub(r"[a-zA-Z%]+", "", height)) if height != "0" else 0

    # Find and modify background elements
    elements_to_modify = []

    # Check all paths, rectangles, and other elements
    for elem in root.iter():
        if (
            elem.tag.endswith("path")
            or elem.tag.endswith("rect")
            or elem.tag.endswith("polygon")
        ):
            # Check if this element has a background-like fill
            style = elem.get("style", "")
            fill = elem.get("fill", "")

            # Look for background colors (light colors, white, etc.)
            background_colors = [
                "#ffffff",
                "#f0ffff",
                "white",
                "lightblue",
                "lightgray",
                "lightgrey",
            ]

            is_background = False
            current_fill = None

            if "fill:" in style:
                # Extract fill from style
                fill_match = re.search(r"fill:\s*([^;]+)", style)
                if fill_match:
                    current_fill = fill_match.group(1).strip()
            elif fill:
                current_fill = fill

            if current_fill and any(
                bg_color in current_fill.lower() for bg_color in background_colors
            ):
                is_background = True

            # For paths, check if it covers a large area (likely background)
            if elem.tag.endswith("path"):
                d = elem.get("d", "")
                # Simple heuristic: if path starts at 0,0 and covers large area, it's likely background
                if "M 0" in d and current_fill:
                    is_background = True

            # For rectangles, check if it covers the full canvas
            if elem.tag.endswith("rect"):
                x = float(elem.get("x", 0))
                y = float(elem.get("y", 0))
                w = float(elem.get("width", 0))
                h = float(elem.get("height", 0))

                # If rectangle covers most/all of the canvas, it's likely background
                if x <= 1 and y <= 1 and w >= width_num * 0.9 and h >= height_num * 0.9:
                    is_background = True

            if is_background:
                elements_to_modify.append(elem)

    # Modify the background elements
    for elem in elements_to_modify:
        style = elem.get("style", "")

        if "fill:" in style:
            # Replace fill in style
            new_style = re.sub(r"fill:\s*[^;]+", "fill: transparent", style)
            elem.set("style", new_style)
        elif elem.get("fill"):
            # Replace fill attribute
            elem.set("fill", "transparent")

    # Save the modified SVG
    tree.write(output_path, encoding="unicode", xml_declaration=True)


def make_all_svg_backgrounds_transparent(directory):
    """
    Makes all SVG files in a directory have transparent backgrounds.
    """

    svg_files = glob.glob(os.path.join(directory, "*.svg"))

    for svg_file in svg_files:
        make_svg_background_transparent(svg_file)

    print(f"Processed {len(svg_files)} SVG files")

def reverse_pascal(double_vals):
    if len(double_vals) == 1:
        return 2
    elif len(double_vals) == 3:
        return 3
    elif len(double_vals) == 6:
        return 4
    elif len(double_vals) == 10:
        return 5
    elif len(double_vals) == 15:
        return 6
    else:
        sys.exit(9)


# Hardcoding for now, I have the formula.... I'm just lazy
def transpose_order(double_vals):
    if len(double_vals) == 1:
        return [0]
    elif len(double_vals) == 3:
        return [0, 1, 2]
    elif len(double_vals) == 6:
        return [0, 1, 3, 2, 4, 5]
    elif len(double_vals) == 10:
        return [0, 1, 4, 2, 5, 7, 3, 6, 8, 9]
    elif len(double_vals) == 15:
        return [0, 1, 5, 2, 6, 9, 3, 7, 10, 12, 4, 8, 11, 13, 14]


def check_st_en_equality(df):
    unequal_rows = df[(df["q_st"] != df["r_st"]) | (df["q_en"] != df["r_en"])]
    unequal_rows.loc[:, ["q_en", "r_en", "q_st", "r_st"]] = unequal_rows[
        ["r_en", "q_en", "r_st", "q_st"]
    ].values

    df = pd.concat([df, unequal_rows], ignore_index=True)

    return df


def make_k(vals):
    return [number / 1000 for number in vals]


def make_m(vals):
    return [ f"{number / 1e6:>4.2f}" for number in vals]

def make_g(vals):
    return [number / 1e9 for number in vals]


def make_scale(vals: list) -> list:
    scaled = [number for number in vals]
    if scaled[-1] < 200000:
        return make_k(scaled)
    elif scaled[-1] > 200000000:
        return make_g(scaled)
    else:
        return make_m(scaled)


def get_colors(sdf, ncolors, is_freq, custom_breakpoints):
    assert ncolors > 2 and ncolors < 12
    try:
        bot = math.floor(min(sdf["perID_by_events"]))
    except ValueError:
        bot = 0
    top = 100.0
    interval = (top - bot) / ncolors
    breaks = []
    if is_freq:
        breaks = np.unique(
            np.quantile(sdf["perID_by_events"], np.arange(0, 1.01, 1 / ncolors))
        )
    else:
        breaks = [bot + i * interval for i in range(ncolors + 1)]
    if custom_breakpoints:
        np.asarray(custom_breakpoints, dtype=np.float64)
    labels = np.arange(len(breaks) - 1)
    # corner case of only one %id value
    if len(breaks) == 1:
        return pd.factorize([1] * len(sdf["perID_by_events"]))[0]
    else:
        tmp = pd.cut(
            sdf["perID_by_events"], bins=breaks, labels=labels, include_lowest=True
        )
        return tmp


# TODO: Remove pandas dependency
def read_df_from_file(file_path):
    data = pd.read_csv(file_path, delimiter="\t")
    return data

def read_df(
    pj,
    palette,
    palette_orientation,
    is_freq,
    custom_colors,
    custom_breakpoints,
    from_file,
):
    df = ""
    if from_file is not None:
        df = from_file
    else:
        data = pj[0]
        df = pd.DataFrame(data[1:], columns=data[0])
    hexcodes = []
    new_hexcodes = []
    if palette in DIVERGING_PALETTES:
        function_name = getattr(diverging, palette)
        hexcodes = function_name.hex_colors
        if palette_orientation == "+":
            palette_orientation = "-"
        else:
            palette_orientation = "+"
    elif palette in QUALITATIVE_PALETTES:
        function_name = getattr(qualitative, palette)
        hexcodes = function_name.hex_colors
    elif palette in SEQUENTIAL_PALETTES:
        function_name = getattr(sequential, palette)
        hexcodes = function_name.hex_colors
    else:
        print(f"Palette {palette} not found. Defaulting to Spectral_11.\n")
        function_name = getattr(diverging, "Spectral_11")
        palette_orientation = "-"
        hexcodes = function_name.hex_colors

    if palette_orientation == "-":
        new_hexcodes = hexcodes[::-1]
    else:
        new_hexcodes = hexcodes

    if custom_colors:
        new_hexcodes = custom_colors

    ncolors = len(new_hexcodes)
    # Get colors for each row based on the values in the dataframe
    df["discrete"] = get_colors(df, ncolors, is_freq, custom_breakpoints)
    # Rename columns if they have different names in the dataframe
    if "query_name" in df.columns or "#query_name" in df.columns:
        df.rename(
            columns={
                "#query_name": "q",
                "query_start": "q_st",
                "query_end": "q_en",
                "reference_name": "r",
                "reference_start": "r_st",
                "reference_end": "r_en",
            },
            inplace=True,
        )

    # Calculate the window size
    try:
        window = max(df["q_en"] - df["q_st"])
    except ValueError:
        window = 0

    # Calculate the position of the first and second intervals
    df["first_pos"] = df["q_st"] / window
    df["second_pos"] = df["r_st"] / window

    return df


def generate_breaks(min_number, max_number, min_breaks=5, max_breaks=9):
    # Determine the order of magnitude
    difference = max_number - min_number

    magnitude = 10 ** int(math.floor(math.log10(difference)))
    threshold = math.ceil(difference / magnitude)

    while threshold > max_breaks:
        magnitude *= 2
        threshold = math.ceil(difference / magnitude)

    while threshold < min_breaks:
        magnitude /= 2
        threshold = math.ceil(difference / magnitude)

    # Round down min_number to the nearest multiple of magnitude
    min_aligned = int(min_number // magnitude * magnitude)

    # Generate breakpoints
    upper_bound = int(min_aligned + (threshold + 1) * magnitude)
    breaks = list(range(min_aligned, upper_bound, int(magnitude)))

    return breaks

def sum_over_ranges(df_ranges, df_values, value_cols):
    
    gr_ranges = pr.PyRanges(
        df_ranges.rename(columns={"q": "Chromosome",
                                  "q_st": "Start",
                                  "q_en": "End"})[["Chromosome", "Start", "End"]]
    )
    
    gr_values = pr.PyRanges(
        df_values.assign(
            Start=df_values["start"],
            End=df_values["end"]
        ).rename(columns={"chromosome": "Chromosome"})[["Chromosome", "Start", "End"] + value_cols]
    )

    joined = gr_ranges.join(gr_values)
    agg = joined.df.groupby(["Chromosome", "Start", "End"], as_index=False)[value_cols].agg(["sum", "median"])
    agg.columns = [ f"{col}_{stat}" if stat != "" else col for col, stat in agg.columns ]
    return agg

def make_tri(
    sdf,
    mdf,
    cdf,
    title_name,
    palette,
    palette_orientation,
    colors,
    breaks,
    xlim,
    num_ticks,
    deraster,
    width,
    chromosome
):
    # Select the color palette
    if hasattr(diverging, palette):
        function_name = getattr(diverging, palette)
    elif hasattr(qualitative, palette):
        function_name = getattr(qualitative, palette)
    elif hasattr(sequential, palette):
        function_name = getattr(sequential, palette)
    else:
        function_name = diverging.Spectral_11  # Default palette
        palette_orientation = "-"

    hexcodes = function_name.hex_colors

    # Adjust palette orientation
    if palette in diverging.__dict__:
        palette_orientation = "-" if palette_orientation == "+" else "+"

    new_hexcodes = hexcodes[::-1] if palette_orientation == "-" else hexcodes
    if colors:
        new_hexcodes = colors  # Override colors if provided
    if not xlim:
        xlim = 0

    # Determine maximum genomic position for scaling
    min_val = max(sdf["q_st"].min(), sdf["r_st"].min())
    max_val = max(sdf["q_en"].max(), sdf["r_en"].max(), xlim)
    
    # If user provides breaks, convert to ints
    if not breaks:
        breaks = generate_breaks(int(min_val), int(max_val), 4, 4)
    else:
        [int(x) for x in breaks]

    xlim = xlim or 0
    # Compute window size (handling exceptions)
    try:
        window = max(sdf["q_en"] - sdf["q_st"])
    except ValueError:  # Empty dataframe case
        return ggplot(aes(x=[], y=[])) + theme_minimal()

    # Determine axis label scale based on genomic position size
    if max_val < 200_000:
        x_label = "Genomic Position (Kbp)"
    elif max_val < 200_000_000:
        x_label = "Genomic Position (Mbp)"
    else:
        x_label = "Genomic Position (Gbp)"

    font_size=14
    apex = max(sdf["r_st"])
    midpoint = (min(sdf["q_st"]) + max(sdf["r_en"])) / 2

    sdf["xstart"] = sdf["q_st"] + window / 2
    sdf["ystart"] = sdf["r_st"] + window / 2

    tri = (
        ggplot(sdf)
        + geom_raster( aes(x="xstart", y="ystart", fill="discrete", height=window, width=window), alpha=1.0)
        + geom_segment(x = min_val, xend = midpoint, y = min(sdf["r_st"]), yend=midpoint, linetype="dotted", size=0.5) 
        + geom_segment(x = midpoint, xend = max_val, y = midpoint, yend = min(sdf["r_st"]), linetype="dotted", size=0.5) 
        + scale_fill_manual(values=new_hexcodes, guide=False)
        + scale_color_discrete(guide=False)
        + scale_x_continuous(
            labels=make_scale, limits=[min_val - 1, max_val + 1], breaks=breaks
        )
        + scale_y_continuous(
            labels=make_scale, limits=[min_val - 1, max_val + 1], breaks=breaks
        )
        + coord_fixed(ratio=1, expand=False)
        + labs(x=x_label, y="", title=title_name)
        + theme(
            legend_position="none",
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            plot_background=element_blank(),
            panel_background=element_blank(),
            axis_text=element_text(family=["DejaVu Sans"], size=font_size),
            axis_text_y=element_text(family=["DejaVu Sans"], size=font_size, color="black"), # make invisible, but keep spacing consistent
            axis_line_x=element_line(),
            axis_line_y=element_line(color="white"), # make invisible
            axis_ticks_major_x=element_line(),
            axis_ticks_major_y=element_blank(),
            axis_ticks_major=element_line(size=(width)),
            title=element_text(size=(width * 1.4), hjust=0.5),
            axis_title_x=element_blank()
            #axis_text_y=element_blank()
        )
    )

    # percent methylation track
    region_mdf = mdf[ ( mdf["chromosome"] == chromosome ) & ( mdf["start"] >= min_val ) & ( mdf["start"] <= max_val ) ]
    region_mod_df = sum_over_ranges(sdf, region_mdf, ["n_mod", "n_valid_cov"])
    region_mod_df["percent"] = 100.0 * region_mod_df["n_mod_sum"] / region_mod_df["n_valid_cov_sum"]

    m = (
        ggplot(region_mod_df, aes(xmin="Start", xmax="End", ymin=0, ymax="percent"))
        + geom_rect(fill="black")
        + scale_x_continuous( labels=make_scale, limits=[min_val, max_val], breaks=breaks )
        + scale_y_continuous( limits=[0, 100], breaks=[0, 100], labels=["   0", "   100"] )
        + labs(x=x_label, y="", title="")
        + coord_cartesian(expand=False)
        + theme(
            legend_position="none",
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            plot_background=element_blank(),
            panel_background=element_blank(),
            axis_text=element_text(family=["DejaVu Sans"], size=font_size),
            axis_line_x=element_line(),
            axis_line_y=element_line(),
            axis_ticks_major_x=element_line(),
            axis_ticks_major_y=element_line(),
            axis_ticks_major=element_line(size=(width)),
            title=element_text(size=(width * 1.4), hjust=0.5),
            axis_title_x=element_blank()
            #axis_text_y=element_blank(),
        )
    )

    # cenpa track
    region_cenpa = sum_over_ranges(sdf, cdf, [ "value" ])
    max_cenpa = max(region_cenpa['value_median'])

    cenpa = (
        ggplot(region_cenpa, aes(xmin="Start", xmax="End", ymin=0, ymax="value_median"))
        + geom_rect(fill="black")
        + scale_x_continuous( labels=make_scale, limits=[min_val, max_val], breaks=breaks )
        + scale_y_continuous(limits=[0, max_cenpa], breaks=[0, max_cenpa], labels=["  0", f"{max_cenpa:>7.0f}"])
        + labs(x=x_label, y="", title="")
        + coord_cartesian(expand=False)
        + theme(
            legend_position="none",
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            plot_background=element_blank(),
            panel_background=element_blank(),
            axis_text=element_text(family=["DejaVu Sans"], size=font_size),
            axis_line_x=element_line(),
            axis_line_y=element_line(),
            axis_ticks_major_x=element_line(),
            axis_ticks_major_y=element_line(),
            axis_ticks_major=element_line(size=(width)),
            title=element_text(size=(width * 1.4), hjust=0.5),
            axis_title_x=element_text(size=font_size, family=["DejaVu Sans"])
            #axis_text_y=element_blank(),
        )
    )

    return tri, m, cenpa

def rotate_rasterized_tri(svg_path, shift_x, shift_y):
    import xml.etree.ElementTree as ET
    import math, re

    tree = ET.parse(svg_path)
    root = tree.getroot()

    ns = {"svg": "http://www.w3.org/2000/svg"}

    for image in root.findall(".//svg:image", ns):
        href = image.get("{http://www.w3.org/1999/xlink}href", "")
        if href.startswith("data:image/png;base64,"):

            # Recompute size after rotation (45° shrinks effective bounding box)
            w = float(image.get("width", 0))
            h = float(image.get("height", 0))
            new_w = w / math.sqrt(2)
            new_h = h / math.sqrt(2)

            image.set("width", str(new_w))
            image.set("height", str(new_h))

            # Apply rotation + translation
            transform = image.get("transform", "")
            if transform:
                new_transform = f"rotate(45,0,0) translate({shift_x},{shift_y}) {transform}"
            else:
                new_transform = f"rotate(45,0,0) translate({shift_x},{shift_y})"

            image.set("transform", new_transform)

    tree.write(svg_path)

from lxml import etree

def get_svg_size(svg_path):
    """Helper to extract width and height of an SVG in pt units."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(svg_path)
    root = tree.getroot()
    w = root.get("width")
    h = root.get("height")
    return w,h

def parse_size(size_str):
    """Convert '648pt' or '800px' -> float(648)."""
    return float(re.sub(r"[a-zA-Z]+", "", size_str))

def get_svg_root_size(svg_path):
    """
    Extracts width, height from <svg> or its viewBox.
    Does NOT parse deep content (safe for big SVGs).
    """
    parser = etree.XMLParser(huge_tree=True)
    tree = etree.parse(svg_path, parser)
    root = tree.getroot()

    # Try width/height first
    w = root.get("width")
    h = root.get("height")

    # If they exist with units, return them
    if w and h:
        return w, h

    # If they are missing, fallback to viewBox
    viewBox = root.get("viewBox")
    if viewBox:
        _, _, vw, vh = viewBox.split()
        return f"{vw}px", f"{vh}px"

    # Fallback default if SVG doesn't specify anything
    # (rare, but safe)
    return "1000px", "1000px"


def merge_svgs_vertically(svg_paths, output_path):
    """
    Stack SVGs vertically, no spacing, left aligned.
    Safe for huge SVGs. No deep XML traversal.
    """
    roots = []
    sizes = []

    # --- Load all SVGs safely ---
    for path in svg_paths:
        fig = sg.fromfile(path)
        root = fig.getroot()      # GroupElement wrapper
        w, h = get_svg_root_size(path)

        # Convert sizes to numeric px
        w_val = float(w.replace("pt", ""))
        h_val = float(h.replace("pt", ""))

        sizes.append((w_val, h_val))
        roots.append(root)

    # --- Compute final SVG size ---
    total_width = max(w for w, h in sizes)
    total_height = sum(h for w, h in sizes)

    fig = sg.SVGFigure(f"{total_width}px", f"{total_height}px")

    # --- Position each SVG ---
    y_offset = 0
    positioned = []
    for root, (w, h) in zip(roots, sizes):
        # LEFT aligned → x = 0
        # TOP stacked → y = cumulative offset
        root.moveto(0, y_offset)
        y_offset += h
        positioned.append(root)

    # --- Add all SVGs to figure ---
    fig.append(positioned)

    # --- Force valid size attributes for CairoSVG ---
    # This ensures NO "SVG size undefined" error.
    fig.root.set("width",  f"{total_width}px")
    fig.root.set("height", f"{total_height}px")
    fig.root.set("viewBox", f"0 0 {total_width} {total_height}")

    # --- Save merged SVG ---
    fig.save(output_path)

def make_hist(sdf, palette, palette_orientation, custom_colors, custom_breakpoints):
    hexcodes = []
    new_hexcodes = []
    if palette in DIVERGING_PALETTES:
        function_name = getattr(diverging, palette)
        hexcodes = function_name.hex_colors
        if palette_orientation == "+":
            palette_orientation = "-"
        else:
            palette_orientation = "+"
    elif palette in QUALITATIVE_PALETTES:
        function_name = getattr(qualitative, palette)
        hexcodes = function_name.hex_colors
    elif palette in SEQUENTIAL_PALETTES:
        function_name = getattr(sequential, palette)
        hexcodes = function_name.hex_colors
    else:
        function_name = getattr(diverging, "Spectral_11")
        palette_orientation = "-"
        hexcodes = function_name.hex_colors

    if palette_orientation == "-":
        new_hexcodes = hexcodes[::-1]
    else:
        new_hexcodes = hexcodes

    if custom_colors:
        new_hexcodes = custom_colors
    try:
        bot = np.quantile(sdf["perID_by_events"], q=0.001)
    except IndexError:
        bot = 0
    count = sdf.shape[0]
    extra = ""

    if count > 1e6:
        extra = "\n(thousands)"

    p = (
        ggplot(data=sdf, mapping=aes(x="perID_by_events", fill="discrete"))
        + geom_histogram(bins=300)
        + scale_color_cmap(cmap_name="plasma")
        + scale_fill_manual(new_hexcodes)
        + theme_light()
        + theme(text=element_text(family=["DejaVu Sans"]))
        + theme(legend_position="none")
        + coord_cartesian(xlim=(bot, 100))
        + xlab("% Identity Estimate")
        + ylab("# of Estimates{}".format(extra))
    )
    return p

def create_plots(
    sdf,
    mdf,
    cdf,
    directory,
    name_x,
    name_y,
    palette,
    palette_orientation,
    no_hist,
    width,
    dpi,
    is_freq,
    xlim,
    custom_colors,
    custom_breakpoints,
    from_file,
    is_pairwise,
    axes_labels,
    axes_tick_number,
    vector_format,
    deraster,
    annotation,
):
    df = read_df(
        sdf,
        palette,
        palette_orientation,
        is_freq,
        custom_colors,
        custom_breakpoints,
        from_file,
    )
    sdf = df

    plot_filename = os.path.join(directory, name_x)

    histy = make_hist(
        sdf, palette, palette_orientation, custom_colors, custom_breakpoints
    )

    tri_plot = make_tri(
        sdf,
        mdf,
        cdf,
        plot_filename,
        palette,
        palette_orientation,
        custom_colors,
        axes_labels,
        xlim,
        axes_tick_number,
        deraster,
        width,
        name_x # hack
    )

    tri_prefix = f"{plot_filename}_TRI_IDENTITY"
    ggsave(
        tri_plot[0],
        width=width,
        height=width,
        dpi=dpi,
        format="svg",
        filename=f"{tri_prefix}.svg",
        verbose=False,
    )

    mod_prefix = f"{plot_filename}_TRI_MOD"
    ggsave(
        tri_plot[1],
        width=width,
        height=width / 5,
        dpi=dpi,
        format="svg",
        filename=f"{mod_prefix}.svg",
        verbose=False,
    )
    
    cenpa_prefix = f"{plot_filename}_TRI_CENPA"
    ggsave(
        tri_plot[2],
        width=width,
        height=width / 5,
        dpi=dpi,
        format="svg",
        filename=f"{cenpa_prefix}.svg",
        verbose=False,
    )

    # these values translate the rendered heatmap into the right position after rotation 
    scaling_values = (46.5 * width, -24.6 * width)
    rotate_rasterized_tri(f"{tri_prefix}.svg", scaling_values[0], scaling_values[1])
    
    merge_svgs_vertically([ f"{tri_prefix}.svg", f"{mod_prefix}.svg", f"{cenpa_prefix}.svg" ], f"{plot_filename}.merged.svg")
    cairosvg.svg2png(url=f"{plot_filename}.merged.svg", write_to=f"{plot_filename}.merged.png", output_height = 3000, output_width = 2000, dpi=300)
