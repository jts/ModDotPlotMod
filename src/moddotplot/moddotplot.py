#!/usr/bin/env python3
import sys
from moddotplot.parse_fasta import (
    readKmersFromFile,
    getInputHeaders,
    isValidFasta,
    extractFiles,
    extractRegion,
)

from moddotplot.estimate_identity import (
    convertToModimizers,
    selfContainmentMatrix,
    pairwiseContainmentMatrix,
    convertMatrixToBed,
    convertMatrixToCool,
    createSelfMatrix,
    createPairwiseMatrix,
    partitionOverlaps,
)
from moddotplot.const import ASCII_ART, VERSION

import argparse
import math
from moddotplot.static_plots import read_df_from_file, create_plots
import json
import pandas as pd
import numpy as np
import pyBigWig
import pickle
import os

def read_bedmethyl(fn):
    # Column names exactly as defined in the modkit documentation
    columns = [
        "chromosome",
        "start",
        "end",
        "mod_code",
        "score",
        "strand",
        "compat_start",
        "compat_end",
        "color",
        "n_valid_cov",
        "percent_modified",
        "n_mod",
        "n_canonical",
        "n_other_mod",
        "n_delete",
        "n_fail",
        "n_diff",
        "n_nocall",
    ]

    dtypes = {
        "chromosome": "string",
        "start": "int64",
        "end": "int64",
        "mod_code_motif": "string",
        "score": "int64",
        "strand": "string",
        "compat_start": "int64",
        "compat_end": "int64",
        "color": "string",
        "n_valid_cov": "int64",
        "percent_modified": "float64",
        "n_mod": "int64",
        "n_canonical": "int64",
        "n_other_mod": "int64",
        "n_delete": "int64",
        "n_fail": "int64",
        "n_diff": "int64",
        "n_nocall": "int64",
    }

    # Read headerless bedmethyl file
    df = pd.read_csv(
        fn,
        sep="\t",
        header=None,
        names=columns,
        dtype=dtypes
    )

    # Keep only chrom, start, end, and all N* columns
    keep_cols = (
        ["chromosome", "start", "end"] +
        [c for c in df.columns if c.startswith("n_")]
    )

    df = df[keep_cols]
    return df

def get_parser():
    """
    Argument parsing for stand-alone runs.

    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ModDotPlot: Visualization of Tandem Repeats",
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Choose mode: interactive or static"
    )
    static_parser = subparsers.add_parser("static", help="Static mode commands")

    # -----------STATIC MODE SUBCOMMANDS-----------
    static_input_group = static_parser.add_mutually_exclusive_group(required=True)
    static_input_group.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="Config file to use. Takes precedence over any other competing command line arguments.",
    )

    static_input_group.add_argument(
        "-l",
        "--load",
        default=argparse.SUPPRESS,
        help="Path to input paired-end bed file(s). Exclusively used in static mode.",
        nargs="+",
    )

    static_input_group.add_argument(
        "-f",
        "--fasta",
        default=argparse.SUPPRESS,
        help="Path to input fasta file(s).",
        nargs="+",
    )

    # Add a mutually exclusive group for compare and compare only.
    static_compare_group = static_parser.add_mutually_exclusive_group(required=False)
    static_window_size_group = static_parser.add_mutually_exclusive_group(
        required=False
    )

    static_parser.add_argument("-b", "--bed", default=None, help="Bed file annotation.")

    static_parser.add_argument(
        "-k", "--kmer", default=21, type=int, help="k-mer length."
    )

    static_parser.add_argument(
        "-m",
        "--modimizer",
        help="Modimizer sketch size. A lower value will reduce the number of modimizers, but will increase performance. Must be less than window length `-w`. ",
        default=1000,
        type=int,
    )

    static_window_size_group.add_argument(
        "-r",
        "--resolution",
        default=1000,
        type=int,
        help="Dotplot resolution, or the number of intervals to compare against.",
    )

    static_window_size_group.add_argument(
        "-w",
        "--window",
        default=None,
        type=int,
        help="Window size, or the length in genomic coordinates of each interval. Default is set to (genome length)/(resolution)",
    )

    static_parser.add_argument(
        "--region",
        default=None,
        help="Genomic region to analyze. Syntax is seq_id:start-end (e.g. chr1:100000-200000).",
        nargs="+",
    )

    static_parser.add_argument(
        "-id",
        "--identity",
        default=86.0,
        type=float,
        help="Identity cutoff threshold.",
    )

    static_parser.add_argument(
        "-d",
        "--delta",
        default=0.5,
        type=float,
        help="Fraction of neighboring partition to include in identity estimation. Must be between 0 and 1, use > 0.5 is not recommended.",
    )

    static_parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory name for saving bed files and plots. Defaults to working directory.",
    )

    static_compare_group.add_argument(
        "--compare",
        action="store_true",
        help="Create a dotplot with two different sequences (in addition to self-identity plots).",
    )

    static_compare_group.add_argument(
        "--compare-only",
        action="store_true",
        help="Create a dotplot with two different sequences (skips self-identity plots).",
    )

    static_parser.add_argument(
        "--compare-order",
        choices=["sequential", "size"],
        default="sequential",
        help="Order in which sequences appear in the comparative plot. Default is 'sequential': First file on x-axis, second file on y-axis. Another option is 'size': The larger sequence on the x-axis and the smaller on y-axis.",
    )

    static_parser.add_argument(
        "--cooler", action="store_true", help="Output matrix to cooler file."
    )

    static_parser.add_argument(
        "--no-bedpe", action="store_true", help="Skip output of paired-end bed file."
    )

    static_parser.add_argument(
        "--no-plot", action="store_true", help="Skip output of plots."
    )

    static_parser.add_argument(
        "--no-hist", action="store_true", help="Skip output of histogram color legend."
    )

    static_parser.add_argument(
        "--width", default=9, type=float, help="Plot width (also height for _FULL)."
    )

    static_parser.add_argument("--dpi", default=300, type=int, help="Plot dpi.")

    # TODO: Create list of accepted colors.
    static_parser.add_argument(
        "--palette",
        default="Spectral_11",
        help="Select color palette. See RColorBrewer for list of accepted palettes. Will default to Spectral_11 if not used.",
        type=str,
    )

    static_parser.add_argument(
        "--palette-orientation",
        default="+",
        choices=["+", "-"],
        help="Color palette orientation. + for forward, - for reverse.",
        type=str,
    )

    static_parser.add_argument(
        "--forward",
        action="store_true",
        help="Enforce forward only k-mers instead of canonical k-mers. Warning: only use if you want strand-specific output!",
    )

    static_parser.add_argument(
        "--plot-direction",
        action="store_true",
        help="Create a plot containing the direction of each k-mer array (relative to the first array). Arrays with inversions will be highlighted in blue (forward) and pink (reverse).",
    )

    static_parser.add_argument(
        "--colors",
        default=None,
        nargs="+",
        help="Use a custom color palette, entered in either hexcode or rgb format.",
    )

    static_parser.add_argument(
        "--breakpoints",
        default=None,
        nargs="+",
        help="Introduce custom color thresholds. Must be between identity threshold and 100.",
    )

    static_parser.add_argument(
        "-a",
        "--axes-limits",
        default=None,
        type=float,
        help="Change x and y axis limits for self identity plots. Default is length of the sequence. Can't be shorter than length of sequence.",
    )

    static_parser.add_argument(
        "-t",
        "--axes-ticks",
        default=None,
        nargs="+",
        type=int,
        help="Tick labels to include in x and y axis for custom plots.",
    )
    # CURRENTLY NOT WORKING
    static_parser.add_argument(
        "--axes-number",
        default=7,
        help="Number of axis ticks labels to include in x and y axis for custom plots, including 0 and seq_length. A minimum of 2 is required, maximum 25.",
    )

    static_parser.add_argument(
        "--bin-freq",
        action="store_true",
        help="By default, histograms are evenly spaced based on the number of colors and the identity threshold. Select this argument to bin based on the frequency of observed identity values.",
    )

    static_parser.add_argument(
        "--ambiguous",
        action="store_true",
        help="Preserve diagonal when handling strings of ambiguous homopolymers (eg. long runs of N's).",
    )

    static_parser.add_argument(
        "--grid",
        action="store_true",
        help="Plot comparative plots in an NxN grid like format.",
    )

    static_parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Plot comparative plots in an NxN grid like format, skipping individual plots.",
    )

    static_parser.add_argument(
        "--vector",
        choices=["svg", "pdf", "ps"],
        default="svg",
        help="Output format for vector format.",
    )

    static_parser.add_argument(
        "--deraster",
        action="store_true",
        help="De-rasterize dotplot in vector format. Note this can lead to large image sizes and make it unusable in image editing software.",
    )

    return parser


def main():
    print(ASCII_ART)
    print(f"v{VERSION} \n")
    args = get_parser().parse_args()

    if args.command != "static":
        sys.stderr.write("modified version only supports static mode")
        sys.exit(1)
        
    print(f"Running ModDotPlot in static mode\n")
    # -----------CONFIG PARSING-----------
    # TODO: Change to yml file, add readme to config folder
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
            # TODO: Remove args that are interactive only
            args.fasta = config.get("fasta")
            args.load = config.get("load")
            args.bed = config.get("bed")

            # Distance matrix commands
            args.kmer = config.get("kmer", args.kmer)
            args.modimizer = config.get("modimizer", args.modimizer)
            args.resolution = config.get("resolution", args.resolution)
            args.window = config.get("window", args.window)
            args.identity = config.get("identity", args.identity)
            args.delta = config.get("delta", args.delta)
            args.output_dir = config.get("output_dir", args.output_dir)
            args.compare = config.get("compare", args.compare)
            args.compare_only = config.get("compare_only", args.compare_only)

            args.no_bedpe = config.get("no_bedpe", args.no_bedpe)
            args.no_plot = config.get("no_plot", args.no_plot)
            args.no_hist = config.get("no_hist", args.no_hist)
            args.width = config.get("width", args.width)
            args.axes_limits = config.get("axes_limits", args.axes_limits)
            args.dpi = config.get("dpi", args.dpi)
            args.palette = config.get("palette", args.palette)
            args.palette_orientation = config.get(
                "palette_orientation", args.palette_orientation
            )
            args.colors = config.get("color", args.colors)
            args.axes_ticks = config.get("axes_ticks", args.axes_ticks)
            args.breakpoints = config.get("breakpoints", args.breakpoints)
            args.bin_freq = config.get("bin_freq", args.bin_freq)
            args.axes_limits = config.get("axes_limits", args.axes_limits)
            args.axes_ticks = config.get("axes_ticks", args.axes_ticks)
            args.vector = config.get("vector", args.vector)
            args.deraster = config.get("deraster", args.deraster)

        # -----------INPUT COMMAND VALIDATION-----------
        # TODO: More tests!
        if args.breakpoints:
            # Check that start value for breakpoints = identity threshold value
            if float(args.breakpoints[0]) != float(args.identity):
                print(
                    f"Identity threshold is {args.identity}, but starting breakpoint is {args.breakpoints[0]}! \n"
                )
                print(
                    f"Please modify identity threshold using --identity {args.breakpoints[0]}.\n"
                )
                # TODO: Chronicle exit codes
                # Exit code 2: breakpoint value != identity threshold
                sys.exit(2)

    # -----------INPUT SEQUENCE VALIDATION-----------
    seq_list = []
    fasta_list = args.fasta.copy()
    for i in args.fasta:
        try:
            isValidFasta(i)
            headers = getInputHeaders(i)

            if len(headers) > 1:
                print(f"File {i} contains multiple fasta entries.\n")

            seq_list.extend(headers)  # Add all headers to seq_list

        except Exception as e:
            print(
                f"\nUnable to open {i}. Please check it is correctly formatted or compressed...\n"
            )
            fasta_list.remove(i)

    # -----------LOAD SEQUENCES INTO MEMORY-----------
    kmer_list = []
    for i in fasta_list:
        kmer_list.append(readKmersFromFile(i, args.kmer, False, False))
    k_list = [item for sublist in kmer_list for item in sublist]
    # Throw error if compare only selected with one sequence.
    if len(k_list) < 2 and args.compare_only:
        print(
            f"Error: Can't create a comparative plot with only one sequence. Please re-run without --compare-only."
        )
        sys.exit(2)

    # -----------SET SPARSITY VALUE-----------
    if args.grid or args.grid_only:
        grid_val_singles = []
        grid_val_single_names = []
    new_sequences = list(zip(seq_list, k_list))
    if args.compare_order == "size":
        sequences = sorted(new_sequences, key=lambda seq: len(seq[1]), reverse=True)
    else:
        sequences = new_sequences
    if len(sequences) > 6 and (args.grid or args.grid_only):
        print("Too many sequences to create a grid. Skipping. \n")

    # Create output directory, if doesn't exist:
    if (args.output_dir) and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # -----------COMPUTE SELF-IDENTITY PLOTS-----------
    if not args.compare_only:
        for i in range(len(sequences)):
            seq_length = len(sequences[i][1])
            seq_name = sequences[i][0]
            seq_range = extractRegion(seq_name)
            if seq_range:
                seq_name = seq_range[0]
            # If region, then I only want to use the subsequence.
            try:
                if args.region:
                    subseq_start_pos = None
                    subseq_end_pos = None
                    for region in args.region:
                        chrom, lower_bound, upper_bound = extractRegion(region)
                        if chrom == seq_name:
                            subseq_start_pos = lower_bound
                            subseq_end_pos = upper_bound
                            seq_start_pos = lower_bound
                            # Validate bounds
                            if subseq_start_pos < 1 or subseq_end_pos > seq_length:
                                print(
                                    f"Error: region {region} is out of bounds for {seq_name}. Will use entire sequence.\n"
                                )
                                subseq_start_pos = 1
                                subseq_end_pos = seq_length
                                seq_name = sequences[i][0]
                                break
                            print(
                                f"Using region {seq_name}:{subseq_start_pos}-{subseq_end_pos}\n"
                            )
                            # Change sequence length, and use a subsequence instead.
                            seq_length = (
                                subseq_end_pos - subseq_start_pos + 1 - args.kmer
                            )
                            seq_range = seq_name, subseq_start_pos, subseq_end_pos
                            seq_name = (
                                f"{seq_name}:{subseq_start_pos}-{subseq_end_pos}"
                            )
                    if not subseq_end_pos or not subseq_start_pos:
                        print(
                            f"Error: region {args.region} not found in {seq_name}. Will use entire sequence.\n"
                        )
                        seq_range = None
            except Exception as e:
                print(
                    f"Error obtaining region for {seq_name}. Will use entire sequence: {e}\n"
                )
            if not seq_range:
                seq_start_pos = 1
            else:
                seq_start_pos = int(seq_range[1])
            win = args.window
            res = args.resolution
            if args.window:
                # Change the resolution of each plot
                res = math.ceil(seq_length / args.window)
            else:
                win = math.ceil(seq_length / args.resolution)

            if win < args.modimizer:
                args.modimizer = win
            if win < 10:
                print(f"Error: sequence too small for analysis.\n")
                print(
                    f"ModDotPlot requires a minimum window size of 10. Sequences less than 10Kbp will not work with ModDotPlot under normal resolution. We recommend rerunning ModDotPlot with --r {math.ceil(seq_length / 10)}.\n"
                )
                sys.exit(0)

            seq_sparsity = round(win / args.modimizer)
            if seq_sparsity <= args.modimizer:
                seq_sparsity = 2 ** int(math.log2(seq_sparsity))
            else:
                seq_sparsity = 2 ** (int(math.log2(seq_sparsity - 1)) + 1)
            expectation = round(win / seq_sparsity)

            print(f"Computing self identity matrix for {seq_name}... \n")
            # TODO: Logging here
            # print(f"\tSparsity value s: {seq_sparsity}\n")
            print(f"\tSequence length n: {seq_length + args.kmer - 1}\n")
            print(f"\tWindow size w: {win}\n")
            print(f"\tModimizer sketch size: {expectation}\n")
            print(f"\tPlot Resolution r: {res}\n")
            if args.region and seq_range:
                subseq = sequences[i][1][
                    subseq_start_pos : (subseq_end_pos - args.kmer + 1)
                ]
                self_mat = createSelfMatrix(
                    seq_length,
                    subseq,
                    win,
                    seq_sparsity,
                    args.delta,
                    args.kmer,
                    args.identity,
                    args.ambiguous,
                    expectation,
                )
            else:
                self_mat = createSelfMatrix(
                    seq_length,
                    sequences[i][1],
                    win,
                    seq_sparsity,
                    args.delta,
                    args.kmer,
                    args.identity,
                    args.ambiguous,
                    expectation,
                )
            bed = convertMatrixToBed(
                self_mat,
                win,
                args.identity,
                seq_name,
                seq_name,
                True,
                seq_start_pos,
                seq_start_pos,
            )
            if args.grid or args.grid_only:
                grid_val_singles.append(bed)
                grid_val_single_names.append(seq_name)

            if args.cooler:
                try:
                    cooler_path = "."
                    if not args.output_dir:
                        cooler_path = os.path.join(cooler_path, seq_name)
                    else:
                        cooler_path = os.path.join(args.output_dir, seq_name)
                    os.makedirs(cooler_path, exist_ok=True)
                    cooler_output = os.path.join(cooler_path, seq_name + ".cooler")
                    convertMatrixToCool(
                        matrix=self_mat,
                        window_size=win,
                        id_threshold=args.identity,
                        x_name=seq_name,
                        y_name=seq_name,
                        self_identity=True,
                        x_offset=seq_start_pos,
                        y_offset=seq_start_pos,
                        chromsizes=seq_length,
                        output_cool=cooler_output,
                    )
                    print(
                        f"Saved self-identity matrix as a cooler file to {cooler_output}\n"
                    )
                except Exception as e:
                    print(f"Error creating cooler file: {e}")

            if not args.no_bedpe:
                # Log saving bed file
                bedpe_path = "."
                if not args.output_dir:
                    bedpe_path = os.path.join(bedpe_path, seq_name)
                    os.makedirs(bedpe_path, exist_ok=True)
                    bedfile_output = os.path.join(seq_name, seq_name + ".bedpe")
                else:
                    bedpe_path = os.path.join(args.output_dir, seq_name)
                    os.makedirs(bedpe_path, exist_ok=True)
                    bedfile_output = os.path.join(bedpe_path, seq_name + ".bedpe")

                with open(bedfile_output, "w") as bedfile:
                    for row in bed:
                        bedfile.write("\t".join(map(str, row)) + "\n")
                print(
                    f"Saved self-identity matrix as a paired-end bed file to {bedfile_output}\n"
                )

            # load additional tracks
            
            # methylation
            fn = "~/incoming/mole_rat/t2t_qc/publication_figures/data/pup_5mC_merged.cleaned.bed"
            mdf = read_bedmethyl(fn)
            mdf["percent"] = 100.0 * mdf["n_mod"] / mdf["n_valid_cov"]
            
            # cenpa
            bw_path = "/Users/jsimpson/incoming/mole_rat/t2t_qc/publication_figures/data/WL3840_CENPA_liver_MA1-20832_hgla_4814_none_TCAG1.bw"
            bw = pyBigWig.open(bw_path)
            values = bw.values(seq_range[0], seq_range[1], seq_range[2])
            bw.close()

            cdf = pd.DataFrame({
                "chromosome": seq_range[0],
                "start": range(seq_range[1], seq_range[2]),
                "end": range(seq_range[1] + 1, seq_range[2] + 1),
                "value": values
            })

            if (not args.no_plot) and (not args.grid_only):
                create_plots(
                    sdf=[bed],
                    mdf=mdf,
                    cdf=cdf,
                    directory=bedpe_path,
                    name_x=seq_name,
                    name_y=seq_name,
                    palette=args.palette,
                    palette_orientation=args.palette_orientation,
                    no_hist=args.no_hist,
                    width=args.width,
                    dpi=args.dpi,
                    is_freq=args.bin_freq,
                    xlim=args.axes_limits,
                    custom_colors=args.colors,
                    custom_breakpoints=args.breakpoints,
                    from_file=None,
                    is_pairwise=False,
                    axes_labels=args.axes_ticks,
                    axes_tick_number=args.axes_number,
                    vector_format=args.vector,
                    deraster=args.deraster,
                    annotation=args.bed,
                )



if __name__ == "__main__":
    main()
