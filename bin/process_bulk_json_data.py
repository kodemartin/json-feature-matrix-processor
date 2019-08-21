"""Script to process a set of files with JSON data in parallel
using the `multiprocessing` library.

The script checks for `json` files within the specified
directory.

Any files generated in the process are stored in the source
directory.

Usage::

    python -m bin.process_bulk_json_data [--dir <data-directory>]
        [--workers <number-of-workers>]

Generate help with::

    python -m bin.process_bulk_json_data -h|--help
"""
import argparse
import os
import glob

from multiprocessing import Pool, cpu_count

from processors import JsonData


def pipeline(filename):
    """Invoke a processing pipeline on JSON
    data included in the given filename.

    Currently the only side-effect of this function
    is the conversion of the original data
    to a csv.

    :param str filename: The name of the file
        to preprocess.
    """
    preprocessor = JsonData(filename)
    preprocessor.to_csv()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess multiple files with JSON data in parallel'
        )
    parser.add_argument(
        '--dir', default='data',
        help='The directory with the data-files to be preprocessed.'
        )
    parser.add_argument(
        '--workers', type=int, default=cpu_count(),
        help= 'The number of workers to enable for parallel processing.'
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Pool(args.workers) as pool:
        files = glob.iglob(os.path.join(args.dir, '*.json'))
        pool.map(pipeline, files)
