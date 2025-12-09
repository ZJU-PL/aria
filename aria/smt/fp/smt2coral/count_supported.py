#!/usr/bin/env python
# vim: set sw=4 ts=4 softtabstop=4 expandtab:
"""
  Read invocation info file for smt-runner
  and report which benchmarks are supported
  by the CoralPrinter.
"""

# HACK: put smt2coral in search path
import os
import sys
_repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _repo_root)

import aria.smt.fp.smt2coral.converter as Converter
import aria.smt.fp.smt2coral.driver_util as DriverUtil
import aria.smt.fp.smt2coral.util as Util
import argparse
import logging
import yaml


_logger = logging.getLogger(__name__)

def loadYaml(openFile):
    if hasattr(yaml, 'CLoader'):
        # Use libyaml which is faster
        _loader = yaml.CLoader
    else:
        _logger.warning('Using slow Python YAML loader')
        _loader = yaml.Loader
    return yaml.load(openFile, Loader=_loader)

def benchmark_can_be_converted(full_path):
    assert os.path.exists(full_path)
    translation_was_sound = None

    # Parse using Z3
    with open(full_path, 'r') as f:
        _logger.debug('Opened "{}"'.format(f.name))
        constraint, err = Util.parse(f)
        if err is not None:
            # Parser failure
            _logger.error('Parser failure ({}): {}'.format(full_path, err))
            sys.exit(1)
    constraints = Util.split_bool_and(constraint)

    # Try to do conversion
    printer = Converter.CoralPrinter()
    try:
        _ = printer.print_constraints(constraints)
        translation_was_sound = printer.translation_was_sound()
    except Converter.CoralPrinterException as e:
        _logger.debug('{}: {}: {}'.format(full_path, type(e).__name__, e))
        return (False, translation_was_sound)
    return (True, translation_was_sound)

def main(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("invocation_info_file",
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
    )
    parser.add_argument('--benchmark-base',
        type=str,
        dest='benchmark_base',
        default="")
    DriverUtil.parserAddLoggerArg(parser)

    pargs = parser.parse_args(args)
    DriverUtil.handleLoggerArgs(pargs, parser)

    # Load invocatin info
    ii = loadYaml(pargs.invocation_info_file)
    assert isinstance(ii, dict)
    assert 'results' in ii
    results = ii['results']

    benchmarks_can_be_converted = set()
    benchmarks_with_sound_translation = set()
    benchmarks_cannot_be_converted = set()
    # Iterate over benchmarks
    for index, benchmark_info in enumerate(results):
        benchmark = benchmark_info['benchmark']
        full_path = os.path.join(pargs.benchmark_base, benchmark)
        if not os.path.exists(full_path):
            _logger.error('Could not find benchmark "{}"'.format(full_path))
            return 1
        can_convert, sound_translation = benchmark_can_be_converted(full_path)
        progress_str = '{} of {}'.format(index + 1, len(results))
        if can_convert:
            _logger.info('{}: Conversion successful ({})'.format(
                benchmark,
                progress_str))
            benchmarks_can_be_converted.add(benchmark)
            if sound_translation:
                benchmarks_with_sound_translation.add(benchmark)
        else:
            _logger.warning('{}: Conversion failed ({})'.format(
                benchmark,
                progress_str))
            benchmarks_cannot_be_converted.add(benchmark)

    # Report
    print("# of benchmarks can be converted: {}".format(
        len(benchmarks_can_be_converted)))
    print("# of benchmarks that could be converted soundly: {}".format(
        len(benchmarks_with_sound_translation)))
    print("# of benchmarks cannot be converted: {}".format(
        len(benchmarks_cannot_be_converted)))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
