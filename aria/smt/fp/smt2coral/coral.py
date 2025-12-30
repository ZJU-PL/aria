#!/usr/bin/env python
# vim: set sw=4 ts=4 softtabstop=4 expandtab:
"""
  Read SMT-LIBv2 query file and attempt to solve
  them using the Coral constraint solver.
"""
import argparse
import os
import pprint
import logging
import subprocess
import sys
import tempfile
import traceback

import aria.smt.fp.smt2coral.converter as Converter
import aria.smt.fp.smt2coral.driver_util as DriverUtil
import aria.smt.fp.smt2coral.util as Util


_logger = logging.getLogger(__name__)

def main(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_file",
        nargs='?',
        type=argparse.FileType('r', encoding='utf-8'),
        default=sys.stdin,
    )
    parser.add_argument("--output",
        type=argparse.FileType('w'),
        default=sys.stdout,
    )
    parser.add_argument("--seed",
        help='Starting seed value',
        type=int,
        default=0,
    )
    parser.add_argument("--seed-iter",
        dest='seed_iter',
        help='Number of seeds to iterate through (default 1). -1 means unbounded.',
        type=int,
        default=1,
    )
    parser.add_argument("--dump-output",
        dest='dump_output',
        default=False,
        action='store_true',
    )
    DriverUtil.parserAddLoggerArg(parser)

    # Do partial argument passing so we can pass the rest to coral
    pargs, unhandled_args = parser.parse_known_args(args)
    DriverUtil.handleLoggerArgs(pargs, parser)

    if pargs.seed_iter < 1 and pargs.seed_iter != -1:
        _logger.error('Invalid value for --seed-iter')
        return 1
    try:
        # Parse using Z3
        constraint, err = Util.parse(pargs.query_file)
        if err is not None:
            # Parser failure
            _logger.error('Parser failure: %s', err)
            return 1
        constraints = Util.split_bool_and(constraint)

        # Do conversion from SMT-LIBv2 to Coral syntax
        printer = Converter.CoralPrinter()
        try:
            constraints = printer.print_constraints(constraints)
        except Converter.CoralPrinterException as e:
            _logger.error('%s: %s', type(e).__name__, e)
            _logger.debug(traceback.format_exc())
            pargs.output.write('unknown\n')
            return 1

        # Invoke coral
        coral_jar = os.path.join(os.path.dirname(__file__), 'coral.jar')
        if not os.path.exists(coral_jar):
            _logger.error('Cannot find "%s"', coral_jar)
            return 1

        starting_seed = pargs.seed
        last_seed = starting_seed + pargs.seed_iter -1

        seed = starting_seed
        if pargs.seed_iter != 1:
            _logger.debug('Using seed iteration mode')
        response = None
        while response is None:
            exit_code, response = run_coral(
                coral_jar,
                constraints,
                seed,
                unhandled_args,
                pargs
            )
            seed += 1
            if (seed > last_seed and pargs.seed_iter != -1) or exit_code != 0:
                break
        if response is True:
            pargs.output.write('sat\n')
        elif response is False:
            pargs.output.write('unsat\n')
        else:
            pargs.output.write('unknown\n')
        return exit_code
    except KeyboardInterrupt:
        pargs.output.write('unknown\n')
        return 1

def run_coral(coral_jar, constraints, seed, unhandled_args, pargs):
    assert isinstance(seed, int)
    cmd_line = [
        'java',
        '-jar',
        coral_jar
    ]
    # Add unhandled arguments
    cmd_line.extend(unhandled_args)

    # Now add the constraint and seed argument
    cmd_line.extend([
        '--inputCONS',
        constraints,
        '--seed={}'.format(seed),
    ])

    _logger.debug('Invoking: %s', pprint.pformat(cmd_line))

    # Write stdout to tempfile so we can parse its output
    with tempfile.TemporaryFile() as stdout:
        proc = subprocess.Popen(args=cmd_line, stdout=stdout)
        try:
            exit_code = proc.wait()
        except KeyboardInterrupt as e:
            proc.kill()
            raise e
        response = parse_coral_output(stdout, pargs.dump_output)
        return (exit_code, response)

def parse_coral_output(stdout, dump_output):
    # Convert stdout to string
    # Move to beginning of stream
    stdout.seek(0)
    lines = []
    for l in stdout.readlines():
        line_as_str = l.decode()
        lines.append(line_as_str)
    if dump_output:
        _logger.info('Coral output:\n%s', pprint.pformat(lines))

    # Walk past parameter dump and get satisfiability response
    num_eq_encounter = 0
    line_to_parse = None
    for l in lines:
        if num_eq_encounter < 2:
            if l.startswith('==='):
                num_eq_encounter += 1
            continue
        assert num_eq_encounter == 2
        line_to_parse = l
        break

    if line_to_parse is None:
        _logger.error('Failed to find required output line')
        return None

    _logger.debug('Line to parse is "%s"', line_to_parse)
    if line_to_parse.startswith('SOLVED'):
        # Satisfiable
        return True
    if line_to_parse.startswith('NOT SOLVED'):
        # Unsat?
        # Not sure, for now just pretend like it's unknown
        return None
    # Unknown
    _logger.error('Unrecognised response from coral: "%s"', line_to_parse)
    return None


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
