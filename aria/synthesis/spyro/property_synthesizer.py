"""Property synthesizer for Spyro."""

import subprocess
import os
import time
import configparser

from aria.synthesis.spyro.util import union_dict
from aria.synthesis.spyro.input_generator import InputGenerator
from aria.synthesis.spyro.output_parser import OutputParser

CONFIG_FILE = "config.ini"

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

SKETCH_BINARY_PATH = config["DEFAULT"]["SKETCH_BINARY_PATH"]
LOG_FILE_DIR = config["DEFAULT"]["LOG_FILE_DIR"]
TEMP_FILE_DIR = config["DEFAULT"]["TEMP_FILE_DIR"]
TEMP_NAME_DEFAULT = config["DEFAULT"]["TEMP_NAME_DEFAULT"]

class PropertySynthesizer:
    """Property synthesizer for Spyro."""

    def __init__(
        self, infiles,
        outfile, verbose, write_log,
        timeout, inline_bnd, slv_seed,
        num_atom_max, disable_min, keep_neg_may):

        # Input/Output file stream
        self.__outfile = outfile

        # Temporary filename for iteration
        self.__tempfile_name = self.__get_tempfile_name(infiles[0], outfile)

        # Template for Sketch synthesis
        self.__code = ''
        for f in infiles:
            self.__code += f.read() + '\n'

        # Sketch Input File Generator
        self.__minimize_terms = not disable_min
        self.__input_generator = InputGenerator(self.__code)
        self.__input_generator.set_num_atom(num_atom_max)

        # Synthesized property
        self.__phi_truth = "out = true;"

        # Options
        self.__verbose = verbose
        self.__write_log = write_log
        self.__move_neg_may = not keep_neg_may
        self.__update_psi = False
        self.__use_delta = False
        self.__discard_all = True
        self.__timeout = timeout
        self.__inline_bnd = inline_bnd
        self.__slv_seed = slv_seed

        # Iterators for descriptive message
        self.__inner_iterator = 0
        self.__outer_iterator = 0

        self.__num_soundness = 0
        self.__num_precision = 0
        self.__num_maxsat = 0
        self.__num_synthesis = 0

        self.__time_soundness = 0
        self.__time_precision = 0
        self.__time_maxsat = 0
        self.__time_synthesis = 0

        self.__max_time_soundness = 0
        self.__max_time_precision = 0
        self.__max_time_maxsat = 0
        self.__max_time_synthesis = 0

        self.__time_last_query = 0

        self.__statistics = []

        # Log file (initialized in run() if needed)
        self.__logfile = None

    def __extract_filename_from_path(self, path):
        basename = os.path.basename(path)
        filename, extension = os.path.splitext(basename)

        return filename

    def __get_tempfile_name(self, infile, outfile):
        infile_path = infile.name

        if infile_path != '<stdin>':
            return self.__extract_filename_from_path(infile_path)

        return TEMP_NAME_DEFAULT

    def __write_tempfile(self, path, code):
        if not os.path.isdir(TEMP_FILE_DIR):
            os.mkdir(TEMP_FILE_DIR)

        with open(path, 'w') as f:
            f.write(code)

    def __open_logfile(self, filename):
        if not os.path.isdir(LOG_FILE_DIR):
            os.mkdir(LOG_FILE_DIR)

        return open(LOG_FILE_DIR + filename, 'w')


    def __get_new_tempfile_path(self):
        path = TEMP_FILE_DIR
        path += self.__tempfile_name
        if self.__verbose:
            path += f'_{self.__outer_iterator}_{self.__inner_iterator}'
        path += ".sk"

        self.__inner_iterator += 1

        return path

    def __try_synthesis(self, code):
        start_time = time.time()

        # Write temp file
        path = self.__get_new_tempfile_path()
        self.__write_tempfile(path, code)

        try:
            # Run Sketch with temp file
            output = subprocess.check_output(
                [SKETCH_BINARY_PATH, path,
                    '--bnd-inline-amnt', str(self.__inline_bnd),
                    '--slv-seed', str(self.__slv_seed),
                    '--slv-timeout', f'{self.__timeout / 60.0:2f}'],
                stderr=subprocess.PIPE)

            end_time = time.time()

            return output, end_time - start_time

        except subprocess.CalledProcessError as e:
            end_time = time.time()
            return None, end_time - start_time

        except subprocess.TimeoutExpired as e:
            if self.__verbose:
                print("Timeout")

            end_time = time.time()

            return None, end_time - start_time

    def __synthesize(self, pos, neg_must, neg_may, lam_functions):
        if self.__verbose:
            print(f'Iteration {self.__outer_iterator} - {self.__inner_iterator}: Try synthesis')

        # Run Sketch with temp file
        code = self.__input_generator.generate_synthesis_input(pos, neg_must, neg_may, lam_functions)
        output, elapsed_time = self.__try_synthesis(code)

        # Update statistics
        self.__num_synthesis += 1
        self.__time_synthesis += elapsed_time
        if elapsed_time > self.__max_time_synthesis:
            self.__max_time_synthesis = elapsed_time

        # Write trace log
        if self.__write_log:
            log = [f'{self.__outer_iterator}', f'{self.__inner_iterator}']
            log += ['Y', f'{elapsed_time}']
            log += [f'{len(pos)}', f'{len(neg_must)}', f'{len(neg_may)}']

            self.__logfile.write(','.join(log) + "\n")

        # Return the result
        if output != None:
            output_parser = OutputParser(output)
            phi = output_parser.parse_property()
            lam = output_parser.get_lam_functions()
            return (phi, lam)
        else:
            return (None, None)

    def __max_synthesize(self, pos, neg_must, neg_may, lam_functions, phi_init):
        if self.__verbose:
            print(f'Iteration {self.__outer_iterator} - {self.__inner_iterator}: Run MaxSat')

        # Run Sketch with temp file
        code = self.__input_generator.generate_maxsat_input(pos, neg_must, neg_may, lam_functions)
        output, elapsed_time = self.__try_synthesis(code)

        # Update statistics
        self.__num_maxsat += 1
        self.__time_maxsat += elapsed_time
        if elapsed_time > self.__max_time_maxsat:
            self.__max_time_maxsat = elapsed_time

        # Write trace log
        if self.__write_log:
            log = [f'{self.__outer_iterator}', f'{self.__inner_iterator}']
            log += ['M', f'{elapsed_time}']
            log += [f'{len(pos)}', f'{len(neg_must)}', f'{len(neg_may)}']

            self.__logfile.write(','.join(log) + "\n")

        # Return the result
        if output != None:
            output_parser = OutputParser(output)
            neg_may, delta = output_parser.parse_maxsat(neg_may)
            phi = output_parser.parse_property()
            lam = output_parser.get_lam_functions()
            return (neg_may, delta, phi, lam)
        else:
            if phi_init == None:
                raise Exception("MaxSynth Failed")

            neg_may, delta = [], neg_may
            phi = phi_init
            lam = lam_functions
            return (neg_may, delta, phi, lam)

    def __check_soundness(self, phi, lam_functions):
        if self.__verbose:
            print(f'Iteration {self.__outer_iterator} - {self.__inner_iterator}: Check soundness')

        # Run Sketch with temp file
        code = self.__input_generator.generate_soundness_input(phi, lam_functions)
        output, elapsed_time = self.__try_synthesis(code)

        # Update statistics
        self.__num_soundness += 1
        self.__time_soundness += elapsed_time
        if elapsed_time > self.__max_time_soundness:
            self.__max_time_soundness = elapsed_time

        # Write trace log
        if self.__write_log:
            log = [f'{self.__outer_iterator}', f'{self.__inner_iterator}']
            log += ['S', f'{elapsed_time}']
            log += ['-', '-', '-']

            self.__logfile.write(','.join(log) + "\n")

        # Return the result
        if output != None:
            output_parser = OutputParser(output)
            e_pos = output_parser.parse_positive_example()
            lam = output_parser.get_lam_functions()
            return (e_pos, lam, False)
        else:
            return (None, None, elapsed_time >= self.__timeout)

    def __check_precision(self, phi, phi_list, pos, neg_must, neg_may, lam_functions):
        if self.__verbose:
            print(f'Iteration {self.__outer_iterator} - {self.__inner_iterator}: Check precision')

        # Run Sketch with temp file
        code = self.__input_generator \
            .generate_precision_input(phi, phi_list, pos, neg_must, neg_may, lam_functions)
        output, elapsed_time = self.__try_synthesis(code)

        # Update statistics
        self.__num_precision += 1
        self.__time_precision += elapsed_time
        if elapsed_time > self.__max_time_precision:
            self.__max_time_precision = elapsed_time

        # Write trace log file
        if self.__write_log:
            log = [f'{self.__outer_iterator}', f'{self.__inner_iterator}']
            log += ['P', f'{elapsed_time}']
            log += [f'{len(pos)}', f'{len(neg_must)}', f'{len(neg_may)}']

            self.__logfile.write(','.join(log) + "\n")

        # Return the result
        if output != None:
            output_parser = OutputParser(output)
            e_neg = output_parser.parse_negative_example_precision()
            phi = output_parser.parse_property()
            lam = output_parser.get_lam_functions()
            return (e_neg, phi, lam)
        else:
            self.__time_last_query = elapsed_time
            return (None, None, None)

    def __check_improves_predicate(self, phi_list, phi, lam_functions):
        if self.__verbose:
            print(f'Iteration {self.__outer_iterator} : Check termination')

        # Run Sketch with temp file
        code = self.__input_generator.generate_improves_predicate_input(phi, phi_list, lam_functions)
        output, _ = self.__try_synthesis(code)

        # Return the result
        if output != None:
            output_parser = OutputParser(output)
            e_neg = output_parser.parse_improves_predicate()
            return e_neg
        else:
            return None

    def __model_check(self, phi, neg_example, lam_functions):
        if self.__verbose:
            print(f'Iteration {self.__outer_iterator} : Model check')

        # Run Sketch with temp file
        code = self.__input_generator \
            .generate_model_check_input(phi, neg_example, lam_functions)
        output, _ = self.__try_synthesis(code)

        # Return the result
        return output != None

    def __filter_neg_delta(self, phi, neg_delta, lam_functions):
        return [e for e in neg_delta if self.__model_check(phi, e, lam_functions)]

    def __synthesizeProperty(
            self, phi_list, phi_init,
            pos, neg_must, neg_may, lam_functions,
            most_precise, update_psi):
        # Assume that current phi is sound
        phi_e = phi_init
        phi_last_sound = None
        neg_delta = []
        phi_sound = []

        while True:
            e_pos, lam, timeout = self.__check_soundness(phi_e, lam_functions)
            if e_pos != None:
                pos.append(e_pos)
                lam_functions = union_dict(lam_functions, lam)

                # First try synthesis
                phi, lam = self.__synthesize(pos, neg_must, neg_may, lam_functions)

                # If neg_may is a singleton set, it doesn't need to call MaxSynth
                # Revert to the last remembered sound property
                if phi == None and ((len(neg_may) == 1 and phi_last_sound != None) or self.__discard_all):
                    phi = phi_last_sound
                    neg_delta += neg_may
                    neg_may = []
                    lam = {}

                # MaxSynth
                elif phi == None:
                    neg_may, delta, phi, lam = self.__max_synthesize(
                        pos, neg_must, neg_may, lam_functions, phi_last_sound)
                    neg_delta += delta

                    # MaxSynth can't minimize the term size, so call the Synth again
                    if self.__input_generator.minimize_terms_enabled:
                        phi, lam = self.__synthesize(pos, neg_must, neg_may, lam_functions)

                phi_e = phi
                lam_functions = union_dict(lam_functions, lam)

            # Return the last sound property found
            elif timeout and phi_last_sound != None:
                neg_delta = self.__filter_neg_delta(phi_last_sound, neg_delta, lam_functions)
                return (phi_last_sound, pos, neg_must + neg_may, neg_delta, lam_functions)

            elif timeout:
                return (self.__phi_truth, pos, [], [], lam_functions)

            # Early termination after finding a sound property with negative example
            elif not most_precise and len(neg_may) > 0:
                neg_delta = self.__filter_neg_delta(phi_e, neg_delta, lam_functions)
                return (phi_e, pos, neg_must + neg_may, neg_delta, lam_functions)

            # Check precision after pass soundness check
            else:
                phi_last_sound = phi_e    # Remember the last sound property

                if update_psi and len(neg_may) > 0:
                    phi_list = phi_list + [phi_e]

                # If phi_e is phi_truth, which is initial candidate of the first call,
                # then phi_e doesn't rejects examples in neg_may.
                if self.__move_neg_may:
                    neg_must += neg_may
                    neg_may = []

                e_neg, phi, lam = self.__check_precision(
                    phi_e, phi_list, pos, neg_must, neg_may, lam_functions)
                if e_neg != None:   # Not precise
                    phi_e = phi
                    neg_may.append(e_neg)
                    lam_functions = lam
                else:               # Sound and Precise
                    neg_delta = self.__filter_neg_delta(phi_e, neg_delta, lam_functions)
                    return (phi_e, pos, neg_must + neg_may, neg_delta, lam_functions)

    def __remove_redundant(self, phi_list):
        for i in range(len(phi_list)):
            phi = phi_list[i]
            others = phi_list[:i] + phi_list[i:]

            e_neg = self.__check_improves_predicate(others, phi)
            if e_neg == None:
                return (True, others)

        return (False, phi_list)


    def __synthesizeAllProperties(self):
        phi_list = []
        fun_list = []
        pos = []
        neg_may = []
        lam_functions = {}

        while True:
            # Find a property improves conjunction as much as possible
            self.__input_generator.disable_minimize_terms()

            if not self.__use_delta:
                neg_may = []

            if len(neg_may) > 0:
                neg_may, _, phi_init, lam = self.__max_synthesize(
                    pos, [], neg_may, lam_functions, self.__phi_truth)
                lam_functions = union_dict(lam_functions, lam)
            else:
                phi_init = self.__phi_truth

            most_precise = self.__minimize_terms
            phi, pos, neg_must, neg_may, lam = \
                self.__synthesizeProperty(
                    phi_list, phi_init, pos, [], neg_may, lam_functions,
                    most_precise, self.__update_psi)
            lam_functions = lam

            # Check if most precise candidates improves property.
            # If neg_must is nonempty, those examples are witness of improvement.
            if len(neg_must) == 0:
                e_neg = self.__check_improves_predicate(phi_list, phi, lam_functions)
                if e_neg != None:
                    neg_must = [e_neg]
                else:
                    stat = self.__statisticsCurrentProperty(pos, neg_must, neg_may, [], [])
                    self.__statistics.append(stat)
                    return phi_list, fun_list

            if self.__minimize_terms:
                self.__input_generator.enable_minimize_terms()

                # Synthesize a new candidate, which is minimized
                # We can always synthesize a property here
                phi, lam = self.__synthesize(pos, neg_must, [], lam_functions)
                lam_functions = union_dict(lam_functions, lam)

            # Strengthen the found property to be most precise L-property
            phi, pos, neg_used, neg_delta, lam = \
                self.__synthesizeProperty([], phi, pos, neg_must, [], lam_functions, True, False)
            lam_functions = union_dict(lam_functions, lam)

            stat = self.__statisticsCurrentProperty(pos, neg_must, neg_may, neg_used, neg_delta)
            self.__statistics.append(stat)
            self.__resetStatistics()

            phi_list.append(phi)

            funs = []
            for function_name, code in lam_functions.items():
                if phi.find(function_name) >= 0:
                    funs.append((function_name, code))

            fun_list.append(funs)

            if self.__verbose:
                print("Obtained a best L-property")
                print(phi + '\n')

                for function_name, code in lam_functions.items():
                    if function_name in phi:
                        print(code + '\n')

            self.__outer_iterator += 1
            self.__inner_iterator = 0

    def __statisticsCurrentProperty(self, pos, neg_must, neg_may, neg_used, neg_delta):
        statistics = {}

        statistics["num_pos"] = len(pos)
        statistics["num_neg_must"] = len(neg_must)
        statistics["num_neg_may"] = len(neg_may)
        statistics["num_neg_used"] = len(neg_used)
        statistics["num_neg_delta"] = len(neg_delta)

        avg_time_synthesis = self.__time_synthesis / self.__num_synthesis \
            if self.__num_synthesis > 0 else 0

        statistics["num_synthesis"] = self.__num_synthesis
        statistics["time_synthesis"] = self.__time_synthesis
        statistics["avg_time_synthesis"] = avg_time_synthesis
        statistics["max_time_synthesis"] = self.__max_time_synthesis

        avg_time_maxsat = self.__time_maxsat / self.__num_maxsat \
            if self.__num_maxsat > 0 else 0

        statistics["num_maxsat"] = self.__num_maxsat
        statistics["time_maxsat"] = self.__time_maxsat
        statistics["avg_time_maxsat"] = avg_time_maxsat
        statistics["max_time_maxsat"] = self.__max_time_maxsat

        avg_time_soundness = self.__time_soundness / self.__num_soundness \
            if self.__num_soundness > 0 else 0

        statistics["num_soundness"] = self.__num_soundness
        statistics["time_soundness"] = self.__time_soundness
        statistics["avg_time_soundness"] = avg_time_soundness
        statistics["max_time_soundness"] = self.__max_time_soundness

        avg_time_precision = self.__time_precision / self.__num_precision \
            if self.__num_precision > 0 else 0

        statistics["num_precision"] = self.__num_precision
        statistics["time_precision"] = self.__time_precision
        statistics["avg_time_precision"] = avg_time_precision
        statistics["max_time_precision"] = self.__max_time_precision

        statistics["time_conjunct"] = self.__time_synthesis
        statistics["time_conjunct"] += self.__time_maxsat
        statistics["time_conjunct"] += self.__time_soundness
        statistics["time_conjunct"] += self.__time_precision

        statistics["last_call"] = self.__time_last_query

        return statistics

    def __resetStatistics(self):
        self.__num_soundness = 0
        self.__num_precision = 0
        self.__num_maxsat = 0
        self.__num_synthesis = 0

        self.__time_soundness = 0
        self.__time_precision = 0
        self.__time_maxsat = 0
        self.__time_synthesis = 0

        self.__max_time_synthesis = 0
        self.__max_time_maxsat = 0
        self.__max_time_soundness = 0
        self.__max_time_precision = 0

        self.__time_last_query = 0

    def __statisticsFromList(self, l):
        if len(l) == 0:
            return (0, 0, 0, 0)
        else:
            return (sum(l), sum(l) / len(l), max(l))

    def __statisticsList(self):
        statistics = []

        # The last cycle is not used as a clause, since it doesn't change any behavior
        num_conjunct = len(self.__statistics) - 1

        nums_synthesis = []
        nums_maxsat = []
        nums_soundness = []
        nums_precision = []

        times_synthesis = []
        times_maxsat = []
        times_soundness = []
        times_precision = []
        times_conjunct = []

        max_times_synthesis = []
        max_times_maxsat = []
        max_times_soundness = []
        max_times_precision = []

        num_pos = []
        num_neg_must = []
        num_neg_may = []
        num_neg_used = []
        num_neg_delta = []

        last_calls = []

        for conj_statistics in self.__statistics:
            nums_synthesis.append(conj_statistics["num_synthesis"])
            nums_maxsat.append(conj_statistics["num_maxsat"])
            nums_soundness.append(conj_statistics["num_soundness"])
            nums_precision.append(conj_statistics["num_precision"])

            times_synthesis.append(conj_statistics["time_synthesis"])
            times_maxsat.append(conj_statistics["time_maxsat"])
            times_soundness.append(conj_statistics["time_soundness"])
            times_precision.append(conj_statistics["time_precision"])
            times_conjunct.append(conj_statistics["time_conjunct"])

            max_times_synthesis.append(conj_statistics["max_time_synthesis"])
            max_times_maxsat.append(conj_statistics["max_time_maxsat"])
            max_times_soundness.append(conj_statistics["max_time_soundness"])
            max_times_precision.append(conj_statistics["max_time_precision"])

            num_pos.append(conj_statistics["num_pos"])
            num_neg_must.append(conj_statistics["num_neg_must"])
            num_neg_may.append(conj_statistics["num_neg_may"])
            num_neg_used.append(conj_statistics["num_neg_used"])
            num_neg_delta.append(conj_statistics["num_neg_delta"])

            last_calls.append(conj_statistics["last_call"])

        total_num_synth, avg_num_synth, max_num_synth = self.__statisticsFromList(nums_synthesis)
        total_num_maxsat, avg_num_maxsat, max_num_maxsat = self.__statisticsFromList(nums_maxsat)
        total_num_soundness, avg_num_soundness, max_num_soundness = self.__statisticsFromList(nums_soundness)
        total_num_precision, avg_num_precision, max_num_precision = self.__statisticsFromList(nums_precision)

        num_query = total_num_synth
        num_query += total_num_maxsat
        num_query += total_num_soundness
        num_query += total_num_precision

        total_time, avg_time, max_time = self.__statisticsFromList(times_conjunct)
        avg_time_per_query = total_time / num_query if num_query > 0 else 0

        total_time_synthesis, avg_time_synthesis_per_clause, total_max_time_synthesis = \
            self.__statisticsFromList(times_synthesis)
        avg_time_synthesis = total_time_synthesis / total_num_synth if total_num_synth > 0 else 0
        _, avg_max_synthesis, max_max_synthesis = self.__statisticsFromList(max_times_synthesis)

        total_time_maxsat, _, _ = \
            self.__statisticsFromList(times_maxsat)

        total_time_soundness, _, _ = \
            self.__statisticsFromList(times_soundness)

        total_time_precision, _, _ = \
            self.__statisticsFromList(times_precision)

        total_last, _, _ = self.__statisticsFromList(last_calls)

        last = self.__statistics[-1]

        statistics.append(num_conjunct)
        statistics.append(total_num_synth + total_num_maxsat)
        statistics.append(total_time_synthesis + total_time_maxsat)
        statistics.append(total_num_soundness)
        statistics.append(total_time_soundness)
        statistics.append(total_num_precision)
        statistics.append(total_time_precision)
        statistics.append(total_last)
        statistics.append(last["time_conjunct"])
        statistics.append(total_time)

        return statistics

    def run(self):
        """Run the property synthesis process."""
        if self.__write_log:
            self.__logfile = self.__open_logfile(self.__tempfile_name)

        phi_list, fun_list = self.__synthesizeAllProperties()
        statistics = self.__statisticsList()

        if self.__write_log:
            self.__logfile.close()

        return phi_list, fun_list, statistics
