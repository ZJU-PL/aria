"""
Parser of QBF
"""


class PaserQDIMACS:
    # ==========================================================================================
    # Parses prefix lines:
    def parse_prefix(self, prefix_lines):
        # we can merge prefix lines if there are the same quatifier type:
        previous_qtype = ''

        for line in prefix_lines:
            # asserting the line is part of prefix:
            assert ("e " in line or "a " in line)
            # removing spaces
            cur_qtype = line[0]
            cur_var_list = line[2:].split(" ")[:-1]
            # print(cur_var_list)

            # changing to integers:
            int_cur_var_list = []

            for var in cur_var_list:
                int_cur_var_list.append(int(var))
                # adding input gates to all gates:
                if int(var) not in self.all_vars:
                    self.all_vars.append(int(var))

            # we are in the same quantifier block:
            if previous_qtype == cur_qtype:
                self.parsed_prefix[-1][1].extend(int_cur_var_list)
            else:
                # a tuple of type and the var list:
                self.parsed_prefix.append((cur_qtype, int_cur_var_list))
                previous_qtype = cur_qtype

            # print((cur_qtype,int_cur_var_list))

        # assert the quantifier are alternating in the parsed_prefix:
        for i in range(len(self.parsed_prefix) - 1):
            assert (self.parsed_prefix[i][0] != self.parsed_prefix[i + 1][0])

    # ==========================================================================================

    # Reads qdimacs format:
    # Parses prefix lines:
    def parse_qdimacs_format(self):

        f = open(self.input_file, "r")
        lines = f.readlines()
        f.close()

        prefix_lines = []

        # seperate prefix, output gate and inner gates:
        for line in lines:
            # print(line)
            # we strip if there are any next lines or empty spaces:
            stripped_line = line.strip("\n").strip(" ")
            # we ignore if comment or empty line:
            if line == "":
                continue
            elif line[0] == "c":
                continue
            elif line[0] == "p":
                self.preamble = stripped_line.split(" ")
            # if exists/forall in the line then it is part of prefix:
            elif "e " in stripped_line or "a " in stripped_line:
                prefix_lines.append(stripped_line)
            else:
                self.clauses.append(stripped_line)

        # print(prefix_lines)

        # Parse prefix lines:
        self.parse_prefix(prefix_lines)

    # we flip universal layers in first k layers and add assumption TODO:
    def flip_and_assume(self, k, assum, assertions):

        flipped_and_assumed_string = ''

        # printing preamble:
        flipped_and_assumed_string += " ".join(self.preamble[:-1]) + " " + str(
            int(self.preamble[-1]) + len(assum) + len(assertions)) + "\n"

        first_layers = ""
        for i in range(len(self.parsed_prefix)):
            layer_string = " ".join(str(x) for x in self.parsed_prefix[i][1])
            if i < k:
                first_layers += " " + layer_string
            elif self.parsed_prefix[i][0] == "e":
                # we merge the first layer:
                if i == k:
                    flipped_and_assumed_string += "e " + layer_string + " 0\n"
                else:
                    flipped_and_assumed_string += "e " + layer_string + " 0\n"
            else:
                flipped_and_assumed_string += "a " + layer_string + " 0\n"

        # appending the flipping at the end:
        flipped_and_assumed_string += "e " + first_layers + " 0\n"

        # adding assumption clauses:
        for var in assum:
            flipped_and_assumed_string += str(var) + " 0\n"

        for clause in assertions:
            flipped_and_assumed_string += " ".join(str(x) for x in clause) + " 0\n"

        # printing all the gates to the file:
        for line in self.clauses:
            flipped_and_assumed_string += line + "\n"

        return flipped_and_assumed_string

    # we append the certificate to the current instance
    # other than the shared variables, we renumber the rest of certificate varaibles:
    # write the large instance to the file directly:
    def sat_renumber_and_append_wrf(self, certificate, shared_vars):

        f = open("intermediate_files/appended_instance.qdimacs", "w")

        clauses_string = ""
        # first writing the instance clauses to the new file:
        for clause in self.clauses:
            clauses_string += clause + "\n"
            # f.write(clause + "\n")

        # we start from max variable + 1 in the instance:
        cur_max_var = int(self.preamble[2]) + 1
        # remembering for the preamble:
        max_var = int(self.preamble[2]) + 1

        cert_vars_map = dict()

        # iterating through certificate clauses:
        for clause in certificate.clauses:
            new_cur_clause = []
            for var in clause:
                if var > 0:
                    non_negated_int = var
                else:
                    non_negated_int = -var
                if non_negated_int in shared_vars:
                    new_cur_clause.append(var)
                else:
                    if non_negated_int in cert_vars_map:
                        if int(var) > 0:
                            new_cur_clause.append(cert_vars_map[non_negated_int])
                        else:
                            new_cur_clause.append(-cert_vars_map[non_negated_int])
                    else:
                        # adding the new var in to dict:
                        cert_vars_map[non_negated_int] = cur_max_var
                        if int(var) > 0:
                            new_cur_clause.append(cur_max_var)
                        else:
                            new_cur_clause.append(-cur_max_var)
                        cur_max_var += 1

                    # print(var)
            clauses_string += " ".join(str(var) for var in new_cur_clause) + " 0\n"
            # f.write(" ".join(str(var) for var in new_cur_clause) + " 0\n")
        # f.close()

        # lastly appending the preamble with computed vars and clauses:
        prefix_string = ''

        # printing preamble:
        prefix_string += "p cnf " + str(cur_max_var - 1) + " " + str(
            int(self.preamble[3]) + len(certificate.clauses)) + "\n"

        # then appending the prefix with new variables:
        for layer in self.parsed_prefix:
            prefix_string += layer[0] + " " + " ".join(str(var) for var in layer[1]) + " 0\n"

        # print(self.parsed_prefix)

        # adding extra variables:
        last_layer = "e "
        for var in range(max_var, cur_max_var):
            last_layer += str(var) + " "

        last_layer += "0\n"
        # print(last_layer)

        # we append the prefix string to the head of the file:
        f.write(prefix_string)
        f.write(last_layer)
        f.write(clauses_string)
        f.close()

    # we append the certificate to the current instance
    # other than the shared variables, we renumber the rest of certificate varaibles:
    # we flip the shared universal variables and add to the end:
    # write the large instance to the file directly:
    def unsat_renumber_and_append_wrf(self, certificate, shared_vars):

        f = open("intermediate_files/appended_instance.qdimacs", "w")

        clauses_string = ""
        # first writing the instance clauses to the new file:
        for clause in self.clauses:
            clauses_string += clause + "\n"
            # f.write(clause + "\n")

        # we start from max variable + 1 in the instance:
        cur_max_var = int(self.preamble[2]) + 1
        # remembering for the preamble:
        max_var = int(self.preamble[2]) + 1

        cert_vars_map = dict()

        # iterating through certificate clauses:
        for clause in certificate.clauses:
            new_cur_clause = []
            for var in clause:
                if var > 0:
                    non_negated_int = var
                else:
                    non_negated_int = -var
                if non_negated_int in shared_vars:
                    new_cur_clause.append(var)
                else:
                    if non_negated_int in cert_vars_map:
                        if int(var) > 0:
                            new_cur_clause.append(cert_vars_map[non_negated_int])
                        else:
                            new_cur_clause.append(-cert_vars_map[non_negated_int])
                    else:
                        # adding the new var in to dict:
                        cert_vars_map[non_negated_int] = cur_max_var
                        if int(var) > 0:
                            new_cur_clause.append(cur_max_var)
                        else:
                            new_cur_clause.append(-cur_max_var)
                        cur_max_var += 1

                    # print(var)
            clauses_string += " ".join(str(var) for var in new_cur_clause) + " 0\n"
            # f.write(" ".join(str(var) for var in new_cur_clause) + " 0\n")
        # f.close()

        # lastly appending the preamble with computed vars and clauses:
        prefix_string = ''

        # printing preamble:
        prefix_string += "p cnf " + str(cur_max_var - 1) + " " + str(
            int(self.preamble[3]) + len(certificate.clauses)) + "\n"

        # remembering shared universal varibles:
        shared_universal_variables = []

        # then appending the prefix with new variables:
        for layer in self.parsed_prefix:
            # we only add universal variables which are not in the share variables:
            if layer[0] == 'a':
                universal_string = ""
                for var in layer[1]:
                    if var not in shared_vars:
                        universal_string += str(var) + " "
                    else:
                        # adding the variables to shared universal list, we will add in the inner most layer:
                        shared_universal_variables.append(var)
                # if some non-shared universal variables present:
                if universal_string != "":
                    prefix_string += layer[0] + " " + universal_string + "0\n"
            else:
                prefix_string += layer[0] + " " + " ".join(str(var) for var in layer[1]) + " 0\n"

        # print(self.parsed_prefix)

        # adding extra variables to the last layer:
        last_layer = "e "
        for var in range(max_var, cur_max_var):
            last_layer += str(var) + " "

        last_layer += "0\n"
        # print(last_layer)

        # adding the shared universal variables to the inner layer:
        shared_universal_variable_layer = "e " + " ".join(str(var) for var in shared_universal_variables) + " 0\n"

        # we append the prefix string to the head of the file:
        f.write(prefix_string)
        # if cur_max_var is more than max_var, we have extra variables
        if max_var != cur_max_var:
            f.write(last_layer)
        # if there are any shared universal layers we add the extra existential layer:
        if len(shared_universal_variables) != 0:
            f.write(shared_universal_variable_layer)
        f.write(clauses_string)
        f.close()

    # assuming no open variables:
    def __init__(self, input_qbf):
        self.input_file = input_qbf
        self.preamble = []
        self.parsed_prefix = []
        self.clauses = []
        # remembering all gates for future assumptions:
        self.all_vars = []

        self.parse_qdimacs_format()

        # print(self.parsed_prefix)
