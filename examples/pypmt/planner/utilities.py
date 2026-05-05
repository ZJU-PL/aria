import os
import z3

def saveToFile(content, path):
    with open(path, 'w') as f:
        f.write(str(content))

def dumpProblem(solver, file_path=None, add_check_sat=True):
    """ Dumps the current state of the asserts/vars to a file/stdout """
    # this is done to avoid printing lots of lets
    z3.set_option("pp.min_alias_size", 1000000)
    z3.set_option("pp.max_depth",1000000)
    content = str(solver.sexpr())

    if add_check_sat:
        content += "\n(check-sat)\n"

    if file_path:
        saveToFile(content, file_path)
    return content

def dumpModel(solver, file_path=None):
    """ Dumps the model """
    model = solver.model()
    content = []
    for var in model:
        content.append(f"Variable: {var}\t\t\t\t Value: {model[var]}")
    if file_path:
        saveToFile(content, file_path)
    return "\n".join(sorted(content))

def dumpSolver(solver, abs_file_path, actionordering = []):

    dirname = os.path.dirname(abs_file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'action-ordering'), 'w') as f:
        for item in actionordering:
            f.write("%s\n" % item)

    with open(abs_file_path, 'w') as f:
        f.write(str(solver.sexpr()))
