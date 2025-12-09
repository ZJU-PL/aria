"""
Plotting the DNNF, OBDD, etc., given the input files.
"""

import copy


def export_dtree_file(output_dtree_file, dtree):
    output = open(output_dtree_file, 'w')
    output.write('dtree {}\n'.format(dtree.node_id + 1))
    output.close()
    output = open(output_dtree_file, 'a')
    dtree.print_info([], output_dtree_file)
    output.close()


def export_dtree_dot(input_dtree_file):
    output_dtree_file = input_dtree_file + '.dot'
    output = open(output_dtree_file, 'w')

    output.write('graph ' + input_dtree_file.split('/')[1] + '{\n')
    output.write('      rankdir=TB;\n')
    output.write('      size="8,5";\n')
    output.write('      node [fontname="Arial"];\n\n')

    node, leaf = [], []
    for line in open(input_dtree_file):
        if line.startswith('dtree'):
            nb_nodes = line.split()[1]

        if line.startswith('L'):
            node.append(line.split()[1])
            leaf.append(line.split()[1])
            output.write('      ' + line.split()[1] + ' [shape=circle];\n')
        if line.startswith('I'):
            l, r = line.split()[1:]
            node.append('I{}'.format(len(node)))
            output.write('      {} [shape=square];\n'.format(node[-1]))
            output.write('      {} -- {};\n'.format(node[-1], node[int(l)]))
            output.write('      {} -- {};\n'.format(node[-1], node[int(r)]))

    output.write('      {rank=same;')
    for lll in leaf:
        output.write('{}; '.format(lll))
    output.write('}\n')
    output.write('}')
    output.close()

    return int(nb_nodes)


def export_nnf_file(output_dnnf_file, nnf):
    output = open(output_dnnf_file, 'w')
    aux = copy.deepcopy(nnf)
    nb_nodes = aux.count_node(0)
    nb_edges = aux.count_edge()
    nb_vars = max(aux.collect_var())
    output.write('nnf {} {} {}\n'.format(nb_nodes, nb_edges, nb_vars))
    output.close()
    output = open(output_dnnf_file, 'a')
    aux = copy.deepcopy(nnf)
    aux.print_nnf(current_id=0, output_file=output_dnnf_file)
    output.close()


def export_nnf_dot(input_nnf_file):
    output_nnf_file = input_nnf_file + '.dot'
    output = open(output_nnf_file, 'w')

    output.write('graph ' + input_nnf_file.split('/')[1] + '{\n')
    output.write('      rankdir=TB;\n')
    output.write('      size="8,5";\n')
    output.write('      node [fontname="Arial"];\n\n')

    node = []
    leaf = []
    nb_nodes, nb_edges, nb_vars = 0, 0, 0
    nb_vars = None
    for line in open(input_nnf_file):
        if line.startswith('nnf'):
            str_nb_nodes, str_nb_edges, str_nb_vars = line.split()[1:4]
            nb_nodes, nb_edges, nb_vars = int(str_nb_nodes), int(str_nb_edges), int(str_nb_vars)
            print('Nb nodes: ', nb_nodes)
            print('Nb edges: ', nb_edges)
            print('Nb vars : ', nb_vars)

        if line.startswith('L'):
            lit = line.split()[1]
            node.append(lit)
            leaf.append(lit)

        if line.startswith('A'):
            nb_childs = line.split()[1]
            childs = line.split()[2:]
            # print('And-node ({0} childs) : {1}'.format(nb_childs, childs))
            output.write('      AND{} [shape=square, label="AND"];\n'.format(len(node)))
            node.append('AND{}'.format(len(node)))
            for child in childs:
                output.write('      AND{0} -- {1};\n'.format(len(node) - 1, node[int(child)]))
        if line.startswith('O'):
            ignore, nb_childs = line.split()[1:3]
            childs = line.split()[3:]
            if int(ignore) > 0:
                # assert int(ignore) <= int(nb_vars)
                assert int(nb_childs) == 2
                # print('Or-node ({0} childs) : {1}'.format(nb_childs, childs))
                output.write('      OR{} [shape=diamond, label="OR"];\n'.format(len(node)))
                node.append('OR{}'.format(len(node)))
                for child in childs:
                    output.write('      OR{0} -- {1};\n'.format(len(node) - 1, node[int(child)]))

    output.write('      {rank=same;')
    for lll in leaf:
        output.write('{}; '.format(lll))
    output.write('}\n')
    output.write('}')
    output.close()
    return nb_nodes, nb_edges, nb_vars


def export_dot_from_bdd(output_bdd_file, bdd, nvars):
    output = open(output_bdd_file, 'w')

    output.write('digraph ' + output_bdd_file.split('/')[-1].split('.')[0] + '{\n')
    output.write('      rankdir=TB;\n')
    output.write('      size="8,5";\n')
    output.write('      node [fontname="Arial"];\n\n')
    output.close()

    rank = bdd.print_info(nvars, output_bdd_file)

    output = open(output_bdd_file, 'a')
    for i in range(len(rank)):
        output.write('      {rank=same; ')
        for node_id in rank[i]:
            output.write('{}; '.format(node_id))
        output.write('}\n')
    output.write('}')
    output.close()
    return 0
