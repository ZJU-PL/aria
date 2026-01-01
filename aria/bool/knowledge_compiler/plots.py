"""
Plotting the DNNF, OBDD, etc., given the input files.
"""

import copy


def export_dtree_file(output_dtree_file, dtree):
    """Export dtree to file."""
    with open(output_dtree_file, "w", encoding="utf-8") as output:
        output.write(f"dtree {dtree.node_id + 1}\n")
    dtree.print_info([], output_dtree_file)


def export_dtree_dot(input_dtree_file):
    """Export dtree to DOT format."""
    output_dtree_file = input_dtree_file + ".dot"
    nb_nodes = 0

    with open(input_dtree_file, "r", encoding="utf-8") as input_file:
        node, leaf = [], []
        for line in input_file:
            if line.startswith("dtree"):
                nb_nodes = line.split()[1]

            if line.startswith("L"):
                node.append(line.split()[1])
                leaf.append(line.split()[1])

            if line.startswith("I"):
                l, r = line.split()[1:]
                node.append(f"I{len(node)}")

    with open(output_dtree_file, "w", encoding="utf-8") as output:
        output.write(f'graph {input_dtree_file.split("/")[1]}{{\n')
        output.write("      rankdir=TB;\n")
        output.write('      size="8,5";\n')
        output.write('      node [fontname="Arial"];\n\n')

        with open(input_dtree_file, "r", encoding="utf-8") as input_file:
            for line in input_file:
                if line.startswith("L"):
                    output.write(f"      {line.split()[1]} [shape=circle];\n")
                if line.startswith("I"):
                    l, r = line.split()[1:]
                    output.write(f"      {node[-1]} [shape=square];\n")
                    output.write(f"      {node[-1]} -- {node[int(l)]};\n")
                    output.write(f"      {node[-1]} -- {node[int(r)]};\n")

        output.write("      {rank=same;")
        for lll in leaf:
            output.write(f"{lll}; ")
        output.write("}\n")
        output.write("}")

    return int(nb_nodes)


def export_nnf_file(output_dnnf_file, nnf):
    """Export NNF to file."""
    aux = copy.deepcopy(nnf)
    nb_nodes = aux.count_node(0)
    nb_edges = aux.count_edge()
    nb_vars = max(aux.collect_var())
    with open(output_dnnf_file, "w", encoding="utf-8") as output:
        output.write(f"nnf {nb_nodes} {nb_edges} {nb_vars}\n")
    aux = copy.deepcopy(nnf)
    aux.print_nnf(current_id=0, output_file=output_dnnf_file)


def export_nnf_dot(input_nnf_file):
    """Export NNF to DOT format."""
    output_nnf_file = input_nnf_file + ".dot"
    node = []
    leaf = []
    nb_nodes, nb_edges, nb_vars = 0, 0, 0

    with open(input_nnf_file, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.startswith("nnf"):
                str_nb_nodes, str_nb_edges, str_nb_vars = line.split()[1:4]
                nb_nodes = int(str_nb_nodes)
                nb_edges = int(str_nb_edges)
                nb_vars = int(str_nb_vars)
                print("Nb nodes: ", nb_nodes)
                print("Nb edges: ", nb_edges)
                print("Nb vars : ", nb_vars)

            if line.startswith("L"):
                lit = line.split()[1]
                node.append(lit)
                leaf.append(lit)

            if line.startswith("A"):
                nb_childs = line.split()[1]
                childs = line.split()[2:]
                node.append(f"AND{len(node)}")
            if line.startswith("O"):
                ignore, nb_childs = line.split()[1:3]
                childs = line.split()[3:]
                if int(ignore) > 0:
                    assert int(nb_childs) == 2
                    node.append(f"OR{len(node)}")

    with open(output_nnf_file, "w", encoding="utf-8") as output:
        output.write(f'graph {input_nnf_file.split("/")[1]}{{\n')
        output.write("      rankdir=TB;\n")
        output.write('      size="8,5";\n')
        output.write('      node [fontname="Arial"];\n\n')

        with open(input_nnf_file, "r", encoding="utf-8") as input_file:
            node_idx = 0
            for line in input_file:
                if line.startswith("L"):
                    lit = line.split()[1]
                    output.write(f"      {lit} [shape=circle];\n")
                if line.startswith("A"):
                    nb_childs = line.split()[1]
                    childs = line.split()[2:]
                    output.write(f'      AND{node_idx} [shape=square, label="AND"];\n')
                    for child in childs:
                        output.write(f"      AND{node_idx} -- {node[int(child)]};\n")
                    node_idx += 1
                if line.startswith("O"):
                    ignore, nb_childs = line.split()[1:3]
                    childs = line.split()[3:]
                    if int(ignore) > 0:
                        assert int(nb_childs) == 2
                        output.write(
                            f'      OR{node_idx} [shape=diamond, label="OR"];\n'
                        )
                        for child in childs:
                            output.write(f"      OR{node_idx} -- {node[int(child)]};\n")
                        node_idx += 1

        output.write("      {rank=same;")
        for lll in leaf:
            output.write(f"{lll}; ")
        output.write("}\n")
        output.write("}")

    return nb_nodes, nb_edges, nb_vars


def export_dot_from_bdd(output_bdd_file, bdd, nvars):
    """Export BDD to DOT format."""
    with open(output_bdd_file, "w", encoding="utf-8") as output:
        output.write(f'digraph {output_bdd_file.split("/")[-1].split(".")[0]}{{\n')
        output.write("      rankdir=TB;\n")
        output.write('      size="8,5";\n')
        output.write('      node [fontname="Arial"];\n\n')

    rank = bdd.print_info(nvars, output_bdd_file)

    with open(output_bdd_file, "a", encoding="utf-8") as output:
        for rank_list in rank:
            output.write("      {rank=same; ")
            for node_id in rank_list:
                output.write(f"{node_id}; ")
            output.write("}\n")
        output.write("}")
    return 0
