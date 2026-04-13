
import os

def read_tasks_files(pddldir: str):
    """
    Returns a list of all planning tasks in the pddl test directory.
    """
    tasks = list()
    for root, dirs, files in os.walk(pddldir):
        for dir_name in dirs:
            planningtask_dir = os.path.join(root, dir_name)
            domainfile       = os.path.join(planningtask_dir, "domain.pddl")
            problemfile      = os.path.join(planningtask_dir, "problem.pddl")
            domainname       = os.path.basename(planningtask_dir)
            if "__pycache__" in domainfile: continue
            tasks.append((domainname, domainfile, problemfile))
    return tasks

# airport-nontemporal-adl/,2
# barman-sequential-optimal/,9
# block-grouping/,2
# cave-diving-sequential-optimal/,9
# child-snack-sequential-optimal/,7
# city-car-sequential-optimal/, 5
# counters/,1
# delivery/,10
# depots/,10
# drone/,4
# expedition/,30
# ext-plant-watering/,62
# farmland/,55
# floor-tile-sequential-optimal/,4
# fo-counters/,2
# fo-farmland/,8
# fo-sailing/,63
# genome-edit-distances-sequential-optimal/,1
# gripper-round-1-adl/,11
# hiking-sequential-optimal/,11
# hydropower/,16
# logistics-round-1-adl/,13
# logistics-strips-typed/,20
# maintenance-sequential-optimal/,4
# markettrader/,8
# movie-round-1-adl/,7
# mprime/,5
# no-mystery-sequential-optimal/,3
# openstacks-sequential-optimal-adl/,7
# parc-printer-sequential-optimal-strips/,8
# parking-sequential-optimal/,3
# pathways-propositional/,6
# pathwaysmetric/,5
# peg-solitaire-sequential-optimal-strips/,2
# petrobras/,5
# pipesworld-propositional/,1
# pipesworld-tankage-nontemporal-strips/,1
# plant-watering/,5
# rover/,17
# rovers-propositional/,10
# sailing/,174
# satellite-numeric-strips/,5
# satellite/,7
# scanalyzer-3d-sequential-optimal-strips/,2
# schedule-adl-typed/,2
# settlers-strips/,??
# settlers/,??
# sokoban-sequential-optimal-strips/,5
# sugar/,5
# tetris-sequential-optimal/,2
# tpp-propositional/,5
# tpp/,3
# transport-sequential-optimal-strips/,5
# trucks-propositional/,13
# visit-all-sequential-optimal/,3
# woodworking-sequential-optimal-strips/,3
# zenotravel/,12
