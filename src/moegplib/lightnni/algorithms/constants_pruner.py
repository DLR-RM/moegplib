""" Constants for the one shot pruners.

Modified scripts from https://github.com/microsoft/nni
"""

from moegplib.lightnni.algorithms.one_shot import LevelPruner, L1FilterPruner, L2FilterPruner, FPGMPruner

PRUNER_DICT = {
    'level': LevelPruner,
    'l1': L1FilterPruner,
    'l2': L2FilterPruner,
    'fpgm': FPGMPruner
}
