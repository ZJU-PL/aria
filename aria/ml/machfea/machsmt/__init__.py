import random

import numpy

from .benchmark import Benchmark
from .config import args
from .util import warning, die

random.seed(args.rng)  # pylint: disable=no-member
numpy.random.seed(args.rng)  # pylint: disable=no-member
