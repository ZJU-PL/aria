from .benchmark import Benchmark
from .config import args
from .util import warning, die
import random
import numpy
random.seed(args.rng)
numpy.random.seed(args.rng)
