"""
pyDatalog

Copyright (C) 2013 Pierre Carbonnelle

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc.  51 Franklin St, Fifth Floor, Boston, MA 02110-1301
USA

"""

import copy
import threading

from . import py_engine, py_parser

class Logic(object):
    """ 
    per-thread singleton class containing the py_engine Logic in the current thread.
    Logic() resets the logic in the current thread and returns it
    Logic(True) returns the py_engine logic in the current thread, so that it can be passed to another thread.
    Logic(logic) initializes the logic in the current thread with logic, and returns it.
    """
    tl = threading.local()  # contains the Logic in the current thread
    def __new__(cls, logic=None):
        if isinstance(logic, cls):
            py_parser.clear()
            Logic.tl.logic = copy.copy(logic) 
            Logic.tl.logic.Subgoals = {} # for memoization of subgoals (tabled resolution)
            Logic.tl.logic.Tasks = None # LIFO stack of tasks
            Logic.tl.logic.Recursive_Tasks = None # FIFO queue of tasks for recursive clauses
            Logic.tl.logic.Recursive = False # True -> process Recursive_tasks. Otherwise, process Tasks
            Logic.tl.logic.Goal = None
            Logic.tl.logic.gc_uncollected = False # did we run gc.collect() yet ?
            py_engine.Fresh_var.tl.counter = 0
        elif not (logic) or not hasattr(Logic.tl, 'logic'):
            Logic.tl.logic = object.__new__(cls)
        return Logic.tl.logic
    
    def __init__(self, logic=None):
        if not (logic) or not (hasattr(self, 'Db')):
            py_parser.clear()
            py_engine.clear()  # make sure the singleton has what's needed
            
    def clear(self):
        """ move the logic to the current thread and clears it """
        Logic(self)  # just to be sure
        py_engine.clear()
                
py_engine.Logic = Logic  # share Logic with py_engine
