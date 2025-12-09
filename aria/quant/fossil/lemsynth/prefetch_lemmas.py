import fileinput
import sys

import time

# Writes to a file everytime it gets something from the input stream for num_lines number of lines
# Filename and num_lines are passed as arguments to the script

# Slightly hacky solution
# This is the only version that works despite timeouts and everything

line_counter = 0
outfile_name = sys.argv[1]
num_lines = int(sys.argv[2])

# Erase file if not already empty
open(outfile_name,'w').close()

with fileinput.input(files='-') as f:
    for line in f:
        outfile_handle = open(outfile_name,'a')
        outfile_handle.write(str(time.time()) + ': ')
        outfile_handle.write(line)
        outfile_handle.close()
        line_counter = line_counter + 1
        if line_counter == num_lines:
            fileinput.close()
            exit(0)
