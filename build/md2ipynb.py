import sys
import os
import time
import notedown
import nbformat

assert len(sys.argv) == 3, 'usage: input.md output.ipynb'

# timeout in second to evaluate a notebook
timeout = 1000
# limit the number of lines in a cell output
max_output_length = 500
# the files will be ingored for execution
ignore_execution = []

input_fn = sys.argv[1]
output_fn = sys.argv[2]

reader = notedown.MarkdownReader(match='strict')

do_eval = int(os.environ.get('EVAL', True))
# do_eval = False

# read
with open(input_fn, 'r') as f:
    notebook = reader.read(f)

if do_eval and not any([i in input_fn for i in ignore_execution]):
    print('=== Evaluate %s with timeout %d sec'%(input_fn, timeout))
    tic = time.time()
    notedown.run(notebook, timeout)
    print('=== Finished in %f sec'%(time.time()-tic))

# write
# need to add language info to for syntax highlight
notebook['metadata'].update({'language_info':{'name':'python'}})

with open(output_fn, 'w') as f:
    f.write(nbformat.writes(notebook))
