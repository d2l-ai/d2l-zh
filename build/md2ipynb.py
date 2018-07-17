import sys
import os
import time
import notedown
import nbformat

assert len(sys.argv) == 3, 'usage: input.md output.ipynb'

def is_ascii(character):
    return ord(character) <= 128

def add_space_between_ascii_and_non_ascii(string):
    punc = (' ','\n','\t','\r','，','。','？','！','、','；','：','“',
            '”','（','）','【','】','—','…','《','》')
    if len(string) == 0:
        return ''
    ret = string[0]
    for i in range(1, len(string)):
        if ((is_ascii(string[i-1]) != is_ascii(string[i]))
            and (string[i-1] not in punc)
            and (string[i] not in punc)):
            ret += ' '
        ret += string[i]
    return ret
# timeout for each notebook, in sec
timeout = 20 * 60

# the files will be ingored for execution
ignore_execution = []

input_fn = sys.argv[1]
output_fn = sys.argv[2]

reader = notedown.MarkdownReader(match='strict')

do_eval = int(os.environ.get('EVAL', True))

# read
with open(input_fn, 'r') as f:
    notebook = reader.read(f)

for c in notebook.cells:
    c.source = add_space_between_ascii_and_non_ascii(c.source)

if do_eval and not any([i in input_fn for i in ignore_execution]):
    tic = time.time()
    notedown.run(notebook, timeout)
    print('=== Finished evaluation in %f sec'%(time.time()-tic))

# write
# need to add language info to for syntax highlight
notebook['metadata'].update({'language_info':{'name':'python'}})

with open(output_fn, 'w') as f:
    f.write(nbformat.writes(notebook))
