import glob
import nbformat
import notedown
import os
from subprocess import check_output
import sys
import time


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

# timeout for each notebook, in sec
timeout = 20 * 60

# the files will be ingored for execution
ignore_execution = []

reader = notedown.MarkdownReader(match='strict')

do_eval = int(os.environ.get('EVAL', True))

for chap in glob.glob('chapter_*'):
    mkdir_if_not_exist(['build', 'win_ipynb', chap])
    mds = filter(lambda x: x.endswith('md'), os.listdir(chap))
    for md in mds:
        if md != 'index.md':
            in_md = os.path.join(chap, md)
            out_nb = os.path.join('build', in_md[:-2] + '.ipynb')
            print('---', in_md)
            # read
            with open(in_md, 'r', encoding="utf8") as f:
                notebook = reader.read(f)

            if do_eval and not any([i in input_fn for i in ignore_execution]):
                tic = time.time()
                notedown.run(notebook, timeout)
                print('=== Finished evaluation in %f sec'%(time.time()-tic))

            # write
            # need to add language info to for syntax highlight
            notebook['metadata'].update({'language_info':{'name':'python'}})

            with open(out_nb, 'w', encoding="utf8") as f:
                f.write(nbformat.writes(notebook))


