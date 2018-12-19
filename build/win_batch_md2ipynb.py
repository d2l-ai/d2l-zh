from distutils.dir_util import copy_tree
import glob
import nbformat
import notedown
import os
from subprocess import check_output
import sys
import time

# To access data/imgs/gluonbook in upper level.
os.chdir('build')

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

# Timeout for each notebook, in sec
timeout = 60 * 60

# The files will be ingored for execution
ignore_execution = ['chapter_computational-performance/async-computation.md']

reader = notedown.MarkdownReader(match='strict')

do_eval = int(os.environ.get('EVAL', True))


for chap in glob.glob(os.path.join('..', 'chapter_*')):
    mkdir_if_not_exist(['win_ipynb', chap[3:]])
    mds = filter(lambda x: x.endswith('md'), os.listdir(chap))
    for md in mds:
        if md != 'index.md':
            in_md = os.path.join(chap, md)
            out_nb = os.path.join('win_ipynb', in_md[3:-2] + 'ipynb')

            if not os.path.exists(out_nb):

                print('---', in_md[3:])
                # read
                with open(in_md, 'r', encoding="utf8") as f:
                    notebook = reader.read(f)

                if do_eval and chap[3:] + '/' + md not in ignore_execution:
                    tic = time.time()
                    notedown.run(notebook, timeout)
                    print('=== Finished evaluation in %f sec'%(time.time()-tic))

                # write
                # need to add language info to for syntax highlight
                notebook['metadata'].update({'language_info':{'name':'python'}})

                with open(out_nb, 'w', encoding="utf8") as f:
                    f.write(nbformat.writes(notebook))


