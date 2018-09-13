import os
import sys

def get_sections():
    assert len(sys.argv) == 2
    index_md = sys.argv[1]
    dirname = os.path.dirname(index_md)

    start = False
    sections = []
    with open(index_md) as f:
        for line in f:
            line = line.rstrip().lstrip()
            if ':maxdepth:' in line:
                start = True
                continue
            elif line == '```':
                break
            if start and len(line) > 1:
                sections.append(os.path.join(dirname, line + '.md'))
    return ' '.join(sections)


def get_chapters():
    assert len(sys.argv) == 2
    index_md = sys.argv[1]

    start = False
    chapters = []
    with open(index_md) as f:
        for line in f:
            line = line.rstrip().lstrip()
            if ':maxdepth:' in line:
                start = True
                continue
            elif line == '```':
                break
            if start and len(line) > 1:
                chapters.append(line.split('/')[0])
    return ' '.join(chapters)

