import git
import pathlib
r=git.Repo(pathlib.Path(__file__).parent.parent.absolute())
__version__ = str(r.head.commit)
if r.is_dirty():
    __version__ += '+dirty'
del r
