import os 

def checkfile(filename, force=False, comm=None, root=0):
    if comm is None:
        if os.path.exists(filename) and not force:
            return True
        else:
            return False
    else:
        rank = comm.rank
        if rank == root:
            if os.path.exists(filename) and not force:
                judge = True
            else:
                judge = False
        else:
            judge = None
        judge = comm.bcast(judge, root=root)
        return judge