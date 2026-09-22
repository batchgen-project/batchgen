import os

from batchgen.server.process_utils import hold_shm_objects


def test_holds_existing_objects_and_skips_missing(tmp_path):
    meta = tmp_path / "shm_meta"
    meta.write_bytes(b"x" * 16)
    fds = hold_shm_objects(("/shm_meta", "/shm_on_hugetlbfs"), shm_root=str(tmp_path))
    try:
        assert len(fds) == 1
        assert os.fstat(fds[0]).st_ino == meta.stat().st_ino
        meta.unlink()  # the held fd survives unlink, as a live server's would
        assert os.fstat(fds[0]).st_nlink == 0
    finally:
        for fd in fds:
            os.close(fd)
