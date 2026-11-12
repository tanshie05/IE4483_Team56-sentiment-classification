#!/bin/bash
# Optional cluster helper. Stops the calling job unless the project folder has at least $1 GB free.
# On a CephFS quota (ceph.quota.max_bytes extended attribute) it checks the quota; otherwise it checks the filesystem's
# free space. Run it after loading Python (compute nodes may lack getfattr, so Python reads the attributes).

# ---- Edit these for your cluster -------------------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/path/to/your/project/storage}   # folder whose free space is checked
QUOTA_DIR=${QUOTA_DIR:-$PROJECT_DIR}                          # folder that carries the quota attributes, if different
# -----------------------------------------------------------------------------------------------------------------------

set -eo pipefail
python - "$1" "$QUOTA_DIR" <<'PYEOF'
import os, shutil, sys
need, path = int(sys.argv[1]) * 1000000000, sys.argv[2]
try:
    quota = int(os.getxattr(path, 'ceph.quota.max_bytes'))
    used = int(os.getxattr(path, 'ceph.dir.rbytes'))
    free, source = quota - used, f'quota {quota}, used {used}'
except (OSError, AttributeError):
    free, source = shutil.disk_usage(path).free, 'filesystem free space'
print(f'{path}: {free} bytes free ({source}); this job needs {need}')
if free < need:
    sys.exit('STOPPING: not enough free space for this job.')
PYEOF
