#!/usr/bin/env bash
# Render the Daru-Tenaud 3D viscous shock-tube z-mid density slice + upload.
# Reads /tmp/mbq/shktube3d.txt ("x y rho"), writes /tmp/mbq/fig_shktube3d.png,
# copies to /mnt/c/Users/user/cfdtmp/, uploads to tmpfiles.org (prints /dl/ url).
set -u
SLICE=/tmp/mbq/shktube3d.txt
PNG=/tmp/mbq/fig_shktube3d.png

python3 - "$SLICE" "$PNG" <<'PY'
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

slice_path, png = sys.argv[1], sys.argv[2]
d = np.loadtxt(slice_path)
x, y, rho = d[:,0], d[:,1], d[:,2]
tri = mtri.Triangulation(x, y)
fig, ax = plt.subplots(figsize=(11, 5.5))
levels = np.linspace(rho.min(), rho.max(), 40)
cf = ax.tricontourf(tri, rho, levels=levels, cmap='jet')
ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('3D viscous shock tube (Daru-Tenaud) Re=200 deg3t-BVD+MLP t=1.0')
fig.colorbar(cf, ax=ax, label='density', shrink=0.85)
fig.tight_layout()
fig.savefig(png, dpi=120)
print(f"Plot saved: {png}  rho=[{rho.min():.4f},{rho.max():.4f}] npts={len(rho)}")
PY

# copy to the Windows-visible folder
mkdir -p /mnt/c/Users/user/cfdtmp 2>/dev/null
cp "$PNG" /mnt/c/Users/user/cfdtmp/ 2>/dev/null && echo "copied to /mnt/c/Users/user/cfdtmp/"

# upload to tmpfiles.org (script file ⇒ avoids the inline @-path mangling)
RESP=$(curl -s -F "file=@${PNG}" https://tmpfiles.org/api/v1/upload)
echo "RAW_UPLOAD: $RESP"
# turn the page url into a direct /dl/ link
URL=$(echo "$RESP" | grep -oE 'https://tmpfiles.org/[0-9]+/[^"\\]+' | head -1)
if [ -n "$URL" ]; then
  DL=$(echo "$URL" | sed 's#tmpfiles.org/#tmpfiles.org/dl/#')
  echo "DL_URL: $DL"
fi
