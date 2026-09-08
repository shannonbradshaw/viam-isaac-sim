"""Generate the low-poly RealSense D435 body mesh embedded in the fragment.

Authors a binary STL in millimetres, centred on the camera frame origin the
way the collision box it replaces was: body bar 90 x 25 x 25 with four lens
cylinders protruding 1.5 mm from the +z (optical) face. Prints the base64 of
the STL for `mesh_data`, or writes the raw STL next to this script with
--stl-out for inspection.

Usage: .venv/bin/python tools/generate_realsense_mesh.py [--stl-out PATH]
"""

from __future__ import annotations

import argparse
import base64
import math
import struct
import sys

BODY_X_MM = 90.0
BODY_Y_MM = 25.0
BODY_Z_MM = 25.0  # depth along the optical axis; the front face is +z
LENS_RADIUS_MM = 5.5
LENS_PROTRUSION_MM = 1.5
# four circles along the front, D435-style: imager, projector, imager, RGB
LENS_CENTRES_X_MM = (-30.0, -10.0, 10.0, 30.0)
LENS_SEGMENTS = 12

Vec = tuple[float, float, float]
Tri = tuple[Vec, Vec, Vec]


def _box_tris(lo: Vec, hi: Vec) -> list[Tri]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    c = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    # each face as two triangles, wound counter-clockwise seen from outside
    quads = [
        (0, 3, 2, 1),  # -z
        (4, 5, 6, 7),  # +z
        (0, 1, 5, 4),  # -y
        (2, 3, 7, 6),  # +y
        (0, 4, 7, 3),  # -x
        (1, 2, 6, 5),  # +x
    ]
    tris: list[Tri] = []
    for a, b, d, e in quads:
        tris.append((c[a], c[b], c[d]))
        tris.append((c[a], c[d], c[e]))
    return tris


def _cylinder_tris(cx: float, cy: float, z0: float, z1: float, radius: float) -> list[Tri]:
    ring = [
        (
            cx + radius * math.cos(2 * math.pi * i / LENS_SEGMENTS),
            cy + radius * math.sin(2 * math.pi * i / LENS_SEGMENTS),
        )
        for i in range(LENS_SEGMENTS)
    ]
    tris: list[Tri] = []
    for i in range(LENS_SEGMENTS):
        (ax, ay), (bx, by) = ring[i], ring[(i + 1) % LENS_SEGMENTS]
        tris.append(((ax, ay, z0), (bx, by, z0), (bx, by, z1)))
        tris.append(((ax, ay, z0), (bx, by, z1), (ax, ay, z1)))
        tris.append(((cx, cy, z1), (ax, ay, z1), (bx, by, z1)))  # front cap, +z
        tris.append(((cx, cy, z0), (bx, by, z0), (ax, ay, z0)))  # back cap, -z
    return tris


def realsense_tris() -> list[Tri]:
    half_x, half_y, half_z = BODY_X_MM / 2, BODY_Y_MM / 2, BODY_Z_MM / 2
    tris = _box_tris((-half_x, -half_y, -half_z), (half_x, half_y, half_z))
    for lens_x in LENS_CENTRES_X_MM:
        tris.extend(
            _cylinder_tris(lens_x, 0.0, half_z, half_z + LENS_PROTRUSION_MM, LENS_RADIUS_MM)
        )
    return tris


def _normal(tri: Tri) -> Vec:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / length, ny / length, nz / length)


def binary_stl(tris: list[Tri]) -> bytes:
    out = [b"viam-isaac-sim low-poly RealSense D435 body, millimetres".ljust(80, b"\0")]
    out.append(struct.pack("<I", len(tris)))
    for tri in tris:
        out.append(struct.pack("<3f", *_normal(tri)))
        for vertex in tri:
            out.append(struct.pack("<3f", *vertex))
        out.append(struct.pack("<H", 0))
    return b"".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl-out", help="also write the raw binary STL here")
    args = parser.parse_args()
    stl = binary_stl(realsense_tris())
    if args.stl_out:
        with open(args.stl_out, "wb") as handle:
            handle.write(stl)
    print(base64.b64encode(stl).decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
