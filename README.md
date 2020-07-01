# lastomesh

Point cloud data mesh conversion tool.

This project is currently focused on creating mesh for [Shizuoka Point Cloud Database](https://pointcloud.pref.shizuoka.jp/).

## Requirements

 - Python>=3.6,!=3.8 (because of using [Open3D](http://open3d.org/))

## Setup

```bash
pip install -r requirements.txt
```

## Usage 

### Generating Meshes

```bash
for i in 28XXX00030007 29C2001011347 29XXX00010002 30XXX03010001 29C2001011323 29W9350011101 30D7318011101 29C2001011346 29XXX00010001 30XXX00010034; do
  python3 lastomesh.py --local-scheduler CreateMeshFromLasData --product-id $i;
done
```

### Rendering Meshes

```bash
for i in 28XXX00030007 29C2001011347 29XXX00010002 30XXX03010001 29C2001011323 29W9350011101 30D7318011101 29C2001011346 29XXX00010001 30XXX00010034; do
  python3 lastomesh.py --local-scheduler RenderProduct --product-id $i;
done
```

