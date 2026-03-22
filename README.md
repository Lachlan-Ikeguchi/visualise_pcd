To install:
1. Clone repository
2. Have python3.11
3. Have pip for python3.11
4. Run `install_dependencies.bash`

To run:
1. Source `source.bash`
2. Run `visualiser.py` followed by the individual .pcd files.  Multiple files per run is supported.

Upside down point cloud:
- If a point cloud is upside down, it can be flipped by `point_cloud_flipper.py`
1. Source `source.bash`
2. Run `point_cloud_flipper.py` followed by input files, then `-o` to signify the following are the output locations, specify in order where it shall be saved
