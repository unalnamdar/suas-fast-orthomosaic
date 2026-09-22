# FastMosaic

GNSS-aided (SfM-free) fast orthomosaic generation.

## Install
pip install -e .

## Run
fastmosaic --help

This orthomosaicing model aims creating appropriate maps in a short period of time. ODM is a heavy model but I optimized it in order to satisfy time constraint in our SUAS 2026 competition. Whole process consists of three main steps:GPS-assisted image preparation, fast orthophoto generation, and output delivery.

For GPS integration, each JPEG captured by the A8 gimbal camera is tagged with its corresponding GPS coordinates using a Python script that reads location data directly from the image filename (which encodes lat/lon at capture time) and injects it into the EXIF metadata via the piexif library. This eliminates the need for manual georeferencing and allows ODM to correctly position every frame in real-world coordinates.

On the ODM side, we run the pipeline in --fast-orthophoto mode, which skips the computationally expensive stages — dense reconstruction (OpenMVS), 3D meshing, and texture mapping — and generates the orthophoto directly from the sparse point cloud produced by Structure from Motion. This reduces processing time from ~25-30 minutes to under 5 minutes on a standard laptop, while preserving orthophoto quality sufficient for competition mapping requirements.

Here are some examples of maps I've created with this cumtomized model:

A map of our drone test area:
<img width="1452" height="1432" alt="yavrucak original" src="https://github.com/user-attachments/assets/57c59772-8b22-47cd-af33-822ded9ad864" />

Another one with higher resolution picture.
<img width="2491" height="2458" alt="saha original" src="https://github.com/user-attachments/assets/c18f4889-cf05-4fc8-941a-03ff4f3a9dc7" />

An older map created created by using interned sourced pictures:
<img width="3845" height="4241" alt="test original" src="https://github.com/user-attachments/assets/1ee41ed9-b34d-4f19-a287-8935e09c2882" />
