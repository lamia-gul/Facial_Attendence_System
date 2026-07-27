DATASET NOTE
------------
This folder contains the registered face photos captured via register_faces.py,
organized in the folder structure expected by encode_faces.py:

    known_faces/
        <PersonName>/
            img_1.jpg
            img_2.jpg
            ...

IMPORTANT: The folder names "Member1_Serena" and "Member2_LamiaGul" are based on
names observed during development/testing. Please verify these match the actual
names used when register_faces.py was originally run, and rename the folders if
needed so they match the "name" typed into register_faces.py exactly
