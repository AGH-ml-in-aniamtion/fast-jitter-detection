# Fast Detection of Jitter Artifacts in Human Motion Capture Models

This is the repository related to the publication "Fast Detection of Jitter Artifacts in Human Motion Capture Models" [from GRAPP 2025 conference](https://www.insticc.org/node/TechnicalProgram/VISIGRAPP/2025/presentationDetails/131444).


## Installation and running
Suggested to run in a separate Python virtual environment (tested on Python 3.10).

1. `pip install -r requirements.txt`
2. `cd src`
3. Set PYTHONPATH to `src` directory (or just add `src` to sources in PyCharm), e.g. `export PYTHONPATH=.`
4. Run scripts with `src` as working directory (e.g. `python publication_scripts/lafan1_obstacles_jitter.py`)

## Datasets
The repository contains small samples of datasets used in the experiments. 
The total size of datasets exceeds 2 GB, and thus it's not feasible to store that in the Git repository. 
These can be downloaded separately using [this Google Drive link](https://drive.google.com/file/d/1BrvLX6iQzCR3fjao-922ctUZVthWB-kT/view?usp=sharing) 
and replacing `src/data/datasets/bvh` directory.   

The following datasets were used for evaluation:
* LAFAN1: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
* Human3.6M: http://vision.imar.ro/human3.6m/description.php
* Bandai Namco Emotes: https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/
* Contemporary Dance (can also be downloaded using `scripts/contemporary_dance.sh`): https://dancedb.cs.ucy.ac.cy/main/performances 