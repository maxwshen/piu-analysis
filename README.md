# piu-analysis

Data-driven analysis of Pump it Up - drives www.piucenter.com.

The git repo for the front-end web app is https://github.com/maxwshen/piu-app.

### About
---
I view this project primarily as a platform for data-driven analysis of Pump it Up stepcharts that enables a truly broad spectrum of possibilities. The core contribution of this platform is an algorithm that annotates where your limbs (feet and hands) are positioned in physical space to perform a stepchart. For example, two single notes could be front-facing or a 180-degree twist depending on the context. In my view, solving this problem is like a tree trunk that enables a huge space of potential downstream projects and applications, including some implemented in this web app like chart clustering/recommendations and data-driven difficulty ratings, and other possibilities like improving the quality of machine-learning driven stepchart generation from music files.

How does the algorithm work? There is no single correct way to play pump, so I focused on recommending movements that would be useful for players: how to use heel-toe at lower levels, and identifying rolling hits that can be bracketed at higher levels. At its core, the program uses a Dijkstra's algorithm to find the minimum-cost path through a graph, prioritizing total minimizing physical distance moved. Each node in the graph is a 'stance-action' tuple comprising positions of each limb. I found it important to use sequence reasoning and segmentation to constrain paths in the graph, for example to guarantee alternating feet in certain situations. For several types of patterns, the "correct" way to move is ambiguous and typically depends on larger contexts: these include jacks vs footswitches, jumping vs single-foot bracketing two simultaneous notes, whether to alternate feet on hold-taps. Even seemingly straightforward "rules" like "always alternate feet on a series of single notes" have important subtleties: gallop jumps appear as single notes, but due to their rhythm they are more easily executed as jumps that have "double-steps".

I would estimate the algorithm as 80%-90% accurate. If you explore the web app enough, it is not hard to find strange and incorrect foot placements. Unfortunately, I suspect that the easiest way to improve the quality of stepchart information is largely manual. If you are interested in contributing towards this, the help would be greatly appreciated - the algorithm can use "foot hints" that simply label which foot should be used for each note. This means that data of foot-annotated charts produced by people like Junare could be used to improve this website's stepchart quality.

### Setup and installation
---
We recommend using virtual environments. We currently use python 3.9.6, though the codebase also works with python 3.7. The required packages are fairly lightweight.

```bash
python3.9 -m pip install -r requirements.txt
```

### Running the back-end
---
This is not very user friendly at the moment, unfortunately. At a high level, perform the following:
- Crawl your local StepF2/P1 folder for .ssc files, using _crawl.py, after editing the STEP_FOLD variable
- Run a_format_data, a2_subset_charts, b_graph, segment, c_dijkstra, d_annotate, merge_features, cluster_featurize, and e_struct.
  - For long or complex stepcharts, c_dijkstra can take 30+ minutes and >8GB RAM
  - Every script runs on single stepcharts, and thus is easily parallelized, except for merging the CSVs in merge_features.py and cluster_featurize.py

While it can be reasonable to run the pipeline on a single stepchart on a single computer, I use a cluster that supports up to 1,000 simultaneous jobs to run the pipeline on all 7k+ stepcharts.

### Accessing pre-processed data
---
http://piu-app.s3-website-us-east-1.amazonaws.com/

work in progress - please request in the discord (link in the footer of www.piucenter.com)


### How to fix stepchart foot placements
---
Making this into an easy pipeline is a work in progress. The procedure at the moment depends on non-public intermediate files. Individual foot hints are accepted - see `data/hints/` for examples.
