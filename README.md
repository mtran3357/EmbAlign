EmbAlign is a fully automated 3D registration framework that determines lineage identities from single embryo snapshots of nuclei positions in C. elegans. Anchoring searches on observed cell counts, EmbAlign retrieves reference templates from a spatiotemporal atlas and refines assignments using an iterative Sinkhorn alignment procedure. This approach robustly handles positional variability and arbitrary orientations in both live and fixed uncompressed embryos.

config.py
- Default hyperparameters for different pipeline configurations.

atlas.py:
- Machinery for constructing and querying empirical spatiotemporal reference atlas (location, canonical time, and valid compositions)

models.py:
- Core data structures. Standardizes inputs via EmbryoFrame for observed data and AtlasFrame for atlas target data.

matcher.py:
- Implementation of hard (Hungarian) and soft (Sinkhorn) observed-atlas single cell correspondences.

transformer.py: 
- Executes rigid-body spatial transformations with the (weighted/unweighted) kabsch algorithm.

engine.py:
- Drives end-to-end alignment pipeline for a single embryo frame.
- Also manages initial coarse sweep to find the best starting orientation for downstream iterative refinement.

runner.py: 
- Manages pipeline execution. Contrains ValidationRunner for scoring against ground-truth data and Inference Runner for annotating unlabeled data. 

oracle.py:
- Machinery for training and utilizing a Random Forest classifier to assign a confidence score for each cell state prediciton. 

benchmarking.py:
- Implementaiton of Leave-One-Out Cross-Calidation framework for benchmarking pipeline performance in ground-truth annotated data. 

report_builder.py:
- Generates HTML interactive alignment report with 3D visualizations

plot_utils.py:
- Visualization swuite to gneerate static and dynamic plots usefuly for assessing alignment performance. 