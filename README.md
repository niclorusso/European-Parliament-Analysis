# European Parliament Voting Behavior Project

This repository collects the data preparation steps, exploratory notebooks, and production scripts used to study roll-call voting (RCV) behaviour in the European Parliament (EP) across legislatures EP6–EP10. The project focuses on:

- Building similarity networks between Members of the European Parliament (MEPs) from RCV records.
- Measuring party cohesion, agreement, frustration, entropy, and other coordination metrics.
- Detecting mesoscale structure (e.g., partitions by party, ideology, country) via modularity and community detection.
- Exporting ready-to-visualise network files (`.gexf`), figures, and CSV summaries for downstream analysis.

## Repository Layout

- `data/`: Cleaned voting tables (`all_votes_EP*.csv`, `all_votes_main_EP*.csv`) plus the raw VoteWatch exports under `data/votewatch_csv/`. Also contains intermediate preprocessing snapshots.
- `figs/`, `results/`, `similarity_matrices/`: Generated artefacts from the notebooks and scripts (plots, modularity summaries, similarity matrices).
- `topic_*`, `policy_*`, `mep_topic_*`, `gexf_*`: Network exports grouped by policy area or topic for visual inspection in Gephi/Cytoscape.
- `*.ipynb`: Jupyter notebooks covering agreement/frustration metrics, entropy measures, PCA, party cohesion, temporal coalitions, etc. Each notebook is self-contained and documents a distinct analysis track.
- `modularity.py`: Batch pipeline that computes Louvain modularity values per policy area for EP6–EP10.

## Getting Started

1. **Create an environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install the scientific stack**
   ```bash
   pip install pandas numpy scipy matplotlib seaborn networkx python-louvain tqdm jupyter
   ```
   The notebooks also expect common libraries such as `plotly`, `scikit-learn`, and `community` (python-louvain). Install any additional packages flagged by Jupyter as needed.
3. **Launch Jupyter Lab/Notebook**
   ```bash
   jupyter lab
   ```

All datasets are already versioned in `data/`. They originate from VoteWatch or EP public sources; consult their terms if you plan to redistribute.

## Reproducing Key Analyses

- **Agreement & frustration**: `agreement_and_frustration.ipynb`, `agreement_per_party.ipynb`, `entropy.ipynb`, and related notebooks compute pairwise agreement matrices, entropy-based measures, and plot distributions per legislature.
- **Party cohesion & temporal dynamics**: `party_cohesion_EP*.ipynb`, `temporal_coalitions.ipynb`, and `network_frustration.ipynb` focus on how party discipline evolves across time windows.
- **Topic & policy networks**: `resin.ipynb`, `resin_per_topic.ipynb`, `topics_mep_networks_EP*/`, and `policy_networks_EP*/` generate per-topic networks plus `.gexf` exports.
- **Tutorials**: `tutorial_resin_current.ipynb` and `tutorial_resin_previous.ipynb` document the workflow for newcomers.

Each notebook starts with cells that load the necessary CSV files from `data/`. Run them sequentially after activating the environment above.

## Running the Modularity Pipeline

`modularity.py` computes, for every policy area within a legislature:

1. Vote similarity matrices (`similarity_matrix`) constructed from FOR/AGAINST/ABSTENTION patterns.
2. Network projections over MEPs (`graph_from_similarity`).
3. Louvain community detection repeated `R` times to estimate `Qmax`.
4. Benchmark modularity values for party, country, left/right, centrist/extreme, and random partitions.

Usage:
```bash
python modularity.py
```
The script expects the CSVs produced during preprocessing to be present under `data/`. Results are stored in `results/modularity_results_EP{N}.csv`, and progress logs will indicate missing datasets or policy areas skipped due to insufficient votes/MEPs.

## Data Notes

- Large CSVs and `.gexf` exports are stored in the repo for reproducibility; keep them out of memory-constrained environments.
- Some datasets originate from VoteWatch's licensed distributions (`data/votewatch_csv/`); please respect the original attribution and licensing constraints before sharing publicly.
- When extending the analyses, prefer writing new notebooks rather than editing the originals, so historical runs remain reproducible.

## Contributing / Extending

1. Fork or branch off main.
2. Add new preprocessing scripts or notebooks under descriptive filenames.
3. Document assumptions, data filters, and novel metrics inside the notebook/script markdown cells.
4. Open a pull request summarising the motivation, datasets touched, and new outputs added.

Feel free to adapt this pipeline to other legislatures or regional parliaments by swapping the source CSVs and updating `SCHEMA` inside `modularity.py`.

