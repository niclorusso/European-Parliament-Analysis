# Modularity Analysis Package

**Standalone package** for computing and visualizing modularity metrics for European Parliament voting networks.

This package is **completely independent** - it computes everything from scratch using only raw vote data from the `data/` folder. No dependencies on external analysis scripts.

## Structure

```
modularity_analysis/
├── analyze.py              # Main analysis script (standalone)
├── utils.py                # Utility functions (similarity matrix, graph building, data loading)
├── modularity_functions.py # Modularity computation functions
├── plot_results.ipynb      # Jupyter notebook for plotting
├── results/                # Output directory (CSVs and plots)
│   ├── global_modularity.csv
│   ├── topic_modularity.csv
│   └── *.png (plots)
└── README.md               # This file
```

## Dependencies

- Python 3.8+
- pandas
- numpy
- networkx
- python-louvain (community)
- matplotlib (for plotting notebook)
- seaborn (for plotting notebook)

## Usage

### 1. Global Network Analysis

Analyze modularity for global networks (all votes combined):

```bash
# All EPs (6-10)
python modularity_analysis/analyze.py --global

# Specific EPs
python modularity_analysis/analyze.py --global --ep 9 10
```

Results are saved to: `modularity_analysis/results/global_modularity.csv`

### 2. Per-Topic Network Analysis

Analyze modularity for topic-specific networks:

```bash
# All topics, all EPs
python modularity_analysis/analyze.py --topics

# Specific EPs
python modularity_analysis/analyze.py --topics --ep 9 10

# Specific topics (case-insensitive, partial match)
python modularity_analysis/analyze.py --topics --topic "environment" "budget"

# Specific EPs and topics
python modularity_analysis/analyze.py --topics --ep 9 10 --topic "environment"
```

Results are saved to: `modularity_analysis/results/topic_modularity.csv`

### 3. Plotting Results

Open the Jupyter notebook:

```bash
jupyter notebook modularity_analysis/plot_results.ipynb
```

Or use JupyterLab:

```bash
jupyter lab modularity_analysis/plot_results.ipynb
```

The notebook will:
- Load CSV files from `results/`
- Generate plots for global modularity evolution
- Generate plots for per-topic modularity evolution
- Save plots to `results/` directory

## Output Files

### CSV Files

- **`results/global_modularity.csv`**: Contains modularity metrics for global networks
  - Columns: `ep`, `qmax`, `qparty`, `qcountry`, `q_left_right`, `q_left_center_right`, `q_extreme_centrist`, `q_majority_opposition`, `votes_used`, `downsampled`, `bootstrapped`, etc.

- **`results/topic_modularity.csv`**: Contains modularity metrics for per-topic networks
  - Columns: `ep`, `topic`, `qmax`, `qparty`, `qcountry`, etc.

### Plot Files

- **`results/global_modularity_evolution.png`**: Evolution of global network modularity across EPs
- **`results/topic_<topic_name>_evolution.png`**: Evolution of modularity for each topic

## Modularity Metrics

The analysis computes the following modularity measures:

1. **Qmax**: Maximum modularity (using Louvain algorithm, 20 runs per subsample)
2. **Qparty**: Modularity of party-based partition
3. **Qcountry**: Modularity of country-based partition
4. **Q_left_right**: Modularity of left-right ideological partition
5. **Q_left_center_right**: Modularity of left-center-right partition
6. **Q_extreme_centrist**: Modularity of extreme-centrist partition
7. **Q_majority_opposition**: Modularity of majority-opposition coalition partition

## Technical Details

- **Standalone**: All computation starts from raw vote data in `data/` folder
- **Similarity Matrix**: Computed from scratch using agreement/disagreement scoring
- **Subsampling**: Networks with >200 nodes are subsampled to 200 nodes (10 iterations)
- **Qmax computation**: For each subsample, Louvain is run 20 times and the maximum Qmax is taken
- **Bootstrapping**: Networks with ≤30 nodes use bootstrapping (100 iterations)
- **Confidence intervals**: Standard deviation across subsamples is reported

## Data Requirements

The package expects the following data files in the project root `data/` folder:

- `data/all_votes_main_EP{6-10}.csv` - Vote matrices for each EP
- `data/votewatch_csv/EP{6-10}_Voted main docs.csv` - Topic metadata for each EP

All computation is done from these raw files - no pre-computed networks or intermediate files are required.

## Examples

```bash
# Quick global analysis for recent EPs
python modularity_analysis/analyze.py --global --ep 9 10

# Analyze specific policy areas
python modularity_analysis/analyze.py --topics --topic "environment" "climate" "budget"

# Full analysis (all EPs, all topics)
python modularity_analysis/analyze.py --topics
```

## Notes

- The analysis uses existing MEP network GEXF files when available (from `results/resin_mep/topics/`)
- If GEXF files are not found, networks are built from raw vote data
- All results are stored in CSV format for easy analysis and plotting
- The plotting notebook can be customized for specific visualizations

