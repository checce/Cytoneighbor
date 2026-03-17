# Cytoneighbor Experimental Framework

A comprehensive validation system for demonstrating the superiority of "relational identity definition" over traditional "coordinate embedding" approaches in single-cell analysis.

## Overview

The Cytoneighbor Experimental Framework validates three core innovations:

1. **SymRelEnc**: Pairwise interaction neural network that learns cell similarity probabilities
2. **Gene-CutMix**: Biologically-inspired data synthesis strategy that repairs manifold breaks  
3. **Recall & Rerank**: Two-stage strategy for high-resolution neighbor identification

## Project Structure

```
Cytoneighbor_Project/
├── data/                    # Dataset storage (.h5ad files)
│   ├── Human_immune.h5ad
│   ├── Pancreas.h5ad
│   ├── Diabetic_Kidney_Disease.h5ad
│   └── Visium/
├── src/                     # Core framework code
│   ├── model.py            # Cytoneighbor model implementation
│   ├── utils.py            # Data loading, metrics, visualization
│   └── trainer.py          # Training utilities and model wrapper
├── experiments/             # Experiment implementations
│   ├── benchmark_resolution.py    # Experiment 1: Resolution comparison
│   ├── mechanism_analysis.py      # Experiment 2: Gene-CutMix validation
│   ├── disease_discovery.py       # Experiment 3: Identity erosion detection
│   └── spatial_validation.py      # Experiment 4: Spatial consistency
├── configs/
│   └── config.yaml         # Configuration parameters
├── main.py                 # Main entry point
├── test_framework.py       # Framework validation tests
└── requirements.txt        # Python dependencies
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python test_framework.py
   ```

## Quick Start

### Running Individual Experiments

```bash
# Run benchmark resolution experiment
python main.py run-experiment benchmark-resolution

# Run mechanism analysis experiment  
python main.py run-experiment mechanism-analysis

# Run disease discovery experiment
python main.py run-experiment disease-discovery

# Run spatial validation experiment
python main.py run-experiment spatial-validation
```

### Running All Experiments

```bash
python main.py run-all
```

### Training a Model

```bash
python main.py train --dataset data/Human_immune.h5ad --cell-type cell_type
```

## Experiments

### 1. Benchmark Resolution (Fine-grained Subtyping)

**Objective**: Demonstrate Cytoneighbor's superior resolution on extremely similar CD8+ T cell subtypes.

**Key Features**:
- Extracts CD8+ T cell subgroups (Naive, Effector Memory, Exhausted)
- Compares against scVI baseline using latent embeddings
- Calculates iLISI (integration) and cLISI (purity) scores
- Generates UMAP visualizations for boundary assessment

**Expected Output**: Quantitative evidence of superior resolution performance

### 2. Mechanism Analysis (CutMix Restoration)

**Objective**: Prove Gene-CutMix repairs manifold breaks and recovers rare cells.

**Key Features**:
- Simulates search failure by downsampling rare epsilon cells to 0.5%
- Ablation study: Case A (no synthesis) vs Case B (with synthesis)
- Calculates Recall@k for rare cell recovery
- Demonstrates biological intermediate state simulation

**Expected Output**: Evidence that Gene-CutMix bridges manifold discontinuities

### 3. Disease Discovery (Identity Erosion)

**Objective**: Reveal identity erosion in Diabetic Kidney Disease using reference-query paradigm.

**Key Features**:
- Trains model exclusively on healthy reference data
- Uses frozen model for disease cell annotation
- Quantifies confidence score degradation in disease cells
- Performs differential expression analysis on eroded cells

**Expected Output**: Detection of disease-specific cellular identity changes

### 4. Spatial Validation (Spatial Mapping)

**Objective**: Verify relational identity consistency in physical tissue space.

**Key Features**:
- Processes Visium spatial transcriptomics data
- Treats each spot as mixed cell population query
- Leverages Gene-CutMix for mixed signal handling
- Validates spatial coherence of predictions

**Expected Output**: Spatial cell type maps with biological consistency

## Configuration

Edit `configs/config.yaml` to customize:

- **Model parameters**: Architecture, training settings, synthesis options
- **Experiment settings**: Dataset paths, cell type columns, metrics
- **Visualization options**: Figure sizes, color schemes, output formats

## Key Features

### Relational Identity Definition
- Cells defined by dynamic pairwise relationships rather than static coordinates
- Captures biological similarity through learned interaction patterns
- Robust to batch effects and technical noise

### Gene-CutMix Synthesis
- Biologically-inspired data augmentation strategy
- Creates realistic intermediate cell states
- Repairs manifold breaks caused by data sparsity
- Maintains gene expression ordering and biological plausibility

### Recall & Rerank Strategy
- Two-stage neighbor identification for extreme precision
- Initial recall phase using approximate nearest neighbors
- Reranking phase using neural similarity scores
- Achieves high resolution on fine-grained cell subtypes

## Output Files

Each experiment generates:
- **Quantitative metrics**: JSON files with performance scores
- **Visualizations**: High-resolution figures for publication
- **Data exports**: CSV/Excel files with predictions and coordinates
- **Summary reports**: Text summaries of key findings

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install all requirements from `requirements.txt`
2. **Dataset not found**: Ensure .h5ad files are in the `data/` directory
3. **Memory errors**: Reduce batch sizes in configuration
4. **CUDA errors**: Set `device: "cpu"` in config for CPU-only execution

### Performance Tips

- Use GPU acceleration when available (`device: "cuda"`)
- Enable mixed precision training (`use_amp: true`)
- Adjust batch sizes based on available memory
- Use early stopping to prevent overfitting

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{cytoneighbor2024,
  title={Cytoneighbor: Relational Identity Definition for Single-Cell Analysis},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For questions and support:
- Check the troubleshooting section above
- Review experiment logs in the results directories
- Open an issue on the project repository           