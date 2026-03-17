#!/usr/bin/env python3
"""
Main entry point for the Cytoneighbor Experimental Framework.
Provides command-line interface for running experiments and managing the framework.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import ConfigManager, setup_logging, ensure_output_dir
from src.trainer import CytoneighborWrapper, TrainingManager


def setup_experiment_environment(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Setup the experimental environment with configuration and logging.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    
    # Setup logging
    log_level = config.get('global', {}).get('log_level', 'INFO')
    setup_logging(log_level)
    
    # Ensure output directory exists
    output_dir = config.get('global', {}).get('output_dir', 'results')
    ensure_output_dir(output_dir)
    
    return config


def run_benchmark_resolution(config: Dict[str, Any]) -> None:
    """Run the benchmark resolution experiment."""
    from experiments.benchmark_resolution import BenchmarkResolutionExperiment
    from src.utils import DataLoader, MetricsCalculator, VisualizationEngine
    
    # Initialize components
    data_loader = DataLoader()
    metrics_calc = MetricsCalculator()
    viz_engine = VisualizationEngine(config.get('visualization', {}))
    
    # Get experiment config
    exp_config = config['experiments']['benchmark_resolution']
    
    # Run experiment (includes setup, execution, visualization, and reporting)
    experiment = BenchmarkResolutionExperiment(exp_config, data_loader, metrics_calc, viz_engine)
    results = experiment.run_experiment()
    experiment.cleanup()
    
    print("Benchmark resolution experiment completed successfully!")
    print(f"Results saved to: {results['output_dir']}")


def run_mechanism_analysis(config: Dict[str, Any]) -> None:
    """Run the mechanism analysis experiment."""
    from experiments.mechanism_analysis import MechanismAnalysisExperiment
    from src.utils import DataLoader, MetricsCalculator, VisualizationEngine
    
    # Initialize components
    data_loader = DataLoader()
    metrics_calc = MetricsCalculator()
    viz_engine = VisualizationEngine(config.get('visualization', {}))
    
    # Get experiment config
    exp_config = config['experiments']['mechanism_analysis']
    
    # Run experiment
    experiment = MechanismAnalysisExperiment(exp_config, data_loader, metrics_calc, viz_engine)
    experiment.setup_experiment()
    results = experiment.run_experiment()
    experiment.generate_results(results)
    experiment.cleanup()
    
    print("Mechanism analysis experiment completed successfully!")


def run_disease_discovery(config: Dict[str, Any]) -> None:
    """Run the disease discovery experiment."""
    from experiments.disease_discovery import DiseaseDiscoveryExperiment
    
    # Setup logger
    logger = logging.getLogger(__name__)
    
    # Run experiment
    experiment = DiseaseDiscoveryExperiment(config)
    results = experiment.run_experiment()
    
    logger.info("Disease discovery experiment completed successfully")
    return results


def run_spatial_validation(config: Dict[str, Any]) -> None:
    """Run the spatial validation experiment."""
    from experiments.spatial_validation import SpatialValidationExperiment
    from src.utils import DataLoader, MetricsCalculator, VisualizationEngine
    
    # Initialize components
    data_loader = DataLoader()
    metrics_calc = MetricsCalculator()
    viz_engine = VisualizationEngine(config.get('visualization', {}))
    
    # Get experiment config
    exp_config = config['experiments']['spatial_validation']
    
    # Run experiment
    experiment = SpatialValidationExperiment(exp_config, data_loader, metrics_calc, viz_engine)
    experiment.setup_experiment()
    results = experiment.run_experiment()
    experiment.generate_results(results)
    experiment.cleanup()
    
    print("Spatial validation experiment completed successfully!")


def run_all_experiments(config: Dict[str, Any]) -> None:
    """Run all experiments in sequence."""
    experiments = [
        ("Benchmark Resolution", run_benchmark_resolution),
        ("Mechanism Analysis", run_mechanism_analysis),
        ("Disease Discovery", run_disease_discovery),
        ("Spatial Validation", run_spatial_validation)
    ]
    
    for exp_name, exp_func in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_name} Experiment")
        print(f"{'='*60}")
        
        try:
            exp_func(config)
        except Exception as e:
            print(f"Error in {exp_name} experiment: {e}")
            logging.error(f"Error in {exp_name} experiment: {e}", exc_info=True)
            continue
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")


def train_model(config: Dict[str, Any], dataset_path: str, cell_type_column: str) -> None:
    """Train a Cytoneighbor model."""
    import scanpy as sc
    
    # Load data
    print(f"Loading dataset: {dataset_path}")
    adata = sc.read_h5ad(dataset_path)
    
    # Initialize training manager
    model_config = config.get('model_params', {})
    trainer = TrainingManager(model_config)
    
    # Train model
    print("Training Cytoneighbor model...")
    model = trainer.train_model(adata, cell_type_column)
    
    print("Model training completed successfully!")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Cytoneighbor Experimental Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python main.py run-all
  
  # Run specific experiment
  python main.py run-experiment benchmark-resolution
  
  # Train a model
  python main.py train --dataset data/Human_immune.h5ad --cell-type cell_type
  
  # Use custom config
  python main.py run-all --config my_config.yaml
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run-all', 'run-experiment', 'train'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'experiment',
        nargs='?',
        choices=['benchmark-resolution', 'mechanism-analysis', 'disease-discovery', 'spatial-validation'],
        help='Specific experiment to run (for run-experiment command)'
    )
    
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Path to configuration file (default: configs/config.yaml)'
    )
    
    parser.add_argument(
        '--dataset',
        help='Path to dataset file (for train command)'
    )
    
    parser.add_argument(
        '--cell-type',
        default='cell_type',
        help='Cell type column name (default: cell_type)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Setup environment
    try:
        config = setup_experiment_environment(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override output directory if specified
    if args.output_dir:
        config.setdefault('global', {})['output_dir'] = args.output_dir
    
    # Execute command
    try:
        if args.command == 'run-all':
            run_all_experiments(config)
            
        elif args.command == 'run-experiment':
            if not args.experiment:
                print("Error: experiment name required for run-experiment command")
                sys.exit(1)
            
            experiment_map = {
                'benchmark-resolution': run_benchmark_resolution,
                'mechanism-analysis': run_mechanism_analysis,
                'disease-discovery': run_disease_discovery,
                'spatial-validation': run_spatial_validation
            }
            
            experiment_map[args.experiment](config)
            
        elif args.command == 'train':
            if not args.dataset:
                print("Error: --dataset required for train command")
                sys.exit(1)
            
            train_model(config, args.dataset, args.cell_type)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Main execution error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()