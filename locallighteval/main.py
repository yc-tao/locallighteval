"""Main script for LocalLightEval - LLM evaluation benchmark."""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from rich.console import Console

from .config import validate_config
from .data_loader import load_dataset
from .inference import VLLMInferenceEngine, BinaryClassificationInference
from .metrics import BinaryClassificationMetrics
from .summarization import ClinicalSummarizationEngine
from .prompts import PromptManager
from .utils import (
    setup_rich_logging, print_banner, print_config_summary, 
    create_progress_bar, print_gpu_info, save_run_metadata,
    create_run_summary, check_gpu_availability, validate_environment
)

console = Console()


def run_summarization_pipeline(cfg: DictConfig, dataset, inference_engine: VLLMInferenceEngine, 
                              output_dir: Path, start_time: datetime, 
                              summarization_engine: VLLMInferenceEngine = None) -> Path:
    """Run the summarization pipeline.
    
    Args:
        cfg: Configuration
        dataset: Input dataset
        inference_engine: vLLM inference engine
        output_dir: Output directory
        start_time: Start time of the run
        
    Returns:
        Path to the generated summaries file
    """
    logger.info("Starting summarization pipeline...")
    
    # Use the provided summarization engine or the default inference engine
    engine_to_use = summarization_engine if summarization_engine is not None else inference_engine
    
    # Initialize prompt manager
    prompt_config_path = cfg.summarization.get('prompt_config_path', 'config/prompts.yaml')
    prompt_type = cfg.summarization.get('prompt_type', 'clinical_summary')

    try:
        # Try to load prompts from config file
        prompt_manager = PromptManager.from_config_file(Path(prompt_config_path))
        logger.info(f"Loaded prompts from {prompt_config_path}")
    except Exception as e:
        logger.warning(f"Failed to load prompt config from {prompt_config_path}: {e}")
        logger.info("Using default prompts")
        prompt_manager = PromptManager()

    # Initialize summarization engine
    debug_mode = cfg.get('debug', False)
    cleanup_discharge_text = cfg.summarization.get('cleanup_discharge_text', False)
    summarizer = ClinicalSummarizationEngine(
        engine_to_use,
        prompt_manager=prompt_manager,
        prompt_type=prompt_type,
        debug=debug_mode,
        cleanup_discharge_text=cleanup_discharge_text
    )
    
    # Convert dataset to list format for processing
    data_list = []
    for batch in dataset.get_batches(len(dataset)):  # Get all data in one batch
        data_list.extend(batch)

    # Get batch size from config, but force to 1 in debug mode
    batch_size = cfg.summarization.get('batch_size', 1)
    if debug_mode:
        batch_size = 1
        logger.info(f"Debug mode enabled: forcing batch size to 1")
    else:
        logger.info(f"Using batch size: {batch_size}")

    # Generate output filename for incremental saving
    input_path = Path(cfg.data.input_path)
    suffix = cfg.summarization.get('output_suffix', '_summaries')
    output_filename = f"{input_path.stem}{suffix}.jsonl"
    output_file = output_dir / output_filename

    # Process with progress tracking and incremental saving
    with create_progress_bar() as progress:
        task_id = progress.add_task("Generating summaries...", total=len(data_list))

        processed_data = summarizer.process_dataset(
            data_list,
            batch_size=batch_size,
            output_path=output_file,
            progress=progress,
            task_id=task_id
        )
    
    # Results are already saved incrementally, but ensure final save for completeness
    logger.info(f"Final results saved to: {output_file}")
    
    # Log completion
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    console.print(f"\n[bold green]Summarization completed![/bold green]")
    console.print(f"[bold]Duration:[/bold] {duration:.2f} seconds")
    console.print(f"[bold]Summaries generated:[/bold] {len(processed_data)}")
    console.print(f"[bold]Output file:[/bold] {output_file}")
    
    return output_file


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    start_time = datetime.now()
    
    try:
        # Print banner
        print_banner()
        
        # Validate configuration
        logger.info("Validating configuration...")
        validate_config(cfg)
        
        # Get Hydra's output directory from the runtime config
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info(f"Output directory: {output_dir}")
        
        # Setup logging
        log_level = "DEBUG" if cfg.debug else "INFO"
        disable_vllm_logging = cfg.get('disable_vllm_logging', False)
        setup_rich_logging(output_dir, log_level=log_level, disable_vllm_logging=disable_vllm_logging)
        
        # Print configuration summary
        print_config_summary(cfg, output_dir)
        
        # Print GPU information
        print_gpu_info()
        
        # Validate environment
        logger.info("Validating environment dependencies...")
        env_validation = validate_environment()
        missing_packages = [pkg for pkg, available in env_validation.items() if not available]
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.error("Please install missing packages and try again")
            sys.exit(1)
        
        logger.info("All dependencies validated successfully")
        
        # Check for dry run mode
        if cfg.get('dry_run', False):
            logger.info("Dry run mode - stopping after configuration validation")
            console.print("[bold green]✓ Configuration validation successful![/bold green]")
            console.print(f"[bold]Would use output directory:[/bold] {output_dir}")
            console.print("[bold]Dry run completed successfully[/bold]")
            return
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset(
            data_path=cfg.data.input_path,
            text_key=cfg.data.text_key,
            label_key=cfg.data.label_key,
            max_samples=cfg.data.max_samples
        )
        
        total_samples = len(dataset)
        logger.info(f"Dataset loaded: {total_samples} samples")
        
        # Initialize inference engine
        logger.info("Initializing vLLM inference engine...")
        inference_engine = VLLMInferenceEngine(
            model_config=cfg.model,
            inference_config=cfg.inference
        )
        
        # Handle different modes
        mode = cfg.get('mode', 'evaluation')
        logger.info(f"Running in {mode} mode")
        
        if mode == 'summarization':
            # Run summarization only
            run_summarization_pipeline(cfg, dataset, inference_engine, output_dir, start_time)
            return
        elif mode == 'end_to_end':
            # Check if we need to use different models for summarization and evaluation
            use_dual_models = cfg.get('dual_model', {}).get('use_different_models', False)
            
            if use_dual_models:
                logger.info("Setting up dual model configuration for end-to-end processing...")
                
                # Initialize separate summarization engine
                from omegaconf import OmegaConf
                sum_model_config = OmegaConf.create(cfg.dual_model.summarization_model)
                
                summarization_engine = VLLMInferenceEngine(
                    model_config=sum_model_config,
                    inference_config=cfg.inference
                )
                
                # Run summarization with the summarization model
                summaries_file = run_summarization_pipeline(
                    cfg, dataset, inference_engine, output_dir, start_time, 
                    summarization_engine=summarization_engine
                )
                
                # Clean up summarization model to free GPU memory
                logger.info("Unloading summarization model to free GPU memory...")
                summarization_engine.cleanup()
                del summarization_engine
                
                # Initialize evaluation engine with different model
                eval_model_config = OmegaConf.create(cfg.dual_model.evaluation_model)
                logger.info(f"Initializing evaluation model: {eval_model_config.name}")
                
                inference_engine = VLLMInferenceEngine(
                    model_config=eval_model_config,
                    inference_config=cfg.inference
                )
                
            else:
                # Use the same model for both summarization and evaluation
                summaries_file = run_summarization_pipeline(cfg, dataset, inference_engine, output_dir, start_time)
            
            # Load the summarized data for evaluation
            logger.info("Loading generated summaries for evaluation...")
            summary_dataset = load_dataset(
                data_path=str(summaries_file),
                text_key="text",
                label_key="label",
                max_samples=cfg.data.max_samples
            )
            dataset = summary_dataset
            total_samples = len(dataset)
            logger.info(f"Summary dataset loaded: {total_samples} samples")
        # If mode == 'evaluation', continue with normal evaluation flow
        
        # Initialize binary classification inference
        binary_classifier = BinaryClassificationInference(inference_engine)
        
        # Initialize metrics
        metrics_computer = BinaryClassificationMetrics()
        
        # Run evaluation with progress tracking
        logger.info("Starting evaluation...")
        
        with create_progress_bar() as progress:
            # Create progress tasks
            data_task = progress.add_task("Loading data...", total=total_samples)
            inference_task = progress.add_task("Running inference...", total=0)
            metrics_task = progress.add_task("Computing metrics...", total=3)
            
            # Process data in batches
            all_predictions = []
            all_true_labels = []
            all_texts = []
            
            batch_count = 0
            for batch in dataset.get_batches(cfg.inference.batch_size):
                batch_texts = [item["text"] for item in batch]
                batch_labels = [item["label"] for item in batch]
                
                # Update progress
                progress.update(data_task, advance=len(batch))
                
                if batch_count == 0:
                    # Initialize inference progress after first batch
                    total_batches = (total_samples + cfg.inference.batch_size - 1) // cfg.inference.batch_size
                    progress.update(inference_task, total=total_batches)
                
                # Run inference on batch
                batch_predictions = binary_classifier.classify_batch(
                    texts=batch_texts,
                    instruction=cfg.inference.instruction
                )
                
                all_predictions.extend(batch_predictions)
                all_true_labels.extend(batch_labels)
                all_texts.extend(batch_texts)
                
                batch_count += 1
                progress.update(inference_task, advance=1)
                
                logger.debug(f"Processed batch {batch_count}, {len(batch)} samples")
            
            progress.update(data_task, completed=total_samples)
            progress.update(inference_task, completed=progress.tasks[1].total)
            
            # Compute metrics
            logger.info("Computing evaluation metrics...")
            
            # Extract predictions and labels
            predicted_labels = [pred.get("predicted_label") for pred in all_predictions]
            
            # Compute main metrics
            progress.update(metrics_task, advance=1, description="Computing classification metrics...")
            metrics = metrics_computer.compute_metrics(
                y_true=all_true_labels,
                y_pred=predicted_labels
            )
            
            # Compute detailed results
            progress.update(metrics_task, advance=1, description="Computing detailed results...")
            detailed_results = metrics_computer.compute_detailed_results(
                predictions=all_predictions,
                true_labels=all_true_labels
            )
            
            # Error analysis
            progress.update(metrics_task, advance=1, description="Performing error analysis...")
            error_analysis = metrics_computer.get_error_analysis()
            
            progress.update(metrics_task, completed=3)
        
        # Print results
        logger.info("Evaluation completed successfully!")
        metrics_computer.print_summary()
        
        # Save results
        if cfg.output.save_metrics or cfg.output.save_predictions:
            logger.info("Saving results...")
            
            # Save metrics and detailed results
            metrics_computer.save_results(
                output_dir=output_dir,
                save_detailed=cfg.output.save_detailed_results
            )
            
            # Save predictions if requested
            if cfg.output.save_predictions:
                import json
                predictions_file = output_dir / "predictions.json"
                with open(predictions_file, 'w') as f:
                    json.dump(all_predictions, f, indent=2, default=str)
                logger.info(f"Predictions saved to {predictions_file}")
            
            # Save error analysis
            error_file = output_dir / "error_analysis.json"
            with open(error_file, 'w') as f:
                json.dump(error_analysis, f, indent=2, default=str)
            logger.info(f"Error analysis saved to {error_file}")
        
        # Save run metadata and summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        save_run_metadata(
            output_dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            start_time=start_time,
            end_time=end_time,
            additional_info={
                "gpu_info": check_gpu_availability(),
                "environment_validation": env_validation,
                "total_samples": total_samples,
                "final_metrics": metrics
            }
        )
        
        create_run_summary(
            output_dir=output_dir,
            metrics=metrics,
            config=OmegaConf.to_container(cfg, resolve=True),
            sample_count=total_samples,
            duration=duration
        )
        
        # Final summary
        console.print("\n" + "="*80)
        console.print(f"[bold green]Evaluation completed successfully![/bold green]")
        console.print(f"[bold]Duration:[/bold] {duration:.2f} seconds")
        console.print(f"[bold]Samples processed:[/bold] {total_samples}")
        console.print(f"[bold]Results saved to:[/bold] {output_dir}")
        
        if metrics.get('valid_predictions', 0) > 0:
            console.print(f"[bold]Final Accuracy:[/bold] {metrics['accuracy']:.4f}")
            console.print(f"[bold]Final F1-Score:[/bold] {metrics['f1_score']:.4f}")
        
        console.print("="*80)
        
        logger.info(f"Run completed. Results available at: {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        console.print(f"[bold red]Evaluation failed:[/bold red] {e}")
        
        # Save error information if output directory exists
        try:
            if 'output_dir' in locals():
                error_file = output_dir / "error.log"
                with open(error_file, 'w') as f:
                    f.write(f"Error: {e}\n")
                    f.write(f"Time: {datetime.now()}\n")
                    import traceback
                    f.write(f"Traceback:\n{traceback.format_exc()}")
        except:
            pass
        
        raise e


def run_with_config_override(**kwargs) -> None:
    """Run evaluation with configuration overrides."""
    # Create a temporary config file or use command-line overrides
    overrides = []
    for key, value in kwargs.items():
        overrides.append(f"{key}={value}")
    
    # This would be used programmatically
    import hydra
    from hydra import compose, initialize
    
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides)
        main(cfg)


if __name__ == "__main__":
    main()