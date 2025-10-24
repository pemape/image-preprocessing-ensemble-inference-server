import sys
import os
from pathlib import Path
import cv2
import yaml
import time
import json
import numpy as np
from datetime import datetime

# Add parent directory to path to import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from fundus_preprocessor import FundusPreprocessor

def test_fixed_preprocessing():
    """Test the fixed FundusPreprocessor with debug functionality"""
    
    # Test image path
    test_image_path = r"D:\FEI STU\ing\2roc\DATABASE\Aptos\train_images\00a8624548a9.png"
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    # Create debug directory first
    debug_dir = Path("./debug")
    debug_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = ["clipped", "rgb_clahe", "min_pooling", "lab_clahe", "max_green_gsc", "variants", "reports"]
    for subdir in subdirs:
        (debug_dir / subdir).mkdir(exist_ok=True)
    
    # Use the existing config but enable debug mode
    config_path = r"c:\ProgrammingProjects\python\dp\image-preprocessing-ensemble-inference-server\configs\preprocessing_config.yaml"
    
    # Load and modify config to enable debug
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable debug features
    if 'debug' not in config:
        config['debug'] = {}
    
    config['debug']['verbose_logging'] = True
    config['debug']['save_intermediate_images'] = True
    config['debug']['log_processing_time'] = True
    
    # Set target resolution to 500x500 as in your original config
    config['general']['target_resolution'] = [500, 500]
    
    # Fix the intermediate paths to be directories (not files) and match the actual code
    # The actual code uses 'min_pooling' not 'ben_graham'
    config['debug']['intermediate_paths'] = {
        'black_border_clipped': './debug/clipped/',
        'rgb_clahe': './debug/rgb_clahe/',
        'min_pooling': './debug/min_pooling/',  # This matches the actual code
        'lab_clahe': './debug/lab_clahe/',
        'max_green_gsc': './debug/max_green_gsc/'
    }
    
    # Also update the image_variants to match what the code expects
    # The code uses 'min_pooling' but your config has 'ben_graham'
    if 'ben_graham' in config['image_variants']:
        # Copy ben_graham config to min_pooling
        config['image_variants']['min_pooling'] = config['image_variants']['ben_graham'].copy()
        config['image_variants']['min_pooling']['enabled'] = True
        config['image_variants']['min_pooling']['description'] = "Min-pooling based enhancement (Ben Graham method)"
    
    # Ensure processed_images directory exists
    processed_dir = Path(config['general']['output_directory'])
    processed_dir.mkdir(exist_ok=True)
    
    # Save modified config
    temp_config_path = "./debug/temp_debug_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TESTING FIXED FUNDUS PREPROCESSOR")
    print(f"{'='*60}")
    print(f"Image: {Path(test_image_path).name}")
    print(f"Debug config: {temp_config_path}")
    
    try:
        # Initialize preprocessor
        print(f"\n🔧 Initializing preprocessor...")
        preprocessor = FundusPreprocessor(temp_config_path)
        print(f"  ✓ Preprocessor initialized successfully")
        
        # Load image
        print(f"\n📊 Loading image...")
        original_image = cv2.imread(test_image_path)
        if original_image is None:
            print(f"  ❌ Failed to load image")
            return False
        
        height, width, channels = original_image.shape
        file_size = Path(test_image_path).stat().st_size / 1024 / 1024
        print(f"  ✓ Image loaded: {width}×{height}×{channels}, {file_size:.2f} MB")
        
        # Analyze original image properties
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_mean = original_rgb.mean(axis=(0, 1))
        original_std = original_rgb.std(axis=(0, 1))
        
        # Process image
        print(f"\n⚙️  Processing image...")
        start_time = time.time()
        
        processed_variants = preprocessor.process_image(original_image, "debug_test_image")
        
        processing_time = time.time() - start_time
        print(f"  ✓ Processing completed in {processing_time:.4f} seconds")
        
        # Check results
        print(f"\n📋 Processing results:")
        print(f"  • Total variants: {len(processed_variants)}")
        print(f"  • Variant names: {list(processed_variants.keys())}")
        
        # Collect detailed variant analysis for report
        variant_analysis = {}
        
        # Analyze each variant
        for variant_name, variant_image in processed_variants.items():
            print(f"\n  🖼️  {variant_name.upper()}:")
            print(f"    • Shape: {variant_image.shape}")
            print(f"    • Data type: {variant_image.dtype}")
            print(f"    • Value range: [{variant_image.min():.4f}, {variant_image.max():.4f}]")
            print(f"    • Mean: {variant_image.mean():.4f}")
            print(f"    • Std: {variant_image.std():.4f}")
            
            # Store analysis for report
            variant_analysis[variant_name] = {
                'shape': list(variant_image.shape),
                'dtype': str(variant_image.dtype),
                'min_value': float(variant_image.min()),
                'max_value': float(variant_image.max()),
                'mean_value': float(variant_image.mean()),
                'std_value': float(variant_image.std()),
                'pixel_count': int(variant_image.size),
                'unique_values': int(len(np.unique(variant_image)))
            }
            
            # Calculate contrast and brightness metrics
            if len(variant_image.shape) == 3:
                # Color image metrics
                variant_analysis[variant_name]['channel_means'] = [float(x) for x in variant_image.mean(axis=(0, 1))]
                variant_analysis[variant_name]['channel_stds'] = [float(x) for x in variant_image.std(axis=(0, 1))]
            
            # Calculate dynamic range
            variant_analysis[variant_name]['dynamic_range'] = float(variant_image.max() - variant_image.min())
        
        # Check debug folders and collect file info
        print(f"\n📁 Debug files check:")
        debug_paths = config['debug']['intermediate_paths']
        
        debug_files_info = {}
        total_debug_files = 0
        
        for debug_name, debug_path in debug_paths.items():
            debug_path_obj = Path(debug_path)  # Use different variable name
            if debug_path_obj.exists():
                files = list(debug_path_obj.glob("*debug_test_image*"))
                if files:
                    print(f"  ✓ {debug_name}: {len(files)} files")
                    debug_files_info[debug_name] = []
                    for file in files:
                        print(f"    - {file.name}")
                        file_info = {
                            'filename': file.name,
                            'size_bytes': file.stat().st_size,
                            'created_time': datetime.fromtimestamp(file.stat().st_ctime).isoformat(),
                            'path': str(file)
                        }
                        debug_files_info[debug_name].append(file_info)
                    total_debug_files += len(files)
                else:
                    print(f"  ⚠️  {debug_name}: No debug files found")
                    debug_files_info[debug_name] = []
            else:
                print(f"  ❌ {debug_name}: Directory doesn't exist")
                debug_files_info[debug_name] = None
        
        # Check processed_images folder
        processed_files_info = []
        processed_dir = Path(config['general']['output_directory'])
        if processed_dir.exists():
            processed_files = list(processed_dir.glob("*debug_test_image*"))
            if processed_files:
                print(f"  ✓ processed_images: {len(processed_files)} final variants")
                for file in processed_files:
                    print(f"    - {file.name}")
                    file_info = {
                        'filename': file.name,
                        'size_bytes': file.stat().st_size,
                        'created_time': datetime.fromtimestamp(file.stat().st_ctime).isoformat(),
                        'path': str(file)
                    }
                    processed_files_info.append(file_info)
                total_debug_files += len(processed_files)
            else:
                print(f"  ⚠️  processed_images: No final variants found")
        
        # Create comprehensive processing report
        processing_report = {
            'metadata': {
                'report_generated': datetime.now().isoformat(),
                'test_name': 'Fixed Fundus Preprocessor Debug Test',
                'version': '1.0',
                'preprocessor_config': str(temp_config_path)
            },
            'input_image': {
                'path': str(test_image_path),
                'filename': Path(test_image_path).name,
                'original_dimensions': [width, height, channels],
                'file_size_mb': round(file_size, 2),
                'data_type': str(original_image.dtype),
                'original_mean_rgb': [float(x) for x in original_mean],
                'original_std_rgb': [float(x) for x in original_std],
                'pixel_count': int(original_image.size)
            },
            'processing_performance': {
                'total_processing_time_seconds': round(processing_time, 4),
                'variants_generated': len(processed_variants),
                'processing_speed_mpx_per_second': round((width * height / 1_000_000) / processing_time, 2)
            },
            'preprocessing_config': {
                'target_resolution': config['general']['target_resolution'],
                'normalize_pixels': config['general'].get('normalize_pixels', True),
                'debug_enabled': config['debug']['save_intermediate_images'],
                'enabled_variants': [name for name, cfg in config.get('image_variants', {}).items() 
                                   if cfg.get('enabled', False)]
            },
            'variant_analysis': variant_analysis,
            'debug_files': {
                'intermediate_files': debug_files_info,
                'final_processed_files': processed_files_info,
                'total_debug_files_created': total_debug_files
            },
            'quality_metrics': {
                'processing_success': len(processed_variants) > 0,
                'all_variants_valid': all(
                    analysis['min_value'] >= 0 and analysis['max_value'] <= 1.1  # Allow slight overflow
                    for analysis in variant_analysis.values()
                ),
                'debug_files_created': total_debug_files > 0,
                'target_resolution_achieved': all(
                    analysis['shape'][:2] == config['general']['target_resolution']
                    for analysis in variant_analysis.values()
                )
            },
            'preprocessing_steps': {
                'border_clipping': {
                    'applied': True,
                    'method': 'automatic_selection',
                    'description': 'Black border detection and removal'
                },
                'variant_generation': {
                    'applied': True,
                    'variants_created': list(processed_variants.keys()),
                    'description': 'Multiple preprocessing variants for ensemble'
                },
                'resolution_processing': {
                    'applied': True,
                    'target_size': config['general']['target_resolution'],
                    'description': 'Resize to target resolution'
                },
                'normalization': {
                    'applied': config['general'].get('normalize_pixels', True),
                    'range': '[0, 1]' if config['general'].get('normalize_pixels', True) else '[0, 255]',
                    'description': 'Pixel value normalization'
                }
            }
        }
        
        # Save comprehensive report to reports folder
        report_filename = f"preprocessing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = debug_dir / "reports" / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(processing_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 SUMMARY:")
        print(f"  • Processing time: {processing_time:.4f} seconds")
        print(f"  • Variants generated: {len(processed_variants)}")
        print(f"  • Debug files created: {total_debug_files}")
        print(f"  • Processing report saved: {report_path}")
        
        # Create a simple summary report as well
        summary_report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'image_processed': Path(test_image_path).name,
                'processing_time_seconds': round(processing_time, 4),
                'variants_created': len(processed_variants),
                'debug_files_created': total_debug_files,
                'success': total_debug_files > 0 and len(processed_variants) > 0
            },
            'variant_names': list(processed_variants.keys()),
            'debug_folders_populated': [name for name, files in debug_files_info.items() 
                                      if files and len(files) > 0]
        }
        
        summary_path = debug_dir / "reports" / "test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print(f"  • Test summary saved: {summary_path}")
        
        if total_debug_files > 0:
            print(f"\n✅ SUCCESS! Debug files and reports are being saved properly!")
            print(f"📁 Check the following folders:")
            print(f"  • Debug intermediate files: ./debug/[variant_name]/")
            print(f"  • Final processed images: {config['general']['output_directory']}")
            print(f"  • Processing reports: ./debug/reports/")
        else:
            print(f"\n⚠️  WARNING: No debug files were created!")
        
        # Cleanup
        if Path(temp_config_path).exists():
            os.remove(temp_config_path)
        
        return total_debug_files > 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        if Path(temp_config_path).exists():
            os.remove(temp_config_path)
        raise

if __name__ == "__main__":
    success = test_fixed_preprocessing()
    if (success):
        print(f"\n🎉 Test completed successfully!")
    else:
        print(f"\n💥 Test failed!")