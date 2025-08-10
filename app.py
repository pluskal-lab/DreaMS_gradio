"""
DreaMS Gradio Web Application

This module provides a web interface for the DreaMS (Deep Representations Empowering 
the Annotation of Mass Spectra) tool using Gradio. It allows users to upload MS/MS 
files and perform library matching with DreaMS embeddings.

Author: DreaMS Team
License: MIT
"""

import gradio as gr
import spaces
import urllib.request
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import base64
from io import BytesIO
from PIL import Image
import io
import dreams.utils.spectra as su
import dreams.utils.io as dio
from dreams.utils.data import MSData
from dreams.api import dreams_embeddings
from dreams.definitions import *

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Default image sizes for different components
SMILES_IMG_SIZE = 200
SPECTRUM_IMG_SIZE = 1500

# Library and data paths
LIBRARY_PATH = Path('DreaMS/data/MassSpecGym_DreaMS.hdf5')
DATA_PATH = Path('./DreaMS/data')
EXAMPLE_PATH = Path('./data')

# Similarity threshold for filtering results
SIMILARITY_THRESHOLD = 0.75

# =============================================================================
# UTILITY FUNCTIONS FOR IMAGE CONVERSION
# =============================================================================

def _validate_input_file(file_path):
    """
    Validate that the input file exists and has a supported format
    
    Args:
        file_path: Path to the input file
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file_path or not Path(file_path).exists():
        return False
    
    supported_extensions = ['.mgf', '.mzML', '.mzml']
    file_ext = Path(file_path).suffix.lower()
    
    return file_ext in supported_extensions


def _convert_pil_to_base64(img, format='PNG'):
    """
    Convert a PIL Image to base64 encoded string
    
    Args:
        img: PIL Image object
        format: Image format (default: 'PNG')
    
    Returns:
        str: Base64 encoded image string
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/{format.lower()};base64,{repr(img_str)[2:-1]}"


def _crop_transparent_edges(img):
    """
    Crop transparent edges from a PIL Image
    
    Args:
        img: PIL Image object (should be RGBA)
    
    Returns:
        PIL Image: Cropped image
    """
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get the bounding box of non-transparent pixels
    bbox = img.getbbox()
    if bbox:
        # Crop the image to remove transparent space
        img = img.crop(bbox)
    
    return img


def smiles_to_html_img(smiles, img_size=SMILES_IMG_SIZE):
    """
    Convert SMILES string to HTML image for display in Gradio dataframe
    
    Args:
        smiles: SMILES string representation of molecule
        img_size: Size of the output image (default: SMILES_IMG_SIZE)
    
    Returns:
        str: HTML img tag with base64 encoded image
    """
    try:
        # Parse SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"<div style='text-align: center; color: red;'>Invalid SMILES</div>"
        
        # Create PNG drawing with Cairo backend for better control
        d2d = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
        opts = d2d.drawOptions()
        opts.clearBackground = False
        opts.padding = 0.05  # Minimal padding
        opts.bondLineWidth = 2.0  # Make bonds more visible
        
        # Draw the molecule
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        
        # Get PNG data and convert to PIL Image
        png_data = d2d.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        
        # Crop transparent edges and convert to base64
        img = _crop_transparent_edges(img)
        img_str = _convert_pil_to_base64(img)
        
        return f"<img src='{img_str}' style='max-width: 100%; height: auto;' title='{smiles}' />"
        
    except Exception as e:
        return f"<div style='text-align: center; color: red;'>Error: {str(e)}</div>"


def spectrum_to_html_img(spec1, spec2, img_size=SPECTRUM_IMG_SIZE):
    """
    Convert spectrum plot to HTML image for display in Gradio dataframe
    
    Args:
        spec1: First spectrum data
        spec2: Second spectrum data (for mirror plot)
        img_size: Size of the output image (default: SPECTRUM_IMG_SIZE)
    
    Returns:
        str: HTML img tag with base64 encoded spectrum plot
    """
    try:
        # Use non-interactive matplotlib backend
        matplotlib.use('Agg')
        
        # Create the spectrum plot using DreaMS utility function
        su.plot_spectrum(spec=spec1, mirror_spec=spec2, figsize=(2, 1))
        
        # Save figure to buffer with transparent background
        buffered = BytesIO()
        plt.savefig(buffered, format='png', bbox_inches='tight', dpi=100, transparent=True)
        buffered.seek(0)
        
        # Convert to PIL Image, crop edges, and convert to base64
        img = Image.open(buffered)
        img = _crop_transparent_edges(img)
        img_str = _convert_pil_to_base64(img)
        
        # Clean up matplotlib figure to free memory
        plt.close()
        
        return f"<img src='{img_str}' style='max-width: 100%; height: auto;' title='Spectrum comparison' />"
        
    except Exception as e:
        return f"<div style='text-align: center; color: red;'>Error: {str(e)}</div>"


# =============================================================================
# DATA DOWNLOAD AND SETUP FUNCTIONS
# =============================================================================

def _download_file(url, target_path, description):
    """
    Download a file from URL if it doesn't exist
    
    Args:
        url: Source URL
        target_path: Target file path
        description: Description for logging
    """
    if not target_path.exists():
        print(f"Downloading {description}...")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target_path)
        print(f"Downloaded {description} to {target_path}")


def setup():
    """
    Initialize the application by downloading required data files
    
    Downloads:
    - MassSpecGym spectral library
    - Example MS/MS files for testing
    
    Raises:
        Exception: If critical setup steps fail
    """
    print("=" * 60)
    print("Setting up DreaMS application...")
    print("=" * 60)
    
    try:
        # Download spectral library
        library_url = 'https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/auxiliary/MassSpecGym_DreaMS.hdf5'
        _download_file(library_url, LIBRARY_PATH, "MassSpecGym spectral library")
        
        # Download example files
        example_urls = [
            ('https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/auxiliary/example_piper_2k_spectra.mgf',
             EXAMPLE_PATH / 'example_piper_2k_spectra.mgf',
             "PiperNET example spectra"),
            ('https://raw.githubusercontent.com/pluskal-lab/DreaMS/cc806fa6fea281c1e57dd81fc512f71de9290017/data/examples/example_5_spectra.mgf',
             EXAMPLE_PATH / 'example_5_spectra.mgf',
             "DreaMS example spectra")
        ]
        
        for url, path, desc in example_urls:
            _download_file(url, path, desc)
        
        # Test DreaMS embeddings to ensure everything works
        print("\nTesting DreaMS embeddings...")
        test_path = EXAMPLE_PATH / 'example_5_spectra.mgf'
        embs = dreams_embeddings(test_path)
        print(f"✓ Setup complete - DreaMS embeddings test successful (shape: {embs.shape})")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        print("The application may not work properly. Please check your internet connection and try again.")
        raise


# =============================================================================
# CORE PREDICTION FUNCTIONS
# =============================================================================

@spaces.GPU
def _predict_gpu(in_pth, progress):
    """
    GPU-accelerated prediction of DreaMS embeddings
    
    Args:
        in_pth: Input file path
        progress: Gradio progress tracker
    
    Returns:
        numpy.ndarray: DreaMS embeddings
    """
    progress(0.1, desc="Loading spectra data...")
    msdata = MSData.load(in_pth)
    
    progress(0.2, desc="Computing DreaMS embeddings...")
    embs = dreams_embeddings(msdata)
    print(f'Shape of the query embeddings: {embs.shape}')
    
    return embs


def _create_result_row(i, j, n, msdata, msdata_lib, sims, cos_sim, embs):
    """
    Create a single result row for the DataFrame
    
    Args:
        i: Query spectrum index
        j: Library spectrum index
        n: Top-k rank
        msdata: Query MS data
        msdata_lib: Library MS data
        sims: Similarity matrix
        cos_sim: Cosine similarity calculator
        embs: Query embeddings
    
    Returns:
        dict: Result row data
    """
    smiles = msdata_lib.get_smiles(j)
    spec1 = msdata.get_spectra(i)
    spec2 = msdata_lib.get_spectra(j)
    
    return {
        'feature_id': i + 1,
        'precursor_mz': msdata.get_prec_mzs(i),
        'topk': n + 1,
        'library_j': j,
        'library_SMILES': smiles_to_html_img(smiles),
        'library_SMILES_raw': smiles,
        'Spectrum': spectrum_to_html_img(spec1, spec2),
        'Spectrum_raw': su.unpad_peak_list(spec1),
        'library_ID': msdata_lib.get_values('IDENTIFIER', j),
        'DreaMS_similarity': sims[i, j],
        'Modified_cosine_similarity': cos_sim(
            spec1=spec1,
            prec_mz1=msdata.get_prec_mzs(i),
            spec2=spec2,
            prec_mz2=msdata_lib.get_prec_mzs(j),
        ),
        'i': i,
        'j': j,
        'DreaMS_embedding': embs[i],
    }


def _process_results_dataframe(df, in_pth):
    """
    Process and clean the results DataFrame
    
    Args:
        df: Raw results DataFrame
        in_pth: Input file path for CSV export
    
    Returns:
        tuple: (processed_df, csv_path)
    """
    # Sort hits by DreaMS similarity
    df_top1 = df[df['topk'] == 1].sort_values('DreaMS_similarity', ascending=False)
    df = df.set_index('feature_id').loc[df_top1['feature_id'].values].reset_index()
    
    # Remove unnecessary columns and round similarity scores
    df = df.drop(columns=['i', 'j', 'library_j'])
    df['DreaMS_similarity'] = df['DreaMS_similarity'].astype(float).round(4)
    df['Modified_cosine_similarity'] = df['Modified_cosine_similarity'].astype(float).round(4)
    df['precursor_mz'] = df['precursor_mz'].astype(float).round(4)
    
    # Rename columns for display
    column_mapping = {
        'topk': 'Top k',
        'library_ID': 'Library ID',
        "feature_id": "Feature ID",
        "precursor_mz": "Precursor m/z",
        "library_SMILES": "Molecule",
        "library_SMILES_raw": "SMILES",
        "Spectrum": "Spectrum",
        "Spectrum_raw": "Input Spectrum",
        "DreaMS_similarity": "DreaMS similarity",
        "Modified_cosine_similarity": "Modified cos similarity",
        "DreaMS_embedding": "DreaMS embedding",
    }
    df = df.rename(columns=column_mapping)
    
    # Save full results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_path = dio.append_to_stem(in_pth, f"MassSpecGym_hits_{timestamp}").with_suffix('.csv')
    df_to_save = df.drop(columns=['Molecule', 'Spectrum', 'Top k'])
    df_to_save.to_csv(df_path, index=False)
    
    # Filter and prepare final display DataFrame
    df = df.drop(columns=['DreaMS embedding', "SMILES", "Input Spectrum"])
    df = df[df['Top k'] == 1].sort_values('DreaMS similarity', ascending=False)
    df = df.drop(columns=['Top k'])
    df = df[df["DreaMS similarity"] >= SIMILARITY_THRESHOLD]
    
    # Add row numbers
    df.insert(0, 'Row', range(1, len(df) + 1))
    
    return df, str(df_path)


def _predict_core(lib_pth, in_pth, progress):
    """
    Core prediction function that orchestrates the entire prediction pipeline
    
    Args:
        lib_pth: Library file path
        in_pth: Input file path
        progress: Gradio progress tracker
    
    Returns:
        tuple: (results_dataframe, csv_file_path)
    """
    in_pth = Path(in_pth)
    
    # Load library data
    progress(0, desc="Loading library data...")
    msdata_lib = MSData.load(lib_pth)
    embs_lib = msdata_lib[DREAMS_EMBEDDING]
    print(f'Shape of the library embeddings: {embs_lib.shape}')
    
    # Get query embeddings
    embs = _predict_gpu(in_pth, progress)
    
    # Compute similarity matrix
    progress(0.4, desc="Computing similarity matrix...")
    sims = cosine_similarity(embs, embs_lib)
    print(f'Shape of the similarity matrix: {sims.shape}')
    
    # Get top-k candidates
    k = 1
    topk_cands = np.argsort(sims, axis=1)[:, -k:][:, ::-1]
    
    # Load query data for processing
    msdata = MSData.load(in_pth)
    print(f'Available columns: {msdata.columns()}')
    
    # Construct results DataFrame
    progress(0.5, desc="Constructing results table...")
    df = []
    cos_sim = su.PeakListModifiedCosine()
    total_spectra = len(topk_cands)
    
    for i, topk in enumerate(topk_cands):
        progress(0.5 + 0.4 * (i / total_spectra), 
                desc=f"Processing hits for spectrum {i+1}/{total_spectra}...")
        
        for n, j in enumerate(topk):
            row_data = _create_result_row(i, j, n, msdata, msdata_lib, sims, cos_sim, embs)
            df.append(row_data)
    
    df = pd.DataFrame(df)
    
    # Process and clean results
    progress(0.9, desc="Post-processing results...")
    df, csv_path = _process_results_dataframe(df, in_pth)
    
    progress(1.0, desc=f"Predictions complete! Found {len(df)} high-confidence matches.")
    
    return df, csv_path


def predict(lib_pth, in_pth, progress=gr.Progress(track_tqdm=True)):
    """
    Main prediction function with error handling
    
    Args:
        lib_pth: Library file path
        in_pth: Input file path
        progress: Gradio progress tracker
    
    Returns:
        tuple: (results_dataframe, csv_file_path)
    
    Raises:
        gr.Error: If prediction fails or input is invalid
    """
    try:
        # Validate input file
        if not _validate_input_file(in_pth):
            raise gr.Error("Invalid input file. Please provide a valid .mgf or .mzML file.")
        
        # Check if library exists
        if not Path(lib_pth).exists():
            raise gr.Error("Spectral library not found. Please ensure the library file exists.")
        
        return _predict_core(lib_pth, in_pth, progress)
        
    except gr.Error:
        # Re-raise Gradio errors as-is
        raise
    except Exception as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg:
            error_msg = f"GPU/CUDA error: {error_msg}. The app is falling back to CPU mode."
        elif "RuntimeError" in error_msg:
            error_msg = f"Runtime error: {error_msg}. This may be due to memory or device issues."
        else:
            error_msg = f"Error: {error_msg}"
        
        print(f"Prediction failed: {error_msg}")
        raise gr.Error(error_msg)


# =============================================================================
# GRADIO INTERFACE SETUP
# =============================================================================

def _create_gradio_interface():
    """
    Create and configure the Gradio interface
    
    Returns:
        gr.Blocks: Configured Gradio app
    """
    # JavaScript for theme management
    js_func = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'light') {
            url.searchParams.set('__theme', 'light');
            window.location.href = url.href;
        }
    }
    """
    
    # Create app with custom theme
    app = gr.Blocks(
        theme=gr.themes.Default(primary_hue="yellow", secondary_hue="pink"), 
        js=js_func
    )
    
    with app:
        # Header and description
        gr.Image("https://raw.githubusercontent.com/pluskal-lab/DreaMS/cc806fa6fea281c1e57dd81fc512f71de9290017/assets/dreams_background.png", 
                label="DreaMS")
        
        gr.Markdown(value="""
            DreaMS (Deep Representations Empowering the Annotation of Mass Spectra) is a transformer-based
             neural network designed to interpret tandem mass spectrometry (MS/MS) data (<a href="https://www.nature.com/articles/s41587-025-02663-3">Bushuiev et al., Nature Biotechnology, 2025</a>).
             This website provides an easy access to perform library matching with DreaMS. Please upload
             your MS/MS file and click on the "Run DreaMS" button. Predictions may currently take up to 10 minutes for files with several thousands of spectra.
        """)
        
        # Input section
        with gr.Row(equal_height=True):
            in_pth = gr.File(
                file_count="single",
                label="Input MS/MS file (.mgf or .mzML)",
            )
        
        # Example files
        examples = gr.Examples(
            examples=["./data/example_5_spectra.mgf", "./data/example_piper_2k_spectra.mgf"],
            inputs=[in_pth],
            label="Examples (click on a file to load as input)",
        )
        
        # Prediction button
        predict_button = gr.Button(value="Run DreaMS", variant="primary")
        
        # Output section
        gr.Markdown("## Predictions")
        df_file = gr.File(label="Download predictions as .csv", interactive=False, visible=True)
        
        # Results table
        df = gr.Dataframe(
            headers=["Row", "Feature ID", "Precursor m/z", "Molecule", "Spectrum", 
                    "Library ID", "DreaMS similarity", "Modified cosine similarity"],
            datatype=["number", "number", "number", "html", "html", "str", "number", "number"],
            col_count=(8, "fixed"),
            column_widths=["25px", "25px", "28px", "60px", "60px", "50px", "40px", "40px"],
            max_height=1000,
            show_fullscreen_button=True,
            show_row_numbers=False,
            show_search='filter',
        )
        
        # Connect prediction logic
        inputs = [in_pth]
        outputs = [df, df_file]
        predict_func = partial(predict, LIBRARY_PATH)
        predict_button.click(predict_func, inputs=inputs, outputs=outputs, show_progress="first")
    
    return app


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize the application
    setup()
    
    # Create and launch the Gradio interface
    app = _create_gradio_interface()
    app.launch(allowed_paths=['./assets'])
else:
    # When imported as a module, just run setup
    setup()
