"""
DreaMS Gradio Web Application

This module provides a web interface for the DreaMS (Deep Representations Empowering
the Annotation of Mass Spectra) tool using Gradio. It allows users to upload MS/MS
files and perform library matching with DreaMS embeddings.

Author: DreaMS Team
License: MIT
"""

import base64
import io
import shutil
from textwrap import wrap
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import gradio as gr
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd
import spaces
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm

import dreams.utils.io as dio
import dreams.utils.spectra as su
from dreams.api import DreaMSSearch, dreams_embeddings
from dreams.definitions import CHARGE, PRECURSOR_MZ, SPECTRUM, DREAMS_EMBEDDING, IONMODE
from dreams.utils.data import MSData
from dreams.utils.dformats import assign_dformat

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Optimized image sizes for better performance
SMILES_IMG_SIZE = 120  # Reduced from 200 for faster rendering
SPECTRUM_IMG_SIZE = 800  # Reduced from 1500 for faster generation

# Supported input formats
SUPPORTED_INPUT_EXTENSIONS = {'.mgf', '.mzml', '.hdf5'}

# Library and data paths
LIBRARY_PATH = Path("DreaMS/data/MassSpecGym_DreaMS.hdf5")
DATA_PATH = Path("./DreaMS/data")
EXAMPLE_PATH = Path("./data")

EXAMPLE_FILES: Tuple[Tuple[str, Path, str], ...] = (
    (
        "https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/auxiliary/Piper55-Leaf-r2_1uL_damiani2023.mzML",
        EXAMPLE_PATH / "Piper55-Leaf-r2_1uL_damiani2023.mzML",
        "PiperNET example spectra",
    ),
    (
        "https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/auxiliary/example_5_drugs_zhao2025.mgf",
        EXAMPLE_PATH / "example_5_drugs_zhao2025.mgf",
        "Drug analogs example spectra",
    ),
)

DATAFRAME_COLUMNS: Tuple[dict[str, str], ...] = (
    {"name": "Row", "header": "Row", "datatype": "number", "width": "50px"},
    {"name": "Scan number", "header": "Scan\nnumber", "datatype": "number", "width": "85px"},
    {"name": "Precursor m/z", "header": "Precursor\nm/z", "datatype": "number", "width": "130px"},
    # {"name": "Adduct", "header": "Adduct", "datatype": "str", "width": "150px"},
    {"name": "RT", "header": "RT", "datatype": "number", "width": "60px"},
    {"name": "Molecule", "header": "Molecule", "datatype": "html", "width": "150px"},
    {"name": "Name", "header": "Name", "datatype": "str", "width": "120px"},
    {"name": "Spectrum", "header": "Spectrum", "datatype": "html", "width": "150px"},
    {"name": "Ref. precursor m/z", "header": "Ref. precursor\nm/z", "datatype": "html", "width": "130px"},
    # {"name": "Ref. adduct", "header": "Ref. adduct", "datatype": "str", "width": "150px"},
    {"name": "Ref. RT", "header": "Ref.\nRT", "datatype": "number", "width": "60px"},
    {"name": "Ref. molecule", "header": "Ref. molecule", "datatype": "html", "width": "135px"},
    {"name": "Ref. name", "header": "Ref. name", "datatype": "str", "width": "150px"},
    {"name": "Ref. scan number", "header": "Ref. scan\nnumber", "datatype": "number", "width": "85px"},
    {"name": "Ref. ID", "header": "Ref.\nID", "datatype": "str", "width": "130px"},
    {"name": "DreaMS similarity", "header": "DreaMS\nsimilarity", "datatype": "number", "width": "110px"},
    {"name": "Modified cos. sim.", "header": "Modified\ncos. sim.", "datatype": "number", "width": "140px"},
)

DATAFRAME_CSS = """
#results-dataframe {
    overflow-x: auto;
}
#results-dataframe table th,
#results-dataframe table th * {
    white-space: pre-line !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    text-overflow: clip;
}
"""

def _extract_dataframe_config(
    existing_columns: Optional[Sequence[str]] = None,
) -> Tuple[list[str], list[str], list[str]]:
    """Build dataframe configuration filtered to the supplied column names."""
    cols = DATAFRAME_COLUMNS
    if existing_columns is not None:
        cols = [col for col in DATAFRAME_COLUMNS if col["name"] in existing_columns]

    headers = [col.get("header", col["name"]) for col in cols]
    datatypes = [col["datatype"] for col in cols]
    widths = [col["width"] for col in cols]
    return headers, datatypes, widths


def _build_empty_results_dataframe() -> pd.DataFrame:
    """Return an empty dataframe that matches the display schema."""
    return pd.DataFrame({col["name"]: pd.Series(dtype="object") for col in DATAFRAME_COLUMNS})


# Styling for analog hits indicator rendered alongside reference precursor m/z
_ANALOG_TAG_STYLE = (
    "display:inline-block;padding:2px 8px;border-radius:999px;"
    "background-color:#f25d64;color:#fff;font-size:12px;font-weight:600;"
    "line-height:1;"
)
_REF_MZ_CONTAINER_STYLE = "display:inline-flex;align-items:center;gap:6px;"
_REF_MZ_VALUE_STYLE = "font-variant-numeric:tabular-nums;font-weight:500;"


# Cache for SMILES images to avoid regeneration
_smiles_cache = {}

def clear_smiles_cache() -> None:
    """Clear the SMILES image cache to free memory"""
    global _smiles_cache
    _smiles_cache.clear()
    print("SMILES image cache cleared")

# =============================================================================
# UTILITY FUNCTIONS FOR IMAGE CONVERSION
# =============================================================================

def _validate_input_file(file_path: Union[str, Path]) -> bool:
    """Return True when the user-supplied input file path is valid."""
    if not file_path or not Path(file_path).exists():
        return False
    
    file_ext = Path(file_path).suffix.lower()
    
    return file_ext in SUPPORTED_INPUT_EXTENSIONS


def _convert_pil_to_base64(img: Image.Image, format: str = 'PNG') -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffered = io.BytesIO()
    img.save(buffered, format=format, optimize=True)  # Added optimize=True
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/{format.lower()};base64,{repr(img_str)[2:-1]}"


def _crop_transparent_edges(img: Image.Image) -> Image.Image:
    """Crop transparent edges from a PIL Image."""
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
    Uses caching to avoid regenerating the same molecule images
    
    Args:
        smiles: SMILES string representation of molecule
        img_size: Size of the output image (default: SMILES_IMG_SIZE)

    Returns:
        str: HTML img tag with base64 encoded image
    """
    # Check cache first
    cache_key = f"{smiles}_{img_size}"
    if cache_key in _smiles_cache:
        return _smiles_cache[cache_key]
    
    try:
        # Parse SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result = f"<div style='text-align: center; color: red;'>Invalid SMILES</div>"
            _smiles_cache[cache_key] = result
            return result
        
        # Create PNG drawing with Cairo backend for better control
        d2d = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
        opts = d2d.drawOptions()
        opts.clearBackground = False
        opts.padding = 0.05  # Minimal padding
        opts.bondLineWidth = 1.5  # Reduced from 2.0 for smaller images
        
        # Draw the molecule
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        
        # Get PNG data and convert to PIL Image
        png_data = d2d.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        
        # Crop transparent edges and convert to base64
        img = _crop_transparent_edges(img)
        img_str = _convert_pil_to_base64(img)
        
        result = f"<img src='{img_str}' style='max-width: 100%; height: auto;' title='{smiles}' />"
        
        # Cache the result
        _smiles_cache[cache_key] = result
        return result
        
    except Exception as e:
        result = f"<div style='text-align: center; color: red;'>Error: {str(e)}</div>"
        _smiles_cache[cache_key] = result
        return result


def _format_ref_precursor_mz_value(value: Any, analog_hit: bool) -> str:
    """Return HTML snippet for ref precursor m/z with optional analog-hit tag."""
    try:
        numeric_value = float(value)
        formatted_value = f"{numeric_value:.4f}".rstrip('0').rstrip('.')
        if not formatted_value:
            formatted_value = "0"
    except (TypeError, ValueError):
        formatted_value = str(value)

    value_html = f"<span style='{_REF_MZ_VALUE_STYLE}'>{formatted_value}</span>"
    parts = [value_html]
    if analog_hit:
        parts.append(f"<span style='{_ANALOG_TAG_STYLE}'>Analog hit</span>")

    return f"<span style='{_REF_MZ_CONTAINER_STYLE}'>{''.join(parts)}</span>"


def spectrum_to_html_img(
    spec1: Any,
    spec2: Any,
    img_size: int = SPECTRUM_IMG_SIZE,
) -> str:
    """Convert a spectrum (and optional mirror spectrum) to an embeddable HTML image."""
    try:
        # Create the spectrum plot using DreaMS utility function
        su.plot_spectrum(spec=spec1, mirror_spec=spec2, figsize=(1.6, 0.8))  # Reduced size for performance

        # Render the current figure using the Agg backend
        fig = plt.gcf()
        canvas = fig.canvas
        if not isinstance(canvas, FigureCanvasAgg):
            canvas = FigureCanvasAgg(fig)
            fig.set_canvas(canvas)
        canvas.draw()
        
        # Save figure to buffer with tight layout to retain axis labels
        with BytesIO() as buffered:
            fig.savefig(
                buffered,
                format='png',
                bbox_inches='tight',
                dpi=70,
                transparent=True,
            )
            buffered.seek(0)
            
            # Convert to PIL Image, crop edges, and convert to base64
            img = Image.open(buffered)
            img.load()
            img = _crop_transparent_edges(img)
            img_str = _convert_pil_to_base64(img)
        
        # Clean up matplotlib figure to free memory
        plt.close(fig)
        
        return f"<img src='{img_str}' style='max-width: 100%; height: auto;' title='Spectrum comparison' />"
        
    except Exception as e:
        return f"<div style='text-align: center; color: red;'>Error: {str(e)}</div>"


# =============================================================================
# DATA DOWNLOAD AND SETUP FUNCTIONS
# =============================================================================

def _download_file(url: str, target_path: Path, description: str) -> None:
    """Download a file from URL if it does not already exist."""
    if not target_path.exists():
        print(f"Downloading {description}...")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target_path)
        print(f"Downloaded {description} to {target_path}")


def setup() -> None:
    """Initialize the application by downloading required data files."""
    print("=" * 60)
    print("Setting up DreaMS application...")
    print("=" * 60)
    
    # Clear any existing cache
    clear_smiles_cache()
    
    try:
        # Download spectral library
        library_url = 'https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/auxiliary/MassSpecGym_DreaMS.hdf5'
        _download_file(library_url, LIBRARY_PATH, "MassSpecGym spectral library")
        
        # Download example files
        for url, path, desc in EXAMPLE_FILES:
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
def _predict_gpu(
    msdata: MSData,
    lib_msdata: MSData,
    similarity_threshold: float,
    progress: gr.Progress,
) -> pd.DataFrame:
    """Execute the search step on GPU (if available) and return raw matches."""
    progress(0.3, desc="Initializing DreaMS search engine...")
    searcher = DreaMSSearch(ref_spectra=lib_msdata)

    progress(0.6, desc="Preparing input spectra for search...")
    df = searcher.query(query_spectra=msdata, k=1, dreams_sim_thld=similarity_threshold, out_embs=True)
    return df


def _rename_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Apply human-friendly column names for presentation."""
    columns = df.columns.tolist()
    columns = [
        c.replace('ref_', 'Ref._')
        .replace('smiles', 'SMILES')
        .replace('precursor_mz', 'precursor_m/z')
        .replace('IDENTIFIER', 'ID')
        # .replace('scan_number', 'feature_ID')
        .replace('SMILES', 'molecule')
        .replace('_', ' ')
        for c in columns
    ]
    def capitalize_first(s):
        return s[0].upper() + s[1:] if s else s
    columns = [capitalize_first(c) for c in columns]
    df.columns = columns
    return df


def _reformat_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns for readability in the results table."""
    for col in df.columns:
        if col.endswith('mz'):
            df[col] = df[col].astype(float).round(4)
        elif col.endswith('rt'):
            df[col] = df[col].astype(float).round(2)
        elif col.endswith('similarity'):
            df[col] = df[col].astype(float).round(4)
        elif col.endswith('RT'):
            df[col] = (df[col] / 60).round(2)  # Seconds to minutes
        elif col.endswith('Modified_cos._sim.'):
            df[col] = df[col].astype(float).round(4)
    return df


def _filter_input_data(
    pth: Path,
    only_single_charge: bool = True,
    only_high_quality_spectra: bool = True,
) -> Path:
    """Return a filtered copy of the input MSData file according to quality filters."""
    msdata = MSData.load(pth, in_mem=True)
    print(f"Original number of rows in {pth.name}: {len(msdata)}")

    idx = []
    for i in tqdm(range(len(msdata)), desc=f"Filtering dataset {pth.name}"):
        if only_single_charge:
            # if IONMODE in msdata.columns():
            #     if msdata.get_values(IONMODE, idx=i) == '-':
            #         continue
            if CHARGE in msdata.columns():
                charge = msdata.get_values(CHARGE, idx=i)
                if charge > 1 or charge < -1:  # -1 if often used for unknown charge?
                    continue
        if only_high_quality_spectra:
            if assign_dformat(msdata.get_values(SPECTRUM, i), msdata.get_values(PRECURSOR_MZ, i)) != 'A':
                continue
        idx.append(i)

    pth_filtered = pth.with_suffix('.filtered.hdf5')
    msdata_filtered = msdata.form_subset(idx=idx, out_pth=pth_filtered)
    print(f"Filtered number of rows in {pth_filtered.name}: {len(msdata_filtered)}")

    if len(msdata_filtered) == 0:
        raise ValueError(f"No spectra passed the quality filters. Please disable 'Only predict on high-quality input"
        "spectra' or check your input file.")

    return pth_filtered


def _predict_core(
    lib_pth: Union[str, Path],
    in_pth: Union[str, Path],
    similarity_threshold: float,
    calculate_modified_cosine: bool,
    progress: gr.Progress,
    only_high_quality_input: bool,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Coordinate the full library search pipeline for DreaMS predictions."""
    in_pth = Path(in_pth)
    lib_pth = Path(lib_pth)

    # Clear cache at start to prevent memory buildup
    clear_smiles_cache()

    # Create temporary copies of library and input files to allow multiple processes
    progress(0, desc="Creating temporary file copies...")
    temp_lib_path = dio.append_to_stem(lib_pth, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    temp_in_path = dio.append_to_stem(in_pth, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(lib_pth, temp_lib_path)
    shutil.copy2(in_pth, temp_in_path)

    try:
        temp_in_path = _filter_input_data(temp_in_path, only_single_charge=True, only_high_quality_spectra=only_high_quality_input)
        # temp_lib_path = _filter_input_data(temp_lib_path)
        df = _predict_gpu(temp_in_path, temp_lib_path, similarity_threshold, progress)

        if df is None or (hasattr(df, "empty") and df.empty):
            progress(1.0, desc="No matches found.")
            return _build_empty_results_dataframe(), None

        # Add modified cosine similarity only if enabled
        if calculate_modified_cosine:
            cos_sims = []
            modified_cosine_sim = su.PeakListModifiedCosine()
            for i in tqdm(range(len(df)), desc="Calculating modified cosine similarity"):
                cos_sims.append(modified_cosine_sim(
                    spec1=df[SPECTRUM].iloc[i],
                    prec_mz1=df[PRECURSOR_MZ].iloc[i],
                    spec2=df[f'ref_{SPECTRUM}'].iloc[i],
                    prec_mz2=df[f'ref_{PRECURSOR_MZ}'].iloc[i],
                ))
            df['Modified_cos._sim.'] = cos_sims

        # Add row number for display
        if 'Row' not in df.columns:
            df.insert(0, 'Row', list(range(1, len(df) + 1)))
        
        df['analog_hit'] = (df[PRECURSOR_MZ] - df[f'ref_{PRECURSOR_MZ}']).round(2).abs() >= 0.01

        # Store results to CSV
        progress(0.7, desc="Saving results to TSV...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_path = dio.append_to_stem(in_pth, f"{lib_pth.stem}_hits_{timestamp}").with_suffix('.tsv')

        # Convert spectrum to lists before saving to TSV
        df_to_save = df.copy()
        for col in df_to_save.columns:
            if col.endswith(SPECTRUM):
                df_to_save[col] = df_to_save[col].apply(lambda x: su.unpad_peak_list(x).tolist())

        df_to_save.to_csv(df_path, index=False, sep='\t')

        for col in df_to_save.columns:
            if col.endswith(IONMODE):
                if '-' in df_to_save[col].tolist():
                    # Note: As of Gradio 3.x/4.x, gr.Warning does not natively support duration control through its API.
                    gr.Warning(
                        "Negative mode spectra found. Please note that the current version of DreaMS was "
                        "trained on positive mode spectra only. This may lead to unexpected results.",
                        duration=30
                    )
                    break

        # Subsequent code is performed after saving to TSV, for display dataframe only
        progress(0.85, desc="Painting molecules...")
        for col in df.columns:
            if col.endswith('smiles'):
                rendered_smiles = []
                for idx, smiles in tqdm(enumerate(df[col], start=1), desc="Painting molecules", total=len(df[col])):
                    rendered_smiles.append(smiles_to_html_img(smiles))
                    if idx % 100 == 0:
                        clear_smiles_cache()
                df[col] = rendered_smiles

        progress(0.9, desc="Painting spectra...")
        df[SPECTRUM] = [
            spectrum_to_html_img(query, ref)
            for query, ref in tqdm(zip(df[SPECTRUM], df[f'ref_{SPECTRUM}']), desc="Painting spectra", total=len(df))
        ]
        
        print('Columns:')
        print(df.columns)

        df = _reformat_columns_for_display(df)

        analog_column = df['analog_hit'] if 'analog_hit' in df.columns else None
        analog_flags = (
            analog_column.fillna(False).astype(bool).tolist()
            if analog_column is not None
            else [False] * len(df)
        )
        ref_precursor_values = df[f'ref_{PRECURSOR_MZ}'].tolist()
        if len(analog_flags) < len(ref_precursor_values):
            analog_flags.extend([False] * (len(ref_precursor_values) - len(analog_flags)))
        elif len(analog_flags) > len(ref_precursor_values):
            analog_flags = analog_flags[:len(ref_precursor_values)]
        df[f'ref_{PRECURSOR_MZ}'] = [
            _format_ref_precursor_mz_value(value, analog_flags[idx])
            for idx, value in enumerate(ref_precursor_values)
        ]
        if 'analog_hit' in df.columns:
            df = df.drop(columns=['analog_hit'])

        df = _rename_columns_for_display(df)

        print('Renamed columns:')
        print(df.columns)

        df = df[[c['name'] for c in DATAFRAME_COLUMNS if c['name'] in df.columns]]
        
        progress(1.0, desc=f"Predictions complete! Found {len(df)} high-confidence matches.")

        return df, str(df_path)
    
    finally:
        # Clean up temporary files
        if temp_lib_path.exists():
            temp_lib_path.unlink()
        if temp_in_path.exists():
            temp_in_path.unlink()


def predict(
    lib_pth: Union[str, Path],
    in_pth: Union[str, Path],
    similarity_threshold: float = 0.75,
    calculate_modified_cosine: bool = False,
    only_high_quality_input: bool = True,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[Any, Any]:
    """Main prediction entry point with user-facing error handling."""
    try:
        # Validate input file
        if not _validate_input_file(in_pth):
            raise gr.Error("Invalid input file. Please provide a valid .mgf, .mzML, or .hdf5 file.")
        
        # Check if library exists
        if not Path(lib_pth).exists():
            raise gr.Error("Spectral library not found. Please ensure the library file exists.")
        
        df_raw, csv_path = _predict_core(
            lib_pth,
            in_pth,
            similarity_threshold,
            calculate_modified_cosine,
            progress,
            only_high_quality_input,
        )

        headers, datatype, column_widths = _extract_dataframe_config(df_raw.columns)

        df = gr.update(
            value=df_raw,
            headers=headers,
            datatype=datatype,
            column_widths=column_widths,
            column_count=(len(headers), "fixed"),
        )

        if isinstance(df_raw, pd.DataFrame) and df_raw.empty:
            gr.Info("No matches were found. Consider lowering the DreaMS similarity threshold for finding analog matches or checking your input file.")

        if csv_path:
            file_update = gr.update(value=csv_path, visible=True, interactive=False)
        else:
            file_update = gr.update(value=None, visible=False, interactive=False)

        return df, file_update
        
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

def _create_gradio_interface() -> gr.Blocks:
    """Create and configure the Gradio Blocks interface."""
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
        js=js_func,
        css=DATAFRAME_CSS,
    )
    
    with app:
        # Header and description
        gr.Image(
            "https://raw.githubusercontent.com/pluskal-lab/DreaMS/cc806fa6fea281c1e57dd81fc512f71de9290017/assets/dreams_background.png",
            label="DreaMS"
        )

        gr.Markdown(
            value=(
                "DreaMS (Deep Representations Empowering the Annotation of Mass Spectra) is a "
                "transformer-based neural network designed to interpret tandem mass spectrometry (MS/MS) "
                "data (<a href=\"https://www.nature.com/articles/s41587-025-02663-3\">Bushuiev et al., Nature Biotechnology, 2025</a>). "
                "This website provides an easy access to perform spectral library searches to identify small molecules or their analogue candidates by querying "
                "<a href=\"https://huggingface.co/datasets/roman-bushuiev/MassSpecGym\">MassSpecGym</a> spectral library (combination of GNPS, MoNA, and Pluskal lab data) "
                "or custom MS/MS datasets. "
                "Please upload your file with MS/MS data and click on the \"Run DreaMS\" button. In case of any issues, questions, or feedback, "
                "please don't hesitate to open an issue on the <a href=\"https://github.com/pluskal-lab/DreaMS/issues\">DreaMS GitHub</a> page."
            )
        )

        # Input section
        with gr.Row(equal_height=True):
            in_pth = gr.File(
                file_count="single",
                label="Input MS/MS file (.mgf, .mzML, .hdf5)",
            )

        # Example files
        examples = gr.Examples(
            examples=[
                "./data/example_5_drugs_zhao2025.mgf",
                "./data/Piper55-Leaf-r2_1uL_damiani2023.mzML"
            ],
            inputs=[in_pth],
            label="Examples (click on a file to load as input)",
        )

        # Settings section
        with gr.Accordion("⚙️ Settings", open=False):
            lib_pth = gr.File(
                file_count="single",
                label="Reference MS/MS file or spectral library (.mgf, .mzML, .hdf5)",
                value=str(LIBRARY_PATH),
                interactive=True,
                visible=True,
            )
            similarity_threshold = gr.Slider(
                minimum=-1.0,
                maximum=1.0,
                value=0.75,
                step=0.01,
                label="Similarity threshold",
                info=(
                    "Only display library matches with DreaMS similarity above this threshold "
                    "(rendering less results also makes calculation faster)"
                ),
            )
            calculate_modified_cosine = gr.Checkbox(
                label="Calculate modified cosine similarity",
                value=False,
                info=(
                    "Enable to also calculate traditional modified cosine similarity scores between "
                    "the input spectra and library hits (a bit slower)"
                ),
            )
            only_high_quality_input = gr.Checkbox(
                label="Only predict on high-quality input spectra",
                value=True,
                info=(
                    "Enable to exclude low-quality input spectra before prediction. MS/MS spectrum is considered "
                    "low-quality if it does not satisfy A quality criteria as defined in the DreaMS paper "
                    "(<a href='https://www.nature.com/articles/s41587-025-02663-3/figures/2'>Fig. 2b</a>)."
                ),
            )
        
        # Prediction button
        predict_button = gr.Button(value="Run DreaMS", variant="primary")
        
        # Results table
        gr.Markdown("## Predictions")
        df_file = gr.File(label="Download predictions as .tsv", interactive=False, visible=True)
        
        headers, datatype, column_widths = _extract_dataframe_config()

        df = gr.Dataframe(
            headers=headers,
            datatype=datatype,
            col_count=(len(headers), "fixed"),
            column_widths=column_widths,
            max_height=1000,
            show_row_numbers=False,
            show_search='filter',
            wrap=True,
            interactive=False,
            pinned_columns=1,
            elem_id="results-dataframe"
        )
        
        # Connect prediction logic
        inputs = [lib_pth, in_pth, similarity_threshold, calculate_modified_cosine, only_high_quality_input]
        outputs = [df, df_file]
        
        # Function to update dataframe headers based on setting
        def update_headers(show_cosine):
            if show_cosine:
                return gr.update(headers=headers + ["Modified\ncosine similarity"],
                                col_count=(len(headers) + 1, "fixed"),
                                column_widths=column_widths + ["40px"])
            else:
                return gr.update(headers=headers,
                                col_count=(len(headers), "fixed"),
                                column_widths=column_widths)
        
        # Update headers when setting changes
        calculate_modified_cosine.change(
            fn=update_headers,
            inputs=[calculate_modified_cosine],
            outputs=[df]
        )
        
        predict_button.click(predict, inputs=inputs, outputs=outputs, show_progress="first")
    
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