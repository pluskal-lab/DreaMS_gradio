import gradio as gr
import spaces
import urllib.request
import torch
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import base64
from io import BytesIO
from PIL import Image
import io
import dreams.utils.spectra as su
import dreams.utils.io as dio
from dreams.utils.spectra import PeakListModifiedCosine
from dreams.utils.data import MSData
from dreams.api import dreams_embeddings
from dreams.definitions import *


def smiles_to_html_img(smiles, img_size=200):
    """
    Convert SMILES to HTML image string for display in Gradio dataframe
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"<div style='text-align: center; color: red;'>Invalid SMILES</div>"
        
        # Use PNG drawing for better control over cropping
        d2d = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
        opts = d2d.drawOptions()
        opts.clearBackground = False
        opts.padding = 0.05  # Minimal padding
        opts.bondLineWidth = 2.0  # Make bonds more visible
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        
        # Get PNG data
        png_data = d2d.GetDrawingText()
        
        # Convert PNG data to PIL Image for cropping
        img = Image.open(io.BytesIO(png_data))
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get the bounding box of non-transparent pixels
        bbox = img.getbbox()
        if bbox:
            # Crop the image to remove transparent space
            img = img.crop(bbox)
        
        # Convert back to base64
        buffered = io.BytesIO()
        img.save(buffered, format='PNG')
        img_str = base64.b64encode(buffered.getvalue())
        img_str = f"data:image/png;base64,{repr(img_str)[2:-1]}"
        
        return f"<img src='{img_str}' style='max-width: 100%; height: auto;' title='{smiles}' />"
    except Exception as e:
        return f"<div style='text-align: center; color: red;'>Error: {str(e)}</div>"


def spectrum_to_html_img(spec1, spec2, img_size=1500):
    """
    Convert spectrum plot to HTML image string for display in Gradio dataframe
    """
    try:
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create the plot using the existing function
        su.plot_spectrum(spec=spec1, mirror_spec=spec2, figsize=(2, 1))
        
        # Save the current figure to a buffer with transparent background
        buffered = BytesIO()
        plt.savefig(buffered, format='png', bbox_inches='tight', dpi=100, transparent=True)
        buffered.seek(0)
        
        # Convert to PIL Image for cropping
        img = Image.open(buffered)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get the bounding box of non-transparent pixels
        bbox = img.getbbox()
        if bbox:
            # Crop the image to remove transparent space
            img = img.crop(bbox)
        
        # Convert back to base64
        buffered_cropped = BytesIO()
        img.save(buffered_cropped, format='PNG')
        img_str = base64.b64encode(buffered_cropped.getvalue())
        img_str = f"data:image/png;base64,{repr(img_str)[2:-1]}"
        
        # Close the figure to free memory
        plt.close()
        
        return f"<img src='{img_str}' style='max-width: 100%; height: auto;' title='Spectrum comparison' />"
    except Exception as e:
        return f"<div style='text-align: center; color: red;'>Error: {str(e)}</div>"


def setup():
    # Download spectral library
    data_path = Path('./DreaMS/data')
    data_path.mkdir(parents=True, exist_ok=True)
    url = 'https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/auxiliary/MassSpecGym_DreaMS.hdf5'
    target_path = data_path / 'MassSpecGym_DreaMS.hdf5'
    if not target_path.exists():
        urllib.request.urlretrieve(url, target_path)

    # Download example file
    # example_url = 'https://huggingface.co/datasets/titodamiani/PiperNET/resolve/main/lcms/rawfiles/202312_147_P55-Leaf-r2_1uL.mzML'
    # example_path = Path('./data/202312_147_P55-Leaf-r2_1uL.mzML')
    example_url = 'https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/auxiliary/example_piper_2k_spectra.mgf'
    example_path = Path('./data/example_piper_2k_spectra.mgf')
    example_path.parent.mkdir(parents=True, exist_ok=True)
    if not example_path.exists():
        urllib.request.urlretrieve(example_url, example_path)

    # Run simple example as a test and to download weights
    example_url = 'https://raw.githubusercontent.com/pluskal-lab/DreaMS/cc806fa6fea281c1e57dd81fc512f71de9290017/data/examples/example_5_spectra.mgf'
    example_path = Path('./data/example_5_spectra.mgf')
    example_path.parent.mkdir(parents=True, exist_ok=True)
    if not example_path.exists():
        urllib.request.urlretrieve(example_url, example_path)
    embs = dreams_embeddings(example_path)
    print("Setup complete")


@spaces.GPU
def _predict_gpu(in_pth, progress):
    progress(0.1, desc="Loading spectra data...")
    msdata = MSData.load(in_pth)
    progress(0.2, desc="Computing DreaMS embeddings...")
    embs = dreams_embeddings(msdata)
    print('Shape of the query embeddings:', embs.shape)
    return embs


def _predict_core(lib_pth, in_pth, progress):
    """Core prediction function without error handling"""
    in_pth = Path(in_pth)
    # # in_pth = Path('DreaMS/data/MSV000086206/peak/mzml/S_N1.mzML')  # Example dataset
    
    progress(0, desc="Loading library data...")
    msdata_lib = MSData.load(lib_pth)
    embs_lib = msdata_lib[DREAMS_EMBEDDING]
    print('Shape of the library embeddings:', embs_lib.shape)
    
    embs = _predict_gpu(in_pth, progress)

    progress(0.4, desc="Computing similarity matrix...")
    sims = cosine_similarity(embs, embs_lib)
    print('Shape of the similarity matrix:', sims.shape)

    k = 1
    topk_cands = np.argsort(sims, axis=1)[:, -k:][:, ::-1]
    topk_cands.shape

    # TODO This is loaded for the 2nd time here, otpimize
    msdata = MSData.load(in_pth)
    print(msdata.columns())

    # Construct a DataFrame with the top-k candidates for each spectrum and their corresponding similarities
    progress(0.5, desc="Constructing results table...")
    df = []
    cos_sim = su.PeakListModifiedCosine()
    total_spectra = len(topk_cands)
    
    for i, topk in enumerate(topk_cands):
        progress(0.5 + 0.4 * (i / total_spectra), desc=f"Processing hits for spectrum {i+1}/{total_spectra}...")
        for n, j in enumerate(topk):
            smiles = msdata_lib.get_smiles(j)
            spec1 = msdata.get_spectra(i)
            spec2 = msdata_lib.get_spectra(j)
            df.append({
                'feature_id': i + 1,
                'precursor_mz': msdata.get_prec_mzs(i),
                # 'RT': msdata.get_values('RTINSECONDS', i),
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
            })
    df = pd.DataFrame(df)

    # Sort hits by DreaMS similarity
    df_top1 = df[df['topk'] == 1].sort_values('DreaMS_similarity', ascending=False)
    df = df.set_index('feature_id').loc[df_top1['feature_id'].values].reset_index()

    progress(0.9, desc="Post-processing results...")
    # Remove unnecessary columns and round similarity scores
    df = df.drop(columns=['i', 'j', 'library_j'])
    df['DreaMS_similarity'] = df['DreaMS_similarity'].astype(float).round(4)
    df['Modified_cosine_similarity'] = df['Modified_cosine_similarity'].astype(float).round(4)
    df['precursor_mz'] = df['precursor_mz'].astype(float).round(4)
    # df['RT'] = df['RT'].round(1)
    df = df.rename(columns={
        'topk': 'Top k',
        'library_ID': 'Library ID',
        "feature_id": "Feature ID",
        "precursor_mz": "Precursor m/z",
        # "RT": "RT",
        "library_SMILES": "Molecule",
        "library_SMILES_raw": "SMILES",
        "Spectrum": "Spectrum",
        "Spectrum_raw": "Input Spectrum",
        "DreaMS_similarity": "DreaMS similarity",
        "Modified_cosine_similarity": "Modified cos similarity",
        "DreaMS_embedding": "DreaMS embedding",
    })

    progress(0.95, desc="Saving results to CSV...")
    # Save full df to .csv
    df_path = dio.append_to_stem(in_pth, f"MassSpecGym_hits_{datetime.now().strftime('%Y%m%d_%H%M%S')}").with_suffix('.csv')
    df_to_save = df.drop(columns=['Molecule', 'Spectrum', 'Top k'])
    df_to_save.to_csv(df_path, index=False)

    progress(0.98, desc="Filtering and sorting results...")
    # Postprocess to only show most relevant hits
    df = df.drop(columns=['DreaMS embedding', "SMILES", "Input Spectrum"])
    df = df[df['Top k'] == 1].sort_values('DreaMS similarity', ascending=False)
    df = df.drop(columns=['Top k'])
    df = df[df["DreaMS similarity"] >= 0.75]
    # Add row numbers as first column
    df.insert(0, 'Row', range(1, len(df) + 1))
    
    progress(1.0, desc=f"Predictions complete! Found {len(df)} high-confidence matches.")

    return df, str(df_path)


def predict(lib_pth, in_pth, progress=gr.Progress(track_tqdm=True)):
    """Wrapper function with error handling"""
    try:
        return _predict_core(lib_pth, in_pth, progress)
    except Exception as e:
        raise gr.Error(e)


# Set up
setup()

# Start the Gradio app
js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""
app = gr.Blocks(theme=gr.themes.Default(primary_hue="yellow", secondary_hue="pink"), js=js_func)
with app:

    # Input GUI
    # gr.Markdown(value="""# DreaMS""")
    gr.Image("https://raw.githubusercontent.com/pluskal-lab/DreaMS/cc806fa6fea281c1e57dd81fc512f71de9290017/assets/dreams_background.png", label="DreaMS")
    gr.Markdown(value="""
        DreaMS (Deep Representations Empowering the Annotation of Mass Spectra) is a transformer-based
         neural network designed to interpret tandem mass spectrometry (MS/MS) data (<a href="https://www.nature.com/articles/s41587-025-02663-3">Bushuiev et al., Nature Biotechnology, 2025</a>).
         This website provides an easy access to perform library matching with DreaMS. Please upload
         your MS/MS file and click on the "Run DreaMS" button. Predictions may currently take up to 10 minutes for files with several thousands of spectra.
    """)
    with gr.Row(equal_height=True):
        in_pth = gr.File(
            file_count="single",
            label="Input MS/MS file (.mgf or .mzML)",
        )
    lib_pth = Path('DreaMS/data/MassSpecGym_DreaMS.hdf5')  # MassSpecGym library
    examples = gr.Examples(
        examples=["./data/example_5_spectra.mgf", "./data/example_piper_2k_spectra.mgf"],
        inputs=[in_pth],
        label="Examples (click on a file to load as input)",
    )

    # Predict GUI
    predict_button = gr.Button(value="Run DreaMS", variant="primary")

    # Output GUI
    gr.Markdown("## Predictions")
    df_file = gr.File(label="Download predictions as .csv", interactive=False, visible=True)
    df = gr.Dataframe(
        headers=["Row", "Feature ID", "Precursor m/z", "Molecule", "Spectrum", "Library ID", "DreaMS similarity", "Modified cosine similarity"],
        datatype=["number", "number", "number", "html", "html", "str", "number", "number"],
        col_count=(8, "fixed"),
        # wrap=True,
        column_widths=["25px", "25px", "28px", "60px", "60px", "50px", "40px", "40px"],
        max_height=1000,
        show_fullscreen_button=True,
        show_row_numbers=False,
        show_search='filter',
    )

    # Main logic
    inputs = [in_pth]
    outputs = [df, df_file]
    predict = partial(predict, lib_pth)
    predict_button.click(predict, inputs=inputs, outputs=outputs, show_progress="first")


app.launch(allowed_paths=['./assets'])
