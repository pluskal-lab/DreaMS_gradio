import gradio as gr
import urllib.request
import os
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
import dreams.utils.spectra as su
import dreams.utils.io as io
from dreams.utils.spectra import PeakListModifiedCosine
from dreams.utils.data import MSData
from dreams.api import dreams_embeddings
from dreams.definitions import *


def smiles_to_html_img(smiles, svg_size=1500):
    """
    Convert SMILES to HTML image string for display in Gradio dataframe
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"<div style='text-align: center; color: red;'>Invalid SMILES</div>"
        
        # Create SVG drawing
        d2d = rdMolDraw2D.MolDraw2DSVG(svg_size, svg_size)
        opts = d2d.drawOptions()
        opts.clearBackground = False
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        svg_str = d2d.GetDrawingText()
        
        # Convert to base64 for HTML embedding
        buffered = BytesIO()
        buffered.write(str.encode(svg_str))
        img_str = base64.b64encode(buffered.getvalue())
        img_str = f"data:image/svg+xml;base64,{repr(img_str)[2:-1]}"
        
        return f"<img src='{img_str}' style='width: {svg_size}px; height: {svg_size}px;' title='{smiles}' />"
    except Exception as e:
        return f"<div style='text-align: center; color: red;'>Error: {str(e)}</div>"


def spectrum_to_html_img(spec1, spec2, img_size=1500):
    """
    Convert spectrum plot to HTML image string for display in Gradio dataframe
    """
    try:
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create the plot using the existing function
        su.plot_spectrum(spec=spec1, mirror_spec=spec2, figsize=(8, 4))
        
        # Save the current figure to a buffer
        buffered = BytesIO()
        plt.savefig(buffered, format='png', bbox_inches='tight', dpi=100)
        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue())
        img_str = f"data:image/png;base64,{repr(img_str)[2:-1]}"
        
        # Close the figure to free memory
        plt.close()
        
        return f"<img src='{img_str}' style='width: {img_size}px; height: auto;' title='Spectrum comparison' />"
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

    # Run simple example as a test and to download weights
    example_url = 'https://raw.githubusercontent.com/pluskal-lab/DreaMS/cc806fa6fea281c1e57dd81fc512f71de9290017/data/examples/example_5_spectra.mgf'
    example_path = Path('./data/example_5_spectra.mgf')
    example_path.parent.mkdir(parents=True, exist_ok=True)
    if not example_path.exists():
        urllib.request.urlretrieve(example_url, example_path)
    embs = dreams_embeddings(example_path)
    print("Setup complete")


def predict(lib_pth, in_pth):
    in_pth = Path(in_pth)
    # # in_pth = Path('DreaMS/data/MSV000086206/peak/mzml/S_N1.mzML')  # Example dataset
    
    msdata_lib = MSData.load(lib_pth)
    embs_lib = msdata_lib[DREAMS_EMBEDDING]
    print('Shape of the library embeddings:', embs_lib.shape)

    msdata = MSData.load(in_pth)
    embs = dreams_embeddings(msdata)
    print('Shape of the query embeddings:', embs.shape)

    sims = cosine_similarity(embs, embs_lib)
    print('Shape of the similarity matrix:', sims.shape)

    k = 5
    topk_cands = np.argsort(sims, axis=1)[:, -k:][:, ::-1]
    topk_cands.shape

    # Construct a DataFrame with the top-k candidates for each spectrum and their corresponding similarities
    df = []
    cos_sim = su.PeakListModifiedCosine()
    for i, topk in enumerate(tqdm(topk_cands)):
        for n, j in enumerate(topk):
            smiles = msdata_lib.get_smiles(j)
            spec1 = msdata.get_spectra(i)
            spec2 = msdata_lib.get_spectra(j)
            df.append({
                'feature_id': i + 1,
                'topk': n + 1,
                'library_j': j,
                'library_SMILES': smiles_to_html_img(smiles),
                'Spectrum': spectrum_to_html_img(spec1, spec2),
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
            })
    df = pd.DataFrame(df)

    # # TODO Add some (random) name to the output file
    df_path = io.append_to_stem(in_pth, 'MassSpecGym_hits').with_suffix('.csv')
    df.to_csv(df_path, index=False)

    # i = df_top1['i'].iloc[25]
    # df_i = df[df['i'] == i]
    # for _, row in df_i.iterrows():
    #     i, j = row['i'], row['j']
    #     print(f'Library ID: {row["library_ID"]} (top {row["topk"]} hit)')
    #     print(f'Query precursor m/z: {msdata.get_prec_mzs(i)}, Library precursor m/z: {msdata_lib.get_prec_mzs(j)}')
    #     print('DreaMS similarity:', row['DreaMS_similarity'])
    #     print('Modified cosine similarity:', row['Modified_cosine_similarity'])
    #     su.plot_spectrum(spec=msdata.get_spectra(i), mirror_spec=msdata_lib.get_spectra(j))
    #     display(Chem.MolFromSmiles(row['library_SMILES']))

    # Sort hits by DreaMS similarity
    df_top1 = df[df['topk'] == 1].sort_values('DreaMS_similarity', ascending=False)
    df = df.set_index('feature_id').loc[df_top1['feature_id'].values].reset_index()

    return df, str(df_path)


setup()
app = gr.Blocks(theme=gr.themes.Default(primary_hue="yellow", secondary_hue="pink"))
with app:

    # Input GUI
    # gr.Markdown(value="""# DreaMS""")
    gr.Image("https://raw.githubusercontent.com/pluskal-lab/DreaMS/cc806fa6fea281c1e57dd81fc512f71de9290017/assets/dreams_background.png", label="DreaMS")
    gr.Markdown(value="""
        DreaMS (Deep Representations Empowering the Annotation of Mass Spectra) is a transformer-based
         neural network designed to interpret tandem mass spectrometry (MS/MS) data. Pre-trained in a
         self-supervised way on millions of unannotated spectra from our new GeMS (GNPS Experimental
         Mass Spectra) dataset, DreaMS acquires rich molecular representations by predicting masked
         spectral peaks and chromatographic retention orders. When fine-tuned for tasks such as spectral
         similarity, chemical properties prediction, and fluorine detection, DreaMS achieves state-of-the-art
         performance across various mass spectrometry interpretation tasks (<a href="https://www.nature.com/articles/s41587-025-02663-3">Bushuiev et al., Nature Biotechnology, 2025</a>).
    """)
    with gr.Row(equal_height=True):
        in_pth = gr.File(
            file_count="single",
            label=".mzML file (TODO Extend to other formats)",
        )
    lib_pth = Path('DreaMS/data/MassSpecGym_DreaMS.hdf5')  # MassSpecGym library
    examples = gr.Examples(
        examples=["./data/S_N1.mzML", "./data/example_5_spectra.mgf"],
        inputs=[in_pth],
        label="Examples (click on a line to pre-fill the inputs)",
        # TODO
        # cache_examples=True
        # outputs=[df, df_file],
        # fn=predict,
    )

    # Predict GUI
    predict_button = gr.Button(value="Run library matching", variant="primary")

    # Output GUI
    gr.Markdown("## Predictions")
    df_file = gr.File(label="Download predictions as .csv", interactive=False, visible=True)
    df = gr.Dataframe(
        headers=["feature_id", "topk", "library_j", "library_SMILES", "Spectrum", "library_ID", "DreaMS_similarity", "Modified_cosine_similarity", "i", "j"],
        datatype=["number", "number", "number", "html", "html", "str", "number", "number", "number", "number"],
        col_count=(10, "fixed"),
        wrap=True,
        column_widths=["80px", "60px", "80px", "400px", "800px", "120px", "120px", "150px", "60px", "60px"],
        max_height=1000,
        show_fullscreen_button=True,
        show_row_numbers=True,
        show_search='filter',
        # pinned_columns=  # TODO
    )

    # Main logic
    inputs = [in_pth]
    outputs = [df, df_file]
    predict = partial(predict, lib_pth)
    predict_button.click(predict, inputs=inputs, outputs=outputs)


app.launch(allowed_paths=['./assets'])
