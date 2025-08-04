import gradio as gr
from functools import partial

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
import dreams.utils.spectra as su
import dreams.utils.io as io
from dreams.utils.spectra import PeakListModifiedCosine
from dreams.utils.data import MSData
from dreams.api import dreams_embeddings
from dreams.definitions import *


def predict(lib_pth, in_pth):
    # in_pth = Path('DreaMS/data/MSV000086206/peak/mzml/S_N1.mzML')  # Example dataset
    
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
            df.append({
                'feature_id': i + 1,
                'topk': n + 1,
                'library_j': j,
                'library_SMILES': msdata_lib.get_smiles(j),
                'library_ID': msdata_lib.get_values('IDENTIFIER', j),
                'DreaMS_similarity': sims[i, j],
                'Modified_cosine_similarity': cos_sim(
                    spec1=msdata.get_spectra(i),
                    prec_mz1=msdata.get_prec_mzs(i),
                    spec2=msdata_lib.get_spectra(j),
                    prec_mz2=msdata_lib.get_prec_mzs(j),
                ),
                'i': i,
                'j': j,
            })
    df = pd.DataFrame(df)

    # TODO Add some (random) name to the output file
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
    # df_top1 = df[df['topk'] == 1].sort_values('DreaMS_similarity', ascending=False)
    # df = df.set_index('feature_id').loc[df_top1['feature_id'].values].reset_index()
    # df

    return df, str(df_path)


app = gr.Blocks(theme=gr.themes.Default(primary_hue="green", secondary_hue="pink"))
with app:

    # Input GUI
    gr.Markdown(value="""
        # DreaMS
    """)
    # gr.Image("assets/readme-dimer-close-up.png")
    # gr.Markdown(value="""
    #     TODO Some description
    # """)
    with gr.Row(equal_height=True):
        in_pth = gr.File(
            file_count="single",
            label=".mzML file (TODO Extend to other formats)"
        )
    lib_pth = Path('DreaMS/data/MassSpecGym_DreaMS.hdf5')  # MassSpecGym library

    # Predict GUI
    predict_button = gr.Button(value="Run library matching", variant="primary")

    # Output GUI
    gr.Markdown("## Predictions")
    df_file = gr.File(label="Download predictions as .csv", interactive=False, visible=True)
    df = gr.Dataframe(
        headers=["feature_id", "topk", "library_j", "library_SMILES", "library_ID", "DreaMS_similarity", "Modified_cosine_similarity", "i", "j"],
        datatype=["number", "number", "number", "str", "str", "number", "number", "number", "number"],
        col_count=(9, "fixed"),
    )
    # dropdown = gr.Dropdown(interactive=True, visible=False)
    # dropdown_choices_to_plot_args = gr.State([])
    # plot = gr.HTML()

    # Main logic
    inputs = [in_pth]
    outputs = [df, df_file]
    predict = partial(predict, lib_pth)
    predict_button.click(predict, inputs=inputs, outputs=outputs)

    # Update plot on dropdown change
    # dropdown.change(update_plot, inputs=[dropdown, dropdown_choices_to_plot_args], outputs=[plot])


app.launch(allowed_paths=['./assets'])
