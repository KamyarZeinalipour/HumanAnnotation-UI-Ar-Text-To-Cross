import os.path
import time

import gradio as gr
import pandas as pd
import numpy as np
import fire
import evaluate
import nltk


def list_dir(folder, **kwargs):
    """
    Utility to retrieve files and/or folders from a given source folder, recursively or not,
    possibly filtered by certain specs
    :param folder: the folder to scan
    :param kwargs: options to filter-out elements of certain type or to keep only elements with given extension
    :return: a list of full paths of all the elements in folder
    """

    dir_only = kwargs.get("dir_only", False)
    files_only = kwargs.get("files_only", False)
    extension_filter = kwargs.get("extension_filter", "")
    assert not (dir_only and files_only and extension_filter), \
        "Arguments dir_only, files_only and extension_filter are mutually exclusive"

    apply_recursively = kwargs.get("apply_recursively", False)

    # scan folder recursively adding all the elements in the folder
    dir_contents = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        dir_contents.append(os.path.join(folder, name))
        if apply_recursively and os.path.isdir(path):
            dir_contents.extend(list_dir(folder=os.path.join(folder, name), **kwargs))

    # keep only the elements compliant to the filters specified
    if dir_only:
        dir_contents = [path for path in dir_contents if os.path.isdir(path)]
    elif files_only:
        dir_contents = [path for path in dir_contents if os.path.isfile(path)]
    elif extension_filter:
        dir_contents = [path for path in dir_contents if path.endswith(extension_filter)]

    return dir_contents


def get_start_index(anns_filepath, start_index):
    anns_df = pd.read_csv(anns_filepath)
    
    return max([start_index] + anns_df.index.tolist()) + 1


def main(
        current_index: int = 0,
        annotator_name: str = "",
        examples_batch_folder: str = ''
):
    css = """
.txtbox textarea {font-size: 10px !important}
.txtbox-highlight textarea {background-color: rgba(79, 70, 229, 0.5)}
"""

    assert annotator_name, "Annotator name MISSING. Set it when you launch the script"
    assert examples_batch_folder, "Examples' batch MISSING. Set it when you launch the script"


    _, dataset_filename = os.path.split(examples_batch_folder)

    chunk_df = pd.read_csv(examples_batch_folder)

    rouge = evaluate.load("rouge")

    annotations_folder = os.path.join(os.getcwd(), "annotations")
    anns_filepath = os.path.join(annotations_folder, f"annotations_{dataset_filename}.csv")
    print(anns_filepath)
    if os.path.exists(anns_filepath):
        current_index = get_start_index(anns_filepath, current_index)
    else:
        os.makedirs(annotations_folder, exist_ok=True)

    print(f"Resume annotations process from {current_index}")
    df_row = chunk_df.iloc[current_index]  # first example loaded
    print(df_row)
    def get_best_sentence(extract, clue):
        if pd.isna(extract) or pd.isna(clue):
            return None
        wiki_sentences = nltk.sent_tokenize(extract)
        res = np.zeros(len(wiki_sentences))
        for i, snt in enumerate(wiki_sentences):
            res[i] = rouge.compute(references=[snt], predictions=[clue], use_stemmer=True)["rougeL"]

        best_snt = np.argmax(res)

        return wiki_sentences[best_snt]

    def store_annotation_and_get_next(curr_idx, rating, comments, validate_btn):
        # store annotation
        if os.path.exists(anns_filepath):
            anns_df = pd.read_csv(anns_filepath)
        else:
            cols = chunk_df.columns.tolist()
            cols.append("timestamp")
            cols.append("rating")
            cols.append("comments")  # Add comments column
            cols.append("annotator")
            anns_df = pd.DataFrame(columns=cols)

        row = chunk_df.iloc[curr_idx].to_dict()
        row["timestamp"] = time.time()
        row["rating"] = rating
        if not comments.strip():  # If comments field is empty, use clue column
            row["comments"] = row["clue"]
        else:
            row["comments"] = comments  # Save comments
        row["annotator"] = annotator_name
        anns_df = pd.concat((anns_df, pd.DataFrame(row, index=[0])), ignore_index=True)
        anns_df.to_csv(anns_filepath,encoding='utf-8-sig' ,index=False)

        # Clear comments field after storing annotation
        comments = '' 

        # get next
        next_idx = curr_idx + 1
        next_df_row = chunk_df.iloc[next_idx]

        return [next_idx, next_df_row.extract, get_best_sentence(next_df_row.extract, next_df_row.clue),
                next_df_row.answer, next_df_row.new_category, next_df_row.clue, "",""]


    with gr.Blocks(theme=gr.themes.Soft(text_size=gr.themes.sizes.text_sm), css=css) as demo:
        index = gr.Number(value=current_index, visible=False, precision=0)

        gr.Markdown(f"#### Annotating: {dataset_filename}\n")
        print(chunk_df.columns)
        print(chunk_df.head())

        with gr.Row():
            with gr.Column():
                wiki_text = gr.Textbox(label="EXTRACT", interactive=False, max_lines=7, value=df_row.extract,
                                        elem_classes="txtbox")
                snt_highlight = gr.Textbox(
                    label="MOST RELATED SENTENCE", show_label=True, interactive=False, max_lines=2,
                    value=get_best_sentence(df_row.extract, df_row.clue),
                    elem_classes="txtbox-highlight"
                )

            with gr.Column():
                answer = gr.Textbox(value=df_row.answer, label="ANSWER", interactive=False)
                category = gr.Textbox(value=df_row.new_category, label="CATEGORY", interactive=False)
                clue = gr.Textbox(label="CLUE", value=df_row.clue, interactive=False)
                comments = gr.Textbox(label="COMMENTS", placeholder="Enter your comments here", type="text")

        with gr.Row():
            rating_radio = gr.Radio(["A", "B", "C", "D", "E", "SKIPPING"], label="Rating")
            with gr.Column():
                gr.Markdown("**Rating-A**: The clue is valid and coherent to the given **context**, **answer** and **category**.")
                gr.Markdown("**Rating-B**: Acceptable clue with minor imperfections - loose correlation with **category**.")
                gr.Markdown("**Rating-C**: The clue is relevant to the **answer** but loosely correlates with the **context**.")
                gr.Markdown("**Rating-D**: The clue is irrelevant and/or incorrect wrt the **answer** or the **context**.")
                gr.Markdown("**Rating-E**: Not acceptable clue, because it contains the **answer** (or a variant of it).")

        eval_btn = gr.Button("Validate", interactive=True)

        rating_radio.select(inputs=[eval_btn], outputs=[eval_btn])

        eval_btn.click(
            store_annotation_and_get_next, inputs=[index, rating_radio, comments, eval_btn],
            outputs=[index, wiki_text, snt_highlight, answer, category, clue, rating_radio,comments]
        )
        """
        wiki_text.script(key_press="if (event.keyCode === 13) {document.querySelector('.component-23').click();}")
        # Adding event listener for "A" key press
        wiki_text.script(key_press="if (event.key === 'a') {document.querySelector('input[value=\"A\"]').click();}")
        # Adding event listener for "s" key press
        wiki_text.script(key_press="if (event.key === 's') {document.querySelector('input[value=\"B\"]').click();}")
        # Adding event listener for "d" key press
        wiki_text.script(key_press="if (event.key === 'd') {document.querySelector('input[value=\"C\"]').click();}")
        # Adding event listener for "A" key press
        wiki_text.script(key_press="if (event.key === 'f') {document.querySelector('input[value=\"D\"]').click();}")
        # Adding event listener for "A" key press
        wiki_text.script(key_press="if (event.key === 'g') {document.querySelector('input[value=\"E\"]').click();}")
        """
    demo.launch()



if __name__ == "__main__":
    fire.Fire(main)
