import json
import time
import codecs
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from sentence_transformers import SentenceTransformer


class TransformerGUI:
    """
    Vectorization GUI layer.
    """

    def __init__(self, config_path, device="cpu", batch_size=256, on_result = None):
        """
            config_path (STR): Path to JSON file holding configuration for embeddings. 
            device (STR): cpu or cuda ==> which device to use
            batch_size (INT): lower batches on lower-tier GPUs
            one_result == callback function that exposes the embeddings back to the notebook without globals!
        """
        self.device = device
        self.batch_size = batch_size

        self.config = self._load_config(config_path)
        self.models = self.config["models"]
        self.file_contents = []

        self.raw_names = []
        self.names = []
        self.quote_setting = True
        self.first_row_setting = True
        self.on_result = on_result

        self.selected_model_key = "-"

        self.instructor_prompt = "" #only required for embeddings models that have an instructor component!

        self._build_widgets()

    def _load_config(self, path):
        """
            Reads the config file  ==> performed at initializaton of the class, required
            to make all the proper UI elements.
        """
        with open(path, "r") as f:
            return json.load(f)

    def _parse_names(self):
        """
            Processes the given input file according to settings active in the DOM. 
        """
        if self.first_row_setting:
            self.raw_names = self.file_contents[1:]
        else:
            self.raw_names = self.file_contents[:]

        if self.quote_setting:
            self.names = [
                x.strip().strip('"').strip("'")
                for x in self.raw_names
                if x.strip()
            ]
        else:
            self.names = [x.strip() for x in self.raw_names if x.strip()]

    def _on_file_upload(self, change):
        """
        event listner for uploading of file. 
        """
        upload = change.new
        if not upload:
            return

        content = upload[0].content
        stream = codecs.decode(content, encoding="utf-8")
        stream = stream.replace("\x0b", "")

        self.file_contents = list(stream.splitlines())
        self._parse_names()

        with self.output:
            print(f"Loaded {len(self.names)} rows")

    def _on_quote_toggle(self, change):
        """
        event listener for quote toggle. 
        """
        self.quote_setting = change.new
        self._parse_names()

    def _on_drop_first_row_toggle(self, change):
        """
        event listener for fileheader toggle. 
        """
        self.first_row_setting = change.new
        self._parse_names()
    
    def _on_batch_change(self, change): 
        """
        event listener for batch changes. 
        """
        self.batch_size = change.new

    def _is_instructor_model(self, model_key):
        """
        Checks if a model needs an instruction component (works with both the pre-defined models as well as a custom
        chosen model where the name contains 'instruct' [instruction/instructor...]   )
        This check is needed to have it included in the _run phase as well as updating the DOM when a custom
        model name is entered. Unwritten rules indicate that models with an isntructor-componenent are either named
        '%instruction%' or '%instructor%' ==> if we look for the substring '%instruct%' this method should capture 
        those cases and provide adequate support for DOM-updates and _prepare_texts modifications. 
        """
        if model_key.lower() == 'custom':
            return 'instruct' in self.custom_model.value.lower()

        if model_key in self.models:
            return (
                'instructor' in model_key.lower()
                or 'instructor' in self.models[model_key].get('label', '').lower()
            )

        return False

    def _prepare_texts(self):
        """
        Returns the texts passed to model.encode().
        DOES NOT modify self.names.
        """
        if self._is_instructor_model(self.selected_model_key):
            prompt = self.instructor_prompt.strip()
            if not prompt:
                raise ValueError("Instructor model requires a prompt.")
            return [f"{prompt} ||| {x}" for x in self.names]

        return self.names

    def _compute_embeddings(self, model_name):
        """
        Performs embedding calcuations on the given dataset.
        """
        model = SentenceTransformer(model_name, device=self.device)

        texts = self._prepare_texts()

        return model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def _run_vectorization(self, button):
        """
        Event listener for when the user clicks the button to start vector extractions: 
        1) checks settings
        2) starts extraction 
        3) reports extraction time 
        ==> embedded vactors are kept in memory as an instance proeprty. 
        """
        with self.output:
            clear_output(wait=True)

            if not self.names:
                print("Upload a data file first.")
                return

            if self.selected_model_key == "-":
                print("Select a vectorizer.")
                return

            if self.selected_model_key == "Custom":
                model_name = self.custom_model.value.strip()
                if not model_name:
                    print("Enter a custom HuggingFace model name.")
                    return
            else:
                model_name = self.models[self.selected_model_key]["name"]

        button.disabled = True
        original_label = button.description
        button.description = "Computing embeddings..."

        try:
            t0 = time.time()
            self.embeddings = self._compute_embeddings(model_name)

            with self.output:
                print(f"Embeddings shape: {self.embeddings.shape}")
                print(f"Time taken: {time.time() - t0:.1f}s")

        except Exception as e:
            with self.output:
                print(f"Error: {e}")

        finally:
            button.description = original_label
            button.disabled = False

        if callable(self.on_result):
            self.on_result(self.embeddings, self.names)

    def _build_widgets(self):
        """
        Makes the UI for the user to interact with. 
        """

        self.file_upload = widgets.FileUpload(
            accept=".csv,.txt,.tsv",
            multiple=False,
            description="Upload data"
        )
        self.file_upload.observe(self._on_file_upload, names="value")

        self.strip_quotes = widgets.Checkbox(
            value=self.quote_setting,
            description="Strip quotes"
        )
        self.strip_quotes.observe(self._on_quote_toggle, names="value")

        self.drop_first_row = widgets.Checkbox(
            value=self.first_row_setting,
            description="Drop first row (header)"
        )
        self.drop_first_row.observe(self._on_drop_first_row_toggle, names="value")

        self.batch_size_input = widgets.IntText(
            value = self.batch_size, 
            description = "Size of batch to process", 
            allow_none = False
        )
        self.batch_size_input.observe(self._on_batch_change, names="value")

        self.file_box = widgets.HBox([
            self.file_upload,
            self.strip_quotes,
            self.drop_first_row, 
            self.batch_size_input
        ])

        model_options = [("-", "-")] + [
            (v.get("label") or k, k)
            for k, v in self.models.items()
        ] + [("Custom", "Custom")]

        self.vectorizer = widgets.Dropdown(
            options=model_options,
            value="-",
            layout=widgets.Layout(width="50%")
        )

        self.custom_model = widgets.Text(
            placeholder="HuggingFace model name",
            disabled=True,
            layout=widgets.Layout(width="95%")
        )

        self.prompt_box = widgets.Textarea(
            placeholder="e.g. Represent the entity name for semantic clustering",
            description="Instructor prompt:",
            layout=widgets.Layout(width="95%"),
        )
        self.prompt_box.observe(
            lambda c: setattr(self, "instructor_prompt", c.new),
            names="value"
        )

        self.prompt_container = widgets.VBox([self.prompt_box])
        self.prompt_container.layout.display = "none"

        def on_vectorizer_change(change):
            if change.owner is self.custom_model:
                model_key = "Custom"
            else:
                model_key = change.new
                self.selected_model_key = model_key

            if model_key == "-":
                return

            is_custom = model_key == "Custom"
            self.custom_model.disabled = not is_custom
            self.custom_box.layout.display = "flex" if is_custom else "none"

            is_instructor = self._is_instructor_model(model_key)
            self.prompt_container.layout.display = "flex" if is_instructor else "none"

        self.vectorizer.observe(on_vectorizer_change, names="value")
        self.custom_model.observe(on_vectorizer_change, names="value")

        self.run_button = widgets.Button(
            description="Get embeddings",
            button_style="success"
        )
        self.run_button.on_click(self._run_vectorization)

        self.output = widgets.Output()

        self.custom_box = widgets.HBox(
            [widgets.Label("Custom model:"), self.custom_model]
        )
        self.custom_box.layout.display = "none"

        self.ui = widgets.VBox([
            self.file_box,
            widgets.HBox([
                widgets.Label("Vector model:"),
                self.vectorizer,
                self.run_button
            ]),
            self.custom_box,
            self.prompt_container,
            self.output
        ])

    def display(self):
        """call method from NB to render the UI"""
        display(self.ui)

    def get_embeddings(self):
        """call method from NB to retrieve embeddings"""
        return self.embeddings

    def get_names(self):
        """call method from NB to retrieve original strings"""
        return self.names