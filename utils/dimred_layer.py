import json
import time
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

import umap
import plotly.express as px


# when you add a new configartion to the JSON file, it needs to be implemented here as well.
#   The key in ALGO_CLASSES should match the value for class in your config file. Don't forget
#   to also import the necessary modules for the dimred to work as intended. When extending this
#   with new algorithms, they should either be available in current moduleset, or new modules
#   should be added.

ALGO_CLASSES = {
    "TruncatedSVD": TruncatedSVD,
    "PCA": PCA,
    "IncrementalPCA": IncrementalPCA,
    "KernelPCA": KernelPCA,
    "Isomap": Isomap,
    "UMAP": umap.UMAP,
    "GaussianRandomProjection": GaussianRandomProjection,
    "SparseRandomProjection": SparseRandomProjection
}


# GUI for dimredtool:
class DimensionalityReductionGUI:
    """
    GUI-driven dimensionality reduction layer.
    Algorithms & hyperparameters are defined via JSON.
    """

    def __init__(self, embeddings, config_path, on_result = None):
        """
            sets up the class to perform dimensionality reduction on the given vectors extracted from strings using transformer models. 
            embeddings ==> getter-function(): A callable getter function that returns the embeddings as a numpy-array from the NB into the class instance. 
            config_path: STR ==> path holding all configurations possible for dimensionality reduction. Be careful when adding dimred algos;
                they need to make sense for the task at hand!! 
        """
        self.embeddingsgetter = embeddings
        self.config = self._load_config(config_path)
        self.Z = None
        self.on_result = on_result

        self._build_widgets()

    def _load_config(self, path):
        """
            Reads the config file  ==> performed at initializaton of the class, required
            to make all the proper UI elements.
        """
        with open(path, "r") as f:
            return json.load(f)

    def _build_widgets(self):
        """
        Makes the UI for the user to interact with. 
        """
        ## add a placeholder for 'nothing selected'
        model_options = [("-", "-")] + [
            (v.get("label") or k, k)
            for k, v in self.config.items()
        ]

        self.method_dropdown = widgets.Dropdown(
            layout={'width': 'initial'},
            options=model_options,
            description="Dimensionality reduction method:",
            style={"description_width": "initial"}
        )

        self.target_dims = widgets.IntSlider(
            value=30,
            min=2,
            max=100,
            step=1,
            description="Output dimensions:",
            style={"description_width": "initial"}
        )

        self.param_box = widgets.VBox()

        self.spinner = widgets.HTML("")
        self.busy = False

        self.run_button = widgets.Button(
            description="Run dimensionality reduction",
            button_style="success"
        )

        self.output = widgets.Output()

        self.method_dropdown.observe(self._update_param_widgets, names="value")
        self.run_button.on_click(self._run)

        self._update_param_widgets()

    def _make_widget(self, spec):
        """ 
        Dynamically makes GUI elements one by one based on JSON config for the given method.
        Supported widget types: 
            int_slider
            float_slider
            dropdown
            int_text
        """

        t = spec["type"]

        if t == "int_slider":
            return widgets.IntSlider(
                value=spec["default"],
                min=spec["min"],
                max=spec["max"],
                description=spec["label"],
                tooltip=spec.get("tooltip", "")
            )

        if t == "float_slider":
            return widgets.FloatSlider(
                value=spec["default"],
                min=spec["min"],
                max=spec["max"],
                step=spec.get("step", 0.01),
                description=spec["label"],
                tooltip=spec.get("tooltip", "")
            )

        if t == "dropdown":
            return widgets.Dropdown(
                options=spec["options"],
                value=spec["default"],
                description=spec["label"],
                tooltip=spec.get("tooltip", "")
            )

        if t == "int_text":
            return widgets.IntText(
                value=spec["default"],
                description=spec["label"],
                tooltip=spec.get("tooltip", "")
            )

        raise ValueError(f"Unknown widget type: {t}")

    def _update_param_widgets(self, *args):
        """
            redraws layout with new options depending on chosen main-method.
        """
        if self.method_dropdown.value == '-':
            with self.output:
                clear_output(wait=False)
            return None

        method_key = self.method_dropdown.value
        params = self.config[method_key]["params"]

        self.param_widgets = {}
        widgets_list = []

        for name, spec in params.items():
            w = self._make_widget(spec)
            self.param_widgets[name] = w
            widgets_list.append(w)

        self.param_box.children = widgets_list

    def _set_busy(self, state: bool):
        """
        Update the UI to show a busy-signal, disables the run-button to prevent double clicks. 
        """
        self.busy = state
        self.run_button.disabled = state
        self.run_button.description = "Running..." if state else "Run dimred"
        self.spinner.value = "Running dimensionality reduction..." if state else ""

    def _run(self, *args):
        """Perform dimred and block UI events"""

        if self.method_dropdown.value == '-':
            with self.output:
                clear_output(wait=False)
                print("No valid dimensionality reduction algorithm selected, select one first.")
            return None

        self.X = self.embeddingsgetter()

        with self.output:
            clear_output(wait=True)
            if not isinstance(self.X, np.ndarray):
                print("No usable vectors found in runtime, provide proper embeddings.")
                return

        self._set_busy(True)

        with self.output:
            clear_output(wait=True)
            method_key = self.method_dropdown.value
            cfg = self.config[method_key]
            algo_class = ALGO_CLASSES[cfg["class"]]

            kwargs = {
                name: widget.value
                for name, widget in self.param_widgets.items()
            }

            kwargs["n_components"] = self.target_dims.value

            model = algo_class(**kwargs)
            start = time.time()
            self.Z = model.fit_transform(self.X)
            duration = time.time() - start

            self._visualize()
            print(f"Output shape: {self.Z.shape}")
            print(f"Execution time: {duration:.2f}s")

        self._set_busy(False)
        if callable(self.on_result):
            self.on_result(self.Z)

    def _visualize(self):
        """
            make a 3D or 2D visualisation of the dimred outcome. 
        """
        if self.Z.shape[1] >= 2:
            d = min(3, self.Z.shape[1])

            if d == 2:
                px.scatter(
                    x=self.Z[:, 0],
                    y=self.Z[:, 1],
                    title="Diagnostic view (2D)"
                ).update_traces(marker=dict(size=3)).show()

            elif d == 3:
                px.scatter_3d(
                    x=self.Z[:, 0],
                    y=self.Z[:, 1],
                    z=self.Z[:, 2],
                    title="Diagnostic view (3D)"
                ).update_traces(marker=dict(size=3)).show()

    def display(self):
        """
            Render UI components to user in NB.
        """
        display(
            widgets.VBox([
                self.method_dropdown,
                self.target_dims,
                self.param_box,
                self.run_button,
                self.spinner,
                self.output
            ])
        )