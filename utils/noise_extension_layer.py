import json
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os


class NoiseExtender:
    """
    Post-processing GUI for extending labels to noise.
    The goal is to have this step performed after the initial clustering step. Once clusters
    have been assigned, the NoiseExtender tries to assign all -1 labels to a non-noise label
    this is done in either of two ways: 
        FAST ==> One single pass over the dataset. 
        SLOW ==> Finds the noise-point closest to a non-noise point and assigns it the non-noise label,
                    then keeps repeating this processs untill all noise-labels are overridden. 
    """

    def __init__(self, reducedgetter, labelgetter, config_path, on_result=None):
        if not callable(reducedgetter):
            raise TypeError("reducedgetter must be callable")
        if not callable(labelgetter):
            raise TypeError("labelgetter must be callable")

        self.reducedgetter = reducedgetter
        self.labelgetter = labelgetter
        self.on_result = on_result

        self.config = self._load_config(config_path)

        self.extended_labels = None
        self.label_source = None

        self._build_widgets()

    def _load_config(self, path):
        """
            Reads the config file  ==> performed at initializaton of the class, required
            to make all the proper UI elements.
        """
        with open(path, "r") as f:
            return json.load(f)

    def _fetch_data(self):
        """
        Safely fetch reduced vectors and labels.
        Returns (X, labels) or None.
        """
        try:
            X = self.reducedgetter()
            labels = self.labelgetter()
        except Exception:
            return None

        if not isinstance(X, np.ndarray):
            return None
        if not isinstance(labels, np.ndarray):
            return None
        if len(X) != len(labels):
            return None

        return X, labels

    # --------------------------------------------------
    # Extension strategies
    # --------------------------------------------------

    def _nearest_iterative(self, X, labels):
        labels = labels.copy()
        source = np.array(["native"] * len(labels), dtype=object)

        noise_idx = np.where(labels == -1)[0]
        non_noise_idx = np.where(labels != -1)[0]

        if len(noise_idx) == 0:
            return labels, source

        n_jobs = os.cpu_count() or 1

        with tqdm(
            total=len(noise_idx),
            desc="Extending noise (iterative)",
            unit="point"
        ) as pbar:
            while np.any(labels == -1):
                noise_idx = np.where(labels == -1)[0]

                nbrs = NearestNeighbors(
                    n_neighbors=1,
                    n_jobs=n_jobs
                ).fit(X[non_noise_idx])

                dists, neigh = nbrs.kneighbors(X[noise_idx])
                pos = np.argmin(dists)

                idx = noise_idx[pos]
                labels[idx] = labels[non_noise_idx[neigh[pos][0]]]
                source[idx] = "extended"

                non_noise_idx = np.append(non_noise_idx, idx)
                pbar.update(1)

        return labels, source

    def _nearest_batch(self, X, labels):
        labels = labels.copy()
        source = np.array(["native"] * len(labels), dtype=object)

        noise_idx = np.where(labels == -1)[0]
        non_noise_idx = np.where(labels != -1)[0]

        if len(noise_idx) == 0:
            return labels, source

        n_jobs = os.cpu_count() or 1

        nbrs = NearestNeighbors(
            n_neighbors=1,
            n_jobs=n_jobs
        ).fit(X[non_noise_idx])

        with tqdm(total=1, desc="Extending noise (batch)") as pbar:
            dists, neigh = nbrs.kneighbors(X[noise_idx])

            for i, idx in enumerate(noise_idx):
                labels[idx] = labels[non_noise_idx[neigh[i][0]]]
                source[idx] = "extended"

            pbar.update(1)

        return labels, source

    def _build_widgets(self):
        """
            Makes the UI for the user to interact with.
        """
        self.method = widgets.Dropdown(
            options=[(v["label"], k) for k, v in self.config.items()],
            description="Extension method:",
            style={"description_width": "initial"}
        )

        self.param_box = widgets.VBox()

        self.run_button = widgets.Button(
            description="Extend noise",
            button_style="warning"
        )
        self.run_button.on_click(self._run)

        self.output = widgets.Output()

    def _run(self, *args):
        """
            Performs the extension of labels to noise points based on chosen strategy.
        """
        with self.output:
            clear_output(wait=True)

            data = self._fetch_data()
            if data is None:
                print("Required data not available. Please run dimensionality reduction and clustering first.")
                return

            X, labels = data
            method = self.method.value

            print(f"Starting noise extension using '{method}'...")

            if method == "nearest_iterative":
                result = self._nearest_iterative(X, labels)

            elif method == "nearest_batch":
                result = self._nearest_batch(X, labels)

            else:
                print(f"Unknown extension method: {method}")
                return

            self.extended_labels, self.label_source = result

            print("Noise extension complete.")
            print(f"Native labels: {(self.label_source == 'native').sum()}")
            print(f"Extended labels: {(self.label_source == 'extended').sum()}")

            if callable(self.on_result):
                self.on_result(self.extended_labels, self.label_source)

    def display(self):
        display(
            widgets.VBox([
                self.method,
                self.param_box,
                self.run_button,
                self.output
            ])
        )
