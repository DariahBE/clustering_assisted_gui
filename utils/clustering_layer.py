import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output
import plotly.express as px
import hdbscan
from sklearn.cluster import KMeans, OPTICS

# when you add a new configartion to the JSON file, it needs to be implemented here as well.
#   The key in CLUSTERING_CLASSES should match the value for class in your config file. Don't forget
#   to also import the necessary modules for the clusering to work as intended. When extending this
#   with new algorithms, they should either be available in current moduleset, or new modules
#   should be added.
CLUSTERING_CLASSES = {
    "HDBSCAN": hdbscan.HDBSCAN,
    "KMeans": KMeans,
    "OPTICS": OPTICS
}


class Clustermachine:
    """
    Clustering GUI.
    Tab driven even switching ==> no hard errors when user interacts wiht the tool
    before actual data is available in the notebook.
    """

    def __init__(self, reduced_vectors, config_path: str, on_result=None, stringgetter = None):
        self.reducedgetter = reduced_vectors
        self.config = self._load_config(config_path)
        self.on_result = on_result
        self.originalstringgetter = stringgetter

        self.X = None
        self.df = None
        self.labels = None
        self.stringlabels = None
        self.fig = None

        self._suspend_axis_observers = False

        self._build_widgets()
        self._build_tabs()

    def _load_config(self, path):
        """
            Reads the config file  ==> performed at initializaton of the class, required
            to make all the proper UI elements.
        """
        with open(path, "r") as f:
            return json.load(f)

    def _build_widgets(self):
        """
            Build widgets safely without assuming reduced data exists.
            Axis controls are created lazily once valid data is available.
        """
        self.method = widgets.Dropdown(
            options=[(v["label"], k) for k, v in self.config.items()],
            description="Clustering:",
            style={"description_width": "initial"}
        )

        self.param_box = widgets.VBox()

        self.axis_box = widgets.HBox()
        self.axis_box.layout.display = "none"

        self.show_noise = widgets.Checkbox(
            value=True,
            description="Show noise (-1)"
        )
        self.show_noise.observe(self._on_axis_change, names="value")

        self.run_button = widgets.Button(
            description="Run clustering",
            button_style="success"
        )

        self.spinner = widgets.HTML("")
        self.output = widgets.Output()

        self.method.observe(self._update_params, names="value")
        self.run_button.on_click(self._run)

        self._update_params()

    def _make_widget(self, spec):
        """
            Helper function to generate the right kind of UI widget in the Notebook 
            interface based on type-key set for the algorithm in the provided JSON file.

            > when adding a new type of widget, it should be defined here HOW to render it.
        """
        t = spec["type"]

        if t == "int_slider":
            return widgets.IntSlider(
                value=spec["default"],
                min=spec["min"],
                max=spec["max"],
                description=spec["label"]
            )
        if t == "float_slider":
            return widgets.FloatSlider(
                value=spec["default"],
                min=spec["min"],
                max=spec["max"],
                step=spec.get("step", 0.01),
                description=spec["label"]
            )
        if t == "dropdown":
            return widgets.Dropdown(
                options=spec["options"],
                value=spec["default"],
                description=spec["label"]
            )
        if t == "int_text":
            return widgets.IntText(
                value=spec["default"],
                description=spec["label"]
            )

        return widgets.Label(f"Unknown param type: {t}")

    def _update_params(self, *args):
        """
            helper required to deal with hyperparameter changes on user inteactivity. 
            Also required for the _run() method to actually provide a proper kv dict 
            with hyperparameter values for the chosen algorithm with set hyperparametrs.
        """
        cfg = self.config[self.method.value]
        self.param_widgets = {}
        self.param_box.children = []

        for name, spec in cfg["params"].items():
            w = self._make_widget(spec)
            self.param_widgets[name] = w
            self.param_box.children += (w,)

    def _build_tabs(self):
        """
            Add tab funtionality: changing tabs triggers refetching 
            the data and renders a new plot.
        """
        self.cluster_tab = widgets.VBox([
            self.method,
            self.param_box,
            self.axis_box,
            self.run_button,
            self.spinner,
            self.output
        ])

        self.diagnostic_output = widgets.Output()
        self.diagnostic_tab = widgets.VBox([self.diagnostic_output])

        self.tabs = widgets.Tab(children=[
            self.cluster_tab,
            self.diagnostic_tab
        ])

        self.tabs.set_title(0, "Clustering")
        self.tabs.set_title(1, "Diagnostics")

        self.tabs.observe(self._on_tab_change, names="selected_index")

    def _fetch_data(self):
        """
            Use the callables given in INIT to fetch the data at any point that
            is necessary in the lifetime of the instance. If types don't match
            or an error is triggered, a soft error is shown in the UIbased on 
            None return
        """
        try:
            X = self.reducedgetter()
            if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
                return None
            return X
        except Exception:
            return None

    def _set_busy(self, state):
        """
            avoid double click in the UI ==> disable button and update text
            to reflect running task.
        """
        self.run_button.disabled = state
        self.run_button.description = "Running..." if state else "Run clustering"
        self.spinner.value = "Clustering..." if state else ""

    def _run(self, *args):
        self._set_busy(True)

        with self.output:
            clear_output(wait=True)
            X = self._fetch_data()
            if X is None:    #soft error: 
                print("No reduced vectors found. Please run dimensionality reduction first.")
                self._set_busy(False)
                return

        self.X = X
        self.df = pd.DataFrame(self.X)
        #rebuilding the axes is needed at every run to catch cases where user goes back in the workflow and changes dimensionatlity 
        self._build_axis_widgets()

        cfg = self.config[self.method.value]
        cls = CLUSTERING_CLASSES[cfg["class"]]
        kwargs = {k: w.value for k, w in self.param_widgets.items()}

        model = cls(**kwargs)

        start = time.time()
        self.labels = model.fit_predict(self.X)
        duration = time.time() - start

        self._render_plot()

        with self.output:
            print(f"Clusters: {len(set(self.labels))}")
            print(f"Time: {duration:.2f}s")

        self._set_busy(False)

        if callable(self.on_result):
            self.on_result(self.labels)

    def _build_axis_widgets(self):
        """Craetes the dropdown for users to visually inspect the clustering output
        after clustering has ran by picking 3 dimensions. 
        ==> recomputed on each run to catch dimensionality changes in the worfklow. 
        """
        cols = list(self.df.columns)

        self.x_axis = widgets.Dropdown(description="X:", options=cols)
        self.y_axis = widgets.Dropdown(description="Y:", options=cols)
        self.z_axis = widgets.Dropdown(description="Z:", options=[None] + cols)

        for w in (self.x_axis, self.y_axis, self.z_axis):
            w.observe(self._on_axis_change, names="value")

        self.axis_box.children = [
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.show_noise
        ]
        self.axis_box.layout.display = "flex"

        self._initialize_axes()

    def _initialize_axes(self):
        """make axis selection widgets from the instance DataFrame.
        """
        cols = list(self.df.columns)
        self._suspend_axis_observers = True
        try:
            self.x_axis.value = cols[0]
            self.y_axis.value = cols[1] if len(cols) > 1 else cols[0]
            self.z_axis.value = cols[2] if len(cols) > 2 else None
        finally:
            self._suspend_axis_observers = False

    def _on_axis_change(self, *args):
        """check if user wants to display another dimension ==> if so do a rerender of the scatterplot."""
        if self._suspend_axis_observers:
            return
        if self.df is None or self.labels is None:
            return
        self._render_plot()

    def _render_plot(self):
        """generates scatterplot in 2D or 3D with original input labels being added
        to the df for direct visual interaction with the user.
        """
        if self.df is None or self.labels is None:
            return

        with self.output:
            clear_output(wait=True)

            # --- ensure numpy arrays for safe masking ---
            stringlabels = np.asarray(self.originalstringgetter())
            labels = np.asarray(self.labels)

            if not self.show_noise.value:
                mask = labels != -1
            else:
                mask = np.ones(len(labels), dtype=bool)

            plot_df = self.df.loc[mask].copy()
            plot_labels = labels[mask].astype(str)
            plot_names = stringlabels[mask]

            # --- add explicit columns for plotting & hover ---
            plot_df["cluster_id"] = plot_labels
            plot_df["input"] = plot_names

            if self.z_axis.value is None:
                fig = px.scatter(
                    plot_df,
                    x=self.x_axis.value,
                    y=self.y_axis.value,
                    color="cluster_id",
                    title="Clustering diagnostic (2D)",
                    hover_data={
                        "cluster_id": True,
                        "input": True
                    }
                )
            else:
                fig = px.scatter_3d(
                    plot_df,
                    x=self.x_axis.value,
                    y=self.y_axis.value,
                    z=self.z_axis.value,
                    color="cluster_id",
                    title="Clustering diagnostic (3D)",
                    hover_data={
                        "cluster_id": True,
                        "input": True
                    }
                )

            fig.update_traces(marker=dict(size=3))
            fig.show()

            self.fig = fig

    def _render_diagnostics(self):
        """
        Renders some basic plots that show the user how the data looks like after clustering. 
        ==> plots are shown in a separate tab
        """
        with self.diagnostic_output:
            clear_output(wait=True)

            if self.labels is None:
                print("Run clustering first to see diagnostics.")
                return

            labels = np.asarray(self.labels)
            noise_count = np.sum(labels == -1)
            clustered_count = np.sum(labels != -1)

            df_noise = pd.DataFrame({
                "Category": ["Clustered", "Noise"],
                "Count": [clustered_count, noise_count]
            })

            px.bar(df_noise, x="Category", y="Count",
                   title="Noise vs Clustered").show()

            non_noise = labels[labels != -1]
            if len(non_noise) == 0:
                print("No non-noise clusters to analyze.")
                return

            sizes = pd.Series(non_noise).value_counts()
            df_sizes = sizes.reset_index()
            df_sizes.columns = ["cluster", "size"]

            px.histogram(df_sizes, x="size",
                         title="Cluster size distribution").show()
            px.box(df_sizes, y="size",
                   title="Cluster size spread").show()

    def _on_tab_change(self, change):
        """Event listener ==> when a change happens, rerender the plots."""
        if change["new"] == 1:
            self._render_diagnostics()

    def display(self):
        """Make the UI in the notebook."""
        display(self.tabs)