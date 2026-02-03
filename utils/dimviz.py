from __future__ import annotations

import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display


class DimensionVisualizer:
    """
    Visualizes per-dimension variability for a dataset fetched live via a datafetcher.
    Diagnostic tool assuming samples are already meaningfully ordered (e.g. clustered).
    Best of all to keep normalization on; this will center each dimension on the mean, will
    help to avoid scaling issues!
    """

    def __init__(self, datafetcher, normalize=True):
        if not callable(datafetcher):
            raise TypeError("datafetcher must be callable")

        self.datafetcher = datafetcher
        self.normalize = normalize

        # UI state
        self.sort_dim = None
        self.sort_dropdown = None

    # -------------------------
    # Data handling
    # -------------------------

    def _try_get_data(self):
        """
        Fetch and validate data from the pipeline.
        Returns a 2D float numpy array or None.
        Applies normalization and sorting if enabled.
        """
        try:
            data = self.datafetcher()
            if (
                data is None
                or not isinstance(data, np.ndarray)
                or data.ndim != 2
                or not np.issubdtype(data.dtype, np.number)
            ):
                return None

            data = data.astype(float, copy=False)
            if self.normalize:
                data = self._normalize(data)

            # apply sorting (descending)
            if self.sort_dim is not None and 0 <= self.sort_dim < data.shape[1]:
                order = np.argsort(data[:, self.sort_dim])[::-1]
                data = data[order]

            return data

        except Exception:
            return None

    @staticmethod
    def _normalize(data):
        """
        provices a way to scale all the data round the mean to avoid
        one wide dimension rendering the values of other dimensions meaningless!
        Maybe consider to take it out of the init-statement!
        """
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (data - mean) / std

    def _update_sort_options(self):
        """
        Refresh available sort dimensions based on current data.
        Called on every visualization update to catch scenarios where the output
        of dimension reduction step changes. 
        """
        data = self._try_get_data()
        if data is None:
            self.sort_dropdown.options = [("No sorting", None)]
            self.sort_dim = None
            return

        n_dims = data.shape[1]
        options = [("No sorting", None)] + [
            (i, i) for i in range(n_dims)
        ]

        self.sort_dropdown.options = options

        # reset values with latest data
        if self.sort_dropdown.value not in [opt[1] for opt in options]:
            self.sort_dropdown.value = None

        self.sort_dim = self.sort_dropdown.value

    def _render_heatmap(self, output):
        """
        Renders the heatmap, when normalization is used, all data is scaled to the same range, making for
        easy to understand plots. When one dimension is sorted, a good dimensionality would show as clustered
        blocks being visible in the data. IT COULD BE THAT SOME OTHER DIMENSIONS DON'T CAPTURE THIS BLOCK-SHAPE
        VERY WELL. THAT'S TO BE EXPECTED. IF THE OVERALL PLOT SHOWS A BLOCKSHAPE, YOU'RE GOOD TO GO!
        """
        with output:
            output.clear_output()
            data = self._try_get_data()
            if data is None:
                print("No data available yet. Please run the previous steps to generate data.")
                return

            fig = go.Figure(
                data=go.Heatmap(
                    z=data.T,
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-3,
                    zmax=3,
                    colorbar=dict(title="Z-score"),
                )
            )

            fig.update_layout(
                title="Dimension Heatmap",
                xaxis_title="Sample index",
                yaxis_title="Dimension",
                height=400,
                template="plotly_white",
            )

            fig.show()

    def _render_lineplots(self, output, cols):
        """
        Plots variability within a single dimension as a lineplot. When sorting on a dimension is applied, 
        it'll render as a descending line. Other dimensions should be grouped in highs-and lows into widths 
        similar to the steps in your sorted-dimension plot. That's an indication the dimensionality reduction
        left enought of the nuance for the clustering to work. 
        It is perfeclty normal that some dimensions show spikes within each block. It cannot be expected that every
        dimension perfectly captures all of the clusters. The overall shape should be visible. 
        """

        with output:
            output.clear_output()
            data = self._try_get_data()
            if data is None:
                print("No data available yet. Please run the previous steps to generate data.")
                return

            _, n_dims = data.shape
            rows = math.ceil(n_dims / cols)

            fig = make_subplots(
                rows=rows,
                cols=cols,
                shared_xaxes=True,
                subplot_titles=[f"Dimension {i}" for i in range(n_dims)],
            )

            for dim in range(n_dims):
                r = dim // cols + 1
                c = dim % cols + 1
                fig.add_trace(
                    go.Scatter(
                        y=data[:, dim],
                        mode="lines",
                        line=dict(width=1.5),
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )

            fig.update_layout(
                height=rows * 220,
                template="plotly_white",
            )

            fig.show()

    def display(self):
        """Render UI in the notebook. """
        heatmap_output = widgets.Output()
        lineplot_output = widgets.Output()

        update_button = widgets.Button(
            description="Update visualization",
            button_style="primary",
            icon="refresh",
            layout=widgets.Layout(width="auto"),
        )

        col_slider = widgets.IntSlider(
            value=2,
            min=1,
            max=6,
            step=1,
            description="Columns:",
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.sort_dropdown = widgets.Dropdown(
            options=[("No sorting", None)],
            description="Sort by dimension (desc):",
            style={"description_width": "initial"},
        )

        heatmap_tab = widgets.VBox([heatmap_output])
        lineplot_tab = widgets.VBox([col_slider, lineplot_output])

        tabs = widgets.Tab(children=[heatmap_tab, lineplot_tab])
        tabs.set_title(0, "Heatmap")
        tabs.set_title(1, "Lineplots")

        def _update_all(_=None):
            self._update_sort_options()
            if tabs.selected_index == 0:
                self._render_heatmap(heatmap_output)
            else:
                self._render_lineplots(lineplot_output, col_slider.value)

        update_button.on_click(_update_all)
        col_slider.observe(lambda _: _update_all(), names="value")
        self.sort_dropdown.observe(lambda _: _update_all(), names="value")
        tabs.observe(lambda c: _update_all() if c["name"] == "selected_index" else None)

        display(
            widgets.VBox([
                widgets.HBox([update_button, self.sort_dropdown]),
                tabs,
            ])
        )