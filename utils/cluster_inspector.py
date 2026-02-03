import random
import csv
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output


class ClusterInspectorGUI:
    """
    GUI for inspecting and exporting clustered names.
    Fully getter-driven and safe against missing data.
    """

    def __init__(
        self,
        namegetter,
        reducedgetter,
        labelgetter,
        extendedlabelsgetter,
        extendedsourcesgetter
    ):
        for fn, name in [
            (namegetter, "namegetter"),
            (reducedgetter, "reducedgetter"),
            (labelgetter, "labelgetter"),
            (extendedlabelsgetter, "extendedlabelsgetter"),
            (extendedsourcesgetter, "extendedsourcesgetter"),
        ]:
            if not callable(fn):
                raise TypeError(f"{name} must be callable")

        self.name_getter = namegetter
        self.reduced_getter = reducedgetter
        self.label_getter = labelgetter
        self.extended_label_getter = extendedlabelsgetter
        self.extended_source_getter = extendedsourcesgetter

        self._build_widgets()

    def _fetch_dataframe(self):
        """
        Use the getter functions to get the data and make a dataframe 
        with them. ==> using getters to keep on fetching latest generated
        data. 
        There's a check in case the user uploads a new dataset with other
        dimensions than the first dataset to thorw an error..
        """
        try:
            names = self.name_getter()
            labels = self.label_getter()
            ext_labels = self.extended_label_getter()
            ext_sources = self.extended_source_getter()
        except Exception:
            return None

        if names is None or labels is None:
            return None

        if ext_labels is None or ext_sources is None:
            ext_labels = labels
            ext_sources = ["native"] * len(labels)
            self.has_extended = False
        else:
            self.has_extended = True

        if not (len(names) == len(labels) == len(ext_labels)):
            return None

        return pd.DataFrame({
            "name": names,
            "label": labels,
            "extended_label": ext_labels,
            "source": ext_sources
        })

    def _active_label_column(self):
        return "extended_label" if self.label_source.value == "extended" else "label"

    def _build_widgets(self):
        """Make all UI components - data is not coupled to this directly."""
        self.label_source = widgets.Dropdown(
            options=[("Base labels", "base"), ("Extended labels", "extended")],
            description="Use labels:",
            style={"description_width": "initial"}
        )

        self.inspect_button = widgets.Button(
            description="Inspect random cluster",
            button_style="info", 
            layout=widgets.Layout(width="auto"),

        )
        self.inspect_button.on_click(self._inspect_random)

        self.search_box = widgets.Text(
            placeholder="Type a name: ",
            description="Find name: ",
            layout=widgets.Layout(width="60%")
        )

        self.suggestions = widgets.Select(
            options=[],
            rows=5,
            layout=widgets.Layout(width="60%")
        )

        self.search_box.observe(self._update_suggestions, names="value")
        self.suggestions.observe(self._inspect_selected_name, names="value")

        self.filename = widgets.Text(
            value="clusternames.csv",
            description="Filename:"
        )

        self.sep = widgets.Dropdown(
            options=[(",", ","), (";", ";"), ("Tab", "\t")],
            value=",",
            description="Separator:"
        )

        self.line_end = widgets.Dropdown(
            options=[("LF", "\n"), ("CRLF", "\r\n")],
            value="\n",
            description="Line ending:"
        )

        self.quoting = widgets.Dropdown(
            options=[
                ("None", csv.QUOTE_NONE),
                ("Minimal", csv.QUOTE_MINIMAL),
                ("All", csv.QUOTE_ALL)
            ],
            value=csv.QUOTE_MINIMAL,
            description="Quoting:"
        )

        self.export_button = widgets.Button(
            description="Export CSV",
            button_style="success"
        )
        self.export_button.on_click(self._export)

        self.output = widgets.Output()

    def _inspect_random(self, *args):
        """Inspect a random cluster=> take random cluster label and dispaly all shared
        strings for this label. """
        with self.output:
            clear_output(wait=True)

            df = self._fetch_dataframe()
            if df is None:
                print("No clustering data available. Run clustering first.")
                return

            col = self._active_label_column()
            valid_clusters = df[df[col] != -1][col].unique()

            if len(valid_clusters) == 0:
                print("No clusters available.")
                return

            self._render_cluster(df, random.choice(valid_clusters))

    def _update_suggestions(self, change):
        """
        Observe the searchbox where a name can be entered and display partial matches. 
        """
        text = change.new.lower().strip()
        if not text:
            self.suggestions.options = []
            return

        names = self.name_getter()
        if names is None:
            return

        matches = [n for n in names if text in n.lower()][:20]
        self.suggestions.options = matches

    def _inspect_selected_name(self, change):
        """observe the name suggestions box, on picking a name to inspect look up which 
        label is given to the name and display all other names in that cluster."""
        if not change.new:
            return

        with self.output:
            clear_output(wait=True)

            df = self._fetch_dataframe()
            if df is None:
                print("No clustering data available.")
                return

            row = df[df["name"] == change.new]
            if row.empty:
                print("Name not found.")
                return

            col = self._active_label_column()
            self._render_cluster(df, row.iloc[0][col])

    def _render_cluster(self, df, cluster_id):
        """Display all cluster elements for a given cluster_id."""
        col = self._active_label_column()
        subset = df[df[col] == cluster_id]

        display(widgets.HTML(
            f"<h4>Cluster {cluster_id} ({len(subset)} items)</h4>"
        ))
        display(subset[["name", "source"]])

    def _export(self, *args):
        """Export the cluster result to a new CSV fil with optional settings defined in dropdowns."""
        with self.output:
            clear_output(wait=True)

            df = self._fetch_dataframe()
            if df is None:
                print("No clustering data available to export.")
                return

            cols = ["name", "label"]
            if self.has_extended:
                cols += ["extended_label", "source"]

            df[cols].to_csv(
                self.filename.value,
                sep=self.sep.value,
                index=False,
                lineterminator=self.line_end.value,
                quoting=self.quoting.value
            )

            print(f"Exported to {self.filename.value}")

    def display(self):
        """Make the interface for the cluster inspection tool in the notebook. """
        display(
            widgets.VBox([
                self.label_source,
                self.inspect_button,
                widgets.Label("Search by name:"),
                self.search_box,
                self.suggestions,
                widgets.HTML("<hr>"),
                widgets.Label("Export options"),
                self.filename,
                self.sep,
                self.line_end,
                self.quoting,
                self.export_button,
                self.output
            ])
        )