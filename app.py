from pathlib import Path

import duckdb
import holoviews as hv
import pandas as pd
import panel as pn
from bokeh.models import HoverTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI

pn.extension(sizing_mode="stretch_width", notifications=True)
hv.extension("bokeh")


RANDOM_NAME_QUERY = """
    SELECT name, count, 
        CASE
            WHEN female_percent >= 0.2 AND female_percent <= 0.8 AND male_percent >= 0.2 AND male_percent <= 0.8 THEN 'unisex'
            WHEN female_percent > 0.6 THEN 'female'
            WHEN male_percent > 0.6 THEN 'male'
        END AS gender
    FROM (
        SELECT 
            name,
            MAX(male + female) AS count,
            (SUM(female) / CAST(SUM(male + female) AS REAL)) AS female_percent,
            (SUM(male) / CAST(SUM(male + female) AS REAL)) AS male_percent
        FROM names
        WHERE name LIKE ?
        GROUP BY name
    )
    WHERE count >= ? AND count <= ?
    AND gender = ?
    ORDER BY RANDOM()
    LIMIT 100
"""

TOP_NAMES_WILDCARD_QUERY = """
    SELECT name, SUM(male  + female) as count
    FROM names
    WHERE lower(name) LIKE ?
    GROUP BY name
    ORDER BY count DESC
    LIMIT 10
"""

TOP_NAMES_SELECT_QUERY = """
    SELECT name, SUM(male  + female) as count
    FROM names
    WHERE lower(name) = ?
    GROUP BY name
    ORDER BY count DESC
"""

DATA_QUERY = """
    SELECT name, year, male, female, SUM(male + female) AS count
    FROM names
    WHERE name in ({placeholders})
    GROUP BY name, year, male, female
    ORDER BY name, year
"""


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", target_attr="value"):
        self.container = container
        self.text = initial_text
        self.target_attr = target_attr

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        setattr(self.container, self.target_attr, self.text)


class NameChronicles:
    def __init__(self, refresh=False):
        super().__init__()
        self.db_path = Path("data/names.db")
        self._initialize_database(refresh=refresh)

        # Main
        self.holoviews_pane = pn.pane.HoloViews(sizing_mode="stretch_both")
        self.selection = hv.streams.Selection1D()

        # Sidebar

        # Name Widgets
        self.names_input = pn.widgets.TextInput(name="Name Input", placeholder="Andrew")
        self.names_input.param.watch(self._add_name, "value")

        self.names_choice = pn.widgets.MultiChoice(
            name="Selected Names",
            options=["Andrew"],
            solid=False,
        )
        self.names_choice.param.watch(self._update_plot, "value")
        self.names_choice.value = ["Andrew"]

        # Reset Widgets
        self.clear_button = pn.widgets.Button(
            name="Clear Names", button_style="outline", button_type="primary"
        )
        self.clear_button.on_click(
            lambda event: setattr(self.names_choice, "value", [])
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh Plot", button_style="outline", button_type="primary"
        )
        self.refresh_button.on_click(self._refresh_plot)

        # Randomize Widgets
        self.name_pattern = pn.widgets.TextInput(
            name="Name Pattern", placeholder="*na*"
        )
        self.count_range = pn.widgets.IntRangeSlider(
            name="Peak Count Range",
            value=(10000, 50000),
            start=0,
            end=100000,
            step=1000,
            margin=(5, 20),
        )
        self.gender_select = pn.widgets.RadioButtonGroup(
            name="Gender",
            options=["Female", "Unisex", "Male"],
            button_style="outline",
            button_type="primary",
        )
        randomize_name = pn.widgets.Button(
            name="Get Name", button_style="outline", button_type="primary"
        )
        randomize_name.param.watch(self._randomize_name, "clicks")
        self.randomize_pane = pn.Card(
            self.name_pattern,
            self.count_range,
            self.gender_select,
            randomize_name,
            title="Get Random Name",
            collapsed=True,
        )

        # AI Widgets
        self.ai_key = pn.widgets.PasswordInput(
            name="OpenAI Key",
            placeholder="",
        )
        self.ai_prompt = pn.widgets.TextInput(
            name="AI Prompt",
            value="Share a little history about the name:",
        )
        ai_button = pn.widgets.Button(
            name="Get Response",
            button_style="outline",
            button_type="primary",
        )
        ai_button.on_click(self._prompt_ai)
        self.ai_response = pn.widgets.TextAreaInput(
            placeholder="",
            disabled=True,
            height=350,
        )
        self.ai_pane = pn.Card(
            self.ai_key,
            self.ai_prompt,
            ai_button,
            self.ai_response,
            collapsed=True,
            title="Ask AI",
        )

    # Database Methods

    def _connect_database(self):
        """
        Connect to the database.
        """
        return duckdb.connect(database=str(self.db_path))

    def _initialize_database(self, refresh):
        """
        Initialize database with data from the Social Security Administration.
        """
        if not refresh and self.db_path.exists():
            return

        df = pd.concat(
            [
                pd.read_csv(
                    path,
                    header=None,
                    names=["state", "gender", "year", "name", "count"],
                )
                for path in Path("data").glob("*.TXT")
            ]
        )
        df_processed = (
            df.groupby(["gender", "year", "name"], as_index=False)[["count"]]
            .sum()
            .pivot(index=["name", "year"], columns="gender", values="count")
            .reset_index()
            .rename(columns={"F": "female", "M": "male"})
            .fillna(0)
        )
        with self._connect_database() as conn:
            conn.execute("DROP TABLE IF EXISTS names")
            conn.execute("CREATE TABLE names AS SELECT * FROM df_processed")

    def _query_names(self, names):
        """
        Query the database for the given name.
        """
        dfs = []
        for name in names:
            if "*" in name or "%" in name:
                name = name.replace("*", "%")
                top_names_query = TOP_NAMES_WILDCARD_QUERY
            else:
                top_names_query = TOP_NAMES_SELECT_QUERY
            with self._connect_database() as conn:
                top_names = (
                    conn.execute(top_names_query, [name.lower()])
                    .fetch_df()["name"]
                    .tolist()
                )
                if len(top_names) == 0:
                    pn.state.notifications.info(f"No names found matching {name!r}")
                    continue
                data_query = DATA_QUERY.format(
                    placeholders=", ".join(["?"] * len(top_names))
                )
                df = conn.execute(data_query, top_names).fetch_df()
            dfs.append(df)

        if len(dfs) > 0:
            self.df = pd.concat(dfs).drop_duplicates(
                subset=["name", "year", "male", "female"]
            )
        else:
            self.df = pd.DataFrame(columns=["name", "year", "male", "female"])

    # Widget Methods

    def _randomize_name(self, event):
        with self._connect_database() as conn:
            name_pattern = self.name_pattern.value.lower()
            if not name_pattern:
                name_pattern = "%"
            else:
                name_pattern = name_pattern.replace("*", "%")
            count_range = self.count_range.value
            gender_select = self.gender_select.value.lower()
            random_names = (
                conn.execute(
                    RANDOM_NAME_QUERY, [name_pattern, *count_range, gender_select]
                )
                .fetch_df()["name"]
                .tolist()
            )
        if random_names:
            for i in range(len(random_names)):
                random_name = random_names[i]
                if random_name in self.names_choice.value:
                    continue
                self.names_input.value = random_name
                break
            else:
                pn.state.notifications.info(
                    "All names matching the criteria are already added!"
                )
        else:
            pn.state.notifications.info("No names found matching the criteria!")

    def _add_name(self, event):
        name = event.new.strip().title()
        self.names_input.value = ""
        if not name:
            return
        elif name in self.names_choice.options and name in self.names_choice.value:
            pn.state.notifications.info(f"{name!r} already added!")
            return
        elif len(self.names_choice.value) > 10:
            pn.state.notifications.info(
                "Maximum of 10 names allowed; please remove some first!"
            )
            return
        value = self.names_choice.value.copy()
        options = self.names_choice.options.copy()
        if name not in options:
            options.append(name)
        if name not in value:
            value.append(name)
        self.names_choice.param.update(
            options=options,
            value=value,
        )

    def _prompt_ai(self, event):
        if not self.ai_key.value:
            pn.state.notifications.info("Please enter an API key!")
            return

        if not self.ai_prompt.value:
            pn.state.notifications.info("Please enter a prompt!")
            return

        stream_handler = StreamHandler(self.ai_response)
        chat = ChatOpenAI(
            max_tokens=500,
            openai_api_key=self.ai_key.value,
            streaming=True,
            callbacks=[stream_handler],
        )
        self.ai_response.loading = True
        try:
            if self.selection.index:
                names = [self._name_indices[self.selection.index[0]]]
            else:
                names = self.names_choice.value[:3]
            chat.predict(f"{self.ai_prompt.value} {names}")
        finally:
            self.ai_response.loading = False

    # Plot Methods

    def _click_plot(self, index):
        gender_nd_overlay = hv.NdOverlay(kdims=["Gender"])
        if not index:
            return hv.NdOverlay(
                {
                    "curve": self._curve_nd_overlay,
                    "scatter": self._scatter_nd_overlay,
                    "label": self._label_nd_overlay,
                }
            )

        name = self._name_indices[index[0]]
        df_name = self.df.loc[self.df["name"] == name].copy()
        df_name["female"] += df_name["male"]
        gender_nd_overlay["Male"] = hv.Area(
            df_name, ["year"], ["male"], label="Male"
        ).opts(alpha=0.3, color="#add8e6", line_alpha=0)
        gender_nd_overlay["Female"] = hv.Area(
            df_name, ["year"], ["male", "female"], label="Female"
        ).opts(alpha=0.3, color="#ffb6c1", line_alpha=0)
        return hv.NdOverlay(
            {
                "curve": self._curve_nd_overlay[[index[0]]],
                "scatter": self._scatter_nd_overlay,
                "label": self._label_nd_overlay[[index[0]]].opts(text_color="black"),
                "gender": gender_nd_overlay,
            },
            kdims=["Gender"],
        ).opts(legend_position="top_left")

    @staticmethod
    def _format_y(value):
        return f"{value / 1000}k"

    def _update_plot(self, event):
        names = event.new
        print(names)
        self._query_names(names)

        self._scatter_nd_overlay = hv.NdOverlay()
        self._curve_nd_overlay = hv.NdOverlay(kdims=["Name"]).opts(
            gridstyle={"xgrid_line_width": 0},
            show_grid=True,
            fontscale=1.28,
            xlabel="Year",
            ylabel="Count",
            yformatter=self._format_y,
            legend_limit=0,
            padding=(0.2, 0.05),
            title="Name Chronicles",
            responsive=True,
        )
        self._label_nd_overlay = hv.NdOverlay(kdims=["Name"])
        hover_tool = HoverTool(
            tooltips=[("Name", "@name"), ("Year", "@year"), ("Count", "@count")],
        )
        self._name_indices = {}
        scatter_cycle = hv.Cycle("Category10")
        curve_cycle = hv.Cycle("Category10")
        label_cycle = hv.Cycle("Category10")
        for i, (name, df_name) in enumerate(self.df.groupby("name")):
            df_name_total = df_name.groupby(
                ["name", "year", "male", "female"], as_index=False
            )["count"].sum()
            df_name_total["male"] = df_name_total["male"] / df_name_total["count"]
            df_name_total["female"] = df_name_total["female"] / df_name_total["count"]
            df_name_peak = df_name.loc[[df_name["count"].idxmax()]]
            df_name_peak[
                "label"
            ] = f'{df_name_peak["name"].item()} ({df_name_peak["year"].item()})'

            hover_tool = HoverTool(
                tooltips=[
                    ("Name", "@name"),
                    ("Year", "@year"),
                    ("Count", "@count{(0a)}"),
                    ("Male", "@male{(0%)}"),
                    ("Female", "@female{(0%)}"),
                ],
            )
            self._scatter_nd_overlay[i] = hv.Scatter(
                df_name_total, ["year"], ["count", "male", "female", "name"], label=name
            ).opts(
                color=scatter_cycle,
                size=4,
                alpha=0.15,
                marker="y",
                tools=["tap", hover_tool],
                line_width=3,
                show_legend=False,
            )
            self._curve_nd_overlay[i] = hv.Curve(
                df_name_total, ["year"], ["count"], label=name
            ).opts(
                color=curve_cycle,
                tools=["tap"],
                line_width=3,
            )
            self._label_nd_overlay[i] = hv.Labels(
                df_name_peak, ["year", "count"], ["label"], label=name
            ).opts(
                text_align="right",
                text_baseline="bottom",
                text_color=label_cycle,
            )
            self._name_indices[i] = name
        self.selection.source = self._curve_nd_overlay
        if len(self._name_indices) == 1:
            self.selection.update(index=[0])
        else:
            self.selection.update(index=[])
        self.dynamic_map = hv.DynamicMap(
            self._click_plot, kdims=[], streams=[self.selection]
        ).opts(responsive=True)
        self._refresh_plot()

    def _refresh_plot(self, event=None):
        self.holoviews_pane.object = self.dynamic_map.clone()

    def view(self):
        reset_row = pn.Row(self.clear_button, self.refresh_button)
        data_url = pn.pane.Markdown(
            "<center>Data from the <a href='https://www.ssa.gov/oact/babynames/limits.html' "
            "target='_blank'>U.S. Social Security Administration</a></center>",
            align="end",
        )
        sidebar = pn.Column(
            self.names_input,
            self.names_choice,
            reset_row,
            pn.layout.Divider(),
            self.randomize_pane,
            self.ai_pane,
            data_url,
        )
        template = pn.template.FastListTemplate(
            sidebar=[sidebar],
            main=[self.holoviews_pane],
            title="Name Chronicles",
            theme="dark",
        )
        return template


NameChronicles().view().servable()
