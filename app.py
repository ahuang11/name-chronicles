from pathlib import Path

import duckdb
import holoviews as hv
import pandas as pd
import panel as pn
from bokeh.models import HoverTool
from bokeh.models import NumeralTickFormatter
from pydantic import BaseModel, Field
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

pn.extension(sizing_mode="stretch_width", notifications=True)
hv.extension("bokeh")

INSTRUCTIONS = """
    #### Name Chronicles lets you explore the history of names in the United States.
    - Enter a name to add to plot!
    - Hover over a line for stats or click for the gender distribution.
    - Chat with AI for inspiration or get a random name based on input criteria.
    - Have ideas? [Open an issue](https://github.com/ahuang11/name-chronicles/issues).
"""

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

MAX_LLM_COUNT = 2000

class FirstNames(BaseModel):
    names: list[str] = Field(description="List of first names")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", target_attr="value"):
        self.container = container
        self.text = initial_text
        self.target_attr = target_attr

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        setattr(self.container, self.target_attr, self.text)


class NameChronicles:
    def __init__(self):
        super().__init__()
        self.llm_use_counter = 0
        self.db_path = Path("data/names.db")

        # Main
        self.holoviews_pane = pn.pane.HoloViews(
            min_height=675, sizing_mode="stretch_both"
        )
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
        self.chat_interface = pn.chat.ChatInterface(
            show_button_name=False,
            callback=self._prompt_ai,
            height=500,
            styles={"background": "white"},
            disabled=True,
        )
        self.chat_interface.send(
            value=(
                "Ask me about name suggestions or their history! "
                "To add suggested names, click the button below!"
            ),
            user="System",
            respond=False,
        )
        self.parse_ai_button = pn.widgets.Button(
            name="Parse and Add Names",
            button_style="outline",
            button_type="primary",
            disabled=False,
        )
        pn.state.onload(self._initialize_database)

    # Database Methods

    def _initialize_database(self):
        """
        Initialize database with data from the Social Security Administration.
        """
        self.conn = duckdb.connect(":memory:")
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
        self.conn.execute("DROP TABLE IF EXISTS names")
        self.conn.execute("CREATE TABLE names AS SELECT * FROM df_processed")

        if self.names_choice.value == []:
            self.names_choice.value = ["Andrew"]
        else:
            self.names_choice.param.trigger("value")
        self.main.objects = [self.holoviews_pane]

        # Start AI
        self.callback_handler = pn.chat.langchain.PanelCallbackHandler(
            self.chat_interface
        )
        self.chat_openai = ChatOpenAI(
            max_tokens=75,
            streaming=True,
            callbacks=[self.callback_handler],
        )
        self.openai = OpenAI(max_tokens=75)
        memory = ConversationBufferMemory()
        self.conversation_chain = ConversationChain(
            llm=self.chat_openai, memory=memory, callbacks=[self.callback_handler]
        )
        self.chat_interface.disabled = False
        self.parse_ai_button.on_click(self._parse_ai_output)
        self.pydantic_parser = PydanticOutputParser(pydantic_object=FirstNames)
        self.prompt_template = PromptTemplate(
            template="{format_instructions}\n{input}\n",
            input_variables=["input"],
            partial_variables={"format_instructions": self.pydantic_parser.get_format_instructions()},
        )

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
            top_names = (
                self.conn.execute(top_names_query, [name.lower()])
                .fetch_df()["name"]
                .tolist()
            )
            if len(top_names) == 0:
                pn.state.notifications.info(f"No names found matching {name!r}")
                continue
            data_query = DATA_QUERY.format(
                placeholders=", ".join(["?"] * len(top_names))
            )
            df = self.conn.execute(data_query, top_names).fetch_df()
            dfs.append(df)

        if len(dfs) > 0:
            self.df = pd.concat(dfs).drop_duplicates(
                subset=["name", "year", "male", "female"]
            )
        else:
            self.df = pd.DataFrame(columns=["name", "year", "male", "female"])

    # Widget Methods

    def _randomize_name(self, event):
        name_pattern = self.name_pattern.value.lower()
        if not name_pattern:
            name_pattern = "%"
        else:
            name_pattern = name_pattern.replace("*", "%")
        count_range = self.count_range.value
        gender_select = self.gender_select.value.lower()
        random_names = (
            self.conn.execute(
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

    def _add_only_unique_names(self, names):
        value = self.names_choice.value.copy()
        options = self.names_choice.options.copy()
        for name in names:
            if " " in name:
                name = name.split(" ", 1)[0]
            if name not in options:
                options.append(name)
            if name not in value:
                value.append(name)
        self.names_choice.param.update(
            options=options,
            value=value,
        )

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
        self._add_only_unique_names([name])

    async def _prompt_ai(self, contents, user, instance):
        if self.llm_use_counter >= MAX_LLM_COUNT:
            pn.state.notifications.info(
                "Sorry, all the available AI credits have been used!"
            )
            return

        prompt = (
            f"One sentence reply to {contents!r} or concisely suggest other relevant names; "
            f"if no name is provided use {self.names_choice.value[-1]!r}."
        )
        self.last_ai_output = await self.conversation_chain.apredict(
            input=prompt,
            callbacks=[self.callback_handler],
        )
        self.llm_use_counter += 1
    
    async def _parse_ai_output(self, _):
        if self.llm_use_counter >= MAX_LLM_COUNT:
            pn.state.notifications.info(
                "Sorry, all the available AI credits have been used!"
            )
            return

        if self.last_ai_output is None:
            pn.state.notifications.info("No available AI output to parse!")
            return

        try:
            names_prompt = self.prompt_template.format_prompt(input=self.last_ai_output).to_string()
            names_text = await self.openai.apredict(names_prompt)
            new_names = (await self.pydantic_parser.aparse(names_text)).names
            print(new_names)
            self._add_only_unique_names(new_names)
        except Exception:
            pn.state.notifications.error("Failed to parse AI output.")
        finally:
            self.last_ai_output = None

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
            yformatter=NumeralTickFormatter(format="0.0a"),
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
            INSTRUCTIONS,
            self.names_input,
            self.names_choice,
            reset_row,
            pn.layout.Divider(),
            self.chat_interface,
            self.parse_ai_button,
            self.randomize_pane,
            data_url,
        )
        self.main = pn.Column(
            pn.widgets.StaticText(
                value="Loading, this may take a few seconds...",
                sizing_mode="stretch_both",
            ),
        )
        template = pn.template.FastListTemplate(
            sidebar_width=500,
            sidebar=[sidebar],
            main=[self.main],
            title="Name Chronicles",
            theme="dark",
        )
        return template


NameChronicles().view().servable()
