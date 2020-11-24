import streamlit as st

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


@st.cache(show_spinner=False)
def load_data(data_to_load):
    """
    Loads data. Using function for this for caching purposes
    Returns None if something unexpected happens
    """
    if data_to_load == "baby_names":
        baby_names = pd.read_csv(
            "data/data_mod/names_combined.csv", keep_default_na=False
        )
        baby_names.set_index(["year", "gender", "name"], inplace=True)
        return baby_names
    elif data_to_load == "biblical_names":
        with open("data/data_external/biblical_names.txt", "r") as f:
            biblical_names = f.read()
            biblical_names = set(biblical_names.split(","))
            return biblical_names
    elif data_to_load == "religion_trend":
        religion_trend = pd.read_csv("data/data_external/religion_trend_america.csv")
        return religion_trend
    return None


@st.cache(show_spinner=False)
def get_top_n_names_sorted(df, gender, n):
    """
    Find top n names for given gender, for each year. Sort by year in which name peaked in popularity
    """
    top_n_names = (
        df.query("gender==@gender")
        .groupby(["year"], group_keys=False)
        .apply(lambda grp: grp.head(n))
    )
    top_n_names = top_n_names.index.get_level_values(2).unique()
    top_n_names_rows = df.query(
        "gender==@gender and name in @top_n_names"
    ).reset_index()
    top_n_names = top_n_names_rows.loc[
        top_n_names_rows.groupby("name")["pct_gender"].idxmax(), ["name", "year"]
    ].sort_values("year")
    return top_n_names


@st.cache(show_spinner=False)
def get_names_data_filled(df, gender, names):
    """
    Return all years' data for a list of names, including years where the name is not in dataset
    Also includes peak year data
    """
    # Pivot to input all years' data for all names
    data_filled = (
        df.reset_index()
        .query("gender==@gender and name in @names")
        .reset_index()
        .pivot(index="year", columns="name", values="pct_gender")
        .fillna(0.0)
    )

    # Fill in year
    data_years = list(data_filled.index) * len(names)
    data_filled = data_filled.melt()
    data_filled.rename({"value": "pct_gender"}, axis=1, inplace=True)
    data_filled["year"] = data_years

    # Fill in data for peaks
    data_filled["peak"] = None
    for name in names:
        curr_name_data = data_filled.query("name==@name")
        curr_name_peaks = find_name_peaks(curr_name_data["pct_gender"])
        curr_name_peaks = curr_name_data.index[curr_name_peaks]
        data_filled.loc[curr_name_peaks, "peak"] = data_filled.loc[
            curr_name_peaks, "pct_gender"
        ]
    data_filled["peak"] = data_filled["peak"].astype(np.float)

    return data_filled


@st.cache(show_spinner=False)
def find_name_peaks(pct):
    """
    Given list of percentages (ordered by year), find the index(es) at which pct peaks
    """
    peak_width = 1
    # Pad start and end with 0's, to account for cases where peak is at start / end
    zero_padding = peak_width
    pct = np.concatenate([[0] * zero_padding, pct, [0] * zero_padding])
    max_height = max(pct)
    prominence_req = min(2.0, 0.4 * max_height)
    peaks = find_peaks(
        pct,
        distance=10,
        width=peak_width,
        height=0.4 * max_height,
        prominence=prominence_req,
    )
    peaks = list(peaks[0])
    peaks = [ele - zero_padding for ele in peaks]
    return peaks


def plot_name_pct_peak(names_annual_data, names_overall_data, initial_name):
    year_range = [list(names_overall_data["year"])[i] for i in [0, -1]]
    fig = px.line(
        names_annual_data.query("name==@initial_name"),
        x="year",
        y=["pct_gender", "peak"],
        range_x=(year_range[0] - 1, year_range[1] + 1),
        title=f"Popularity of name '<b>{initial_name.capitalize()}</b>' by year",
        labels={"value": "% of babies born", "year": "Year"},
    )
    fig.data[1].update(mode="markers", marker_symbol="x", marker_size=10)
    fig.layout.yaxis["rangemode"] = "tozero"
    fig.update_layout(showlegend=False)

    # Create and add slider
    steps = []
    for _, row in names_overall_data.iterrows():
        curr_name = row["name"]
        curr_name_data = names_annual_data.query("name==@curr_name")
        step = {
            "method": "update",
            "args": [
                {"y": [curr_name_data["pct_gender"], curr_name_data["peak"]]},
                {
                    "title": f"Popularity of name '<b>{curr_name.capitalize()}</b>' by year"
                },
            ],
            "label": curr_name.capitalize(),
        }
        steps.append(step)
    sliders = [
        dict(
            active=list(names_overall_data["name"]).index(initial_name),
            currentvalue={"prefix": "Name: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders)
    st.write(fig)


def main():
    st.set_page_config(
        page_title="US Baby Names",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="auto",
    )

    # Load data
    baby_names = load_data("baby_names")
    biblical_names = load_data("biblical_names")
    religion_trends = load_data("religion_trends")

    # Get data for top n names
    gender, n = "F", 10
    top_10_female_names = get_top_n_names_sorted(baby_names, gender, n)
    top_10_female_names_data = get_names_data_filled(
        baby_names, gender, top_10_female_names["name"]
    )

    gender, n = "M", 10
    top_10_male_names = get_top_n_names_sorted(baby_names, gender, n)
    top_10_male_names_data = get_names_data_filled(
        baby_names, gender, top_10_male_names["name"]
    )

    # Visualize top n names
    gender_input = st.radio("Gender", options=["F", "M"])

    @st.cache(suppress_st_warning=True)
    def get_name_dropdown_mapping(names_peak_year):
        labels = list(
            names_peak_year.apply(
                lambda row: f"({row['year']}) {row['name'].capitalize()}", axis=1
            )
        )
        names = names_peak_year["name"]
        mapping = {label: name for label, name in zip(labels, names)}
        return mapping

    names_dropdown_mapping = {
        "M": get_name_dropdown_mapping(top_10_male_names),
        "F": get_name_dropdown_mapping(top_10_female_names),
    }

    if gender_input == "M":
        name_input = st.selectbox(
            "Names",
            options=list(names_dropdown_mapping["M"].keys()),
            index=list(names_dropdown_mapping["M"].values()).index("david"),
        )
    else:
        name_input = st.selectbox(
            "Names",
            options=list(names_dropdown_mapping["F"].keys()),
            index=list(names_dropdown_mapping["F"].values()).index("karen"),
        )

    name_selected = names_dropdown_mapping[gender_input][name_input]
    if gender_input == "M":
        plot_name_pct_peak(top_10_male_names_data, top_10_male_names, name_selected)
    else:
        plot_name_pct_peak(top_10_female_names_data, top_10_female_names, name_selected)


if __name__ == "__main__":
    main()
