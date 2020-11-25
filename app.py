import os, re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


def init_sidebar_elements():
    st.sidebar.header("Data")
    st.sidebar.markdown(
        "Extracted from [Data.gov](https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-level-data)"
    )
    st.sidebar.write(
        """
        The data lists 1) the first names of babies born in the US, and 2) the number of babies born with each name, from 1880 to 2019.  \n
        Only names with >5 occurences in a year are included in the dataset.  \n
        Note that we have converted all counts to percentages.  \n
        For example, if there were 10 million female babies born in 2000, and 1 million of them were named Mary, we converted the 1 million to 10%
        """
    )


@st.cache(show_spinner=False)
def load_data(data_to_load, prereq_data=None):
    """
    Loads data. Using function for this for caching purposes
    Returns None if something unexpected happens
    """
    data_dir = "data/"
    if data_to_load == "baby_names":
        # Concat all data files
        baby_names = []
        for filename in os.listdir(data_dir):
            if filename.startswith("yob"):
                year = int(re.match("yob(.+)\.txt", filename).groups(0)[0])
                curr_names_df = pd.read_csv(
                    data_dir + filename, names=["name", "gender", "count"]
                )
                curr_names_df["year"] = year
                baby_names.append(curr_names_df)
        baby_names = pd.concat(baby_names)
        baby_names.sort_values(
            by=["year", "gender", "count"], ascending=[True, True, False]
        )

        # Count as a pct of total count (each gender)
        total_count_per_year_gender = baby_names.groupby(["year", "gender"])[
            "count"
        ].sum()
        baby_names["pct_gender"] = baby_names.groupby(["year", "gender"])[
            "count"
        ].transform(
            lambda row: row
            / total_count_per_year_gender.loc[row.name[0], row.name[1]]
            * 100
        )

        # Count as a pct of total count (both genders)
        total_count_per_year = baby_names.groupby(["year"])["count"].sum()
        baby_names["pct_total"] = baby_names.groupby(["year"])["count"].transform(
            lambda row: row / total_count_per_year.loc[row.name] * 100
        )

        # Convert name to lowercase
        baby_names["name"] = baby_names["name"].apply(lambda name: name.lower())

        baby_names.set_index(["year", "gender", "name"], inplace=True)
        baby_names.sort_values(
            by=["year", "gender", "count"], ascending=[True, True, False], inplace=True
        )
        return baby_names
    elif data_to_load == "biblical_names":
        with open("data/data_external/biblical_names.txt", "r") as f:
            biblical_names = f.read()
            biblical_names = set(biblical_names.split(","))
            return biblical_names
    elif data_to_load == "religion_trend":
        religion_trend = pd.read_csv("data/data_external/religion_trend_america.csv")
        return religion_trend
    elif data_to_load == "all_names_by_gender":
        all_names_by_gender = {}
        all_names_by_gender["M"] = list(
            prereq_data.query("gender=='M'").index.get_level_values(2).unique()
        )
        all_names_by_gender["F"] = list(
            prereq_data.query("gender=='F'").index.get_level_values(2).unique()
        )
        return all_names_by_gender
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
def get_names_data_filled(df, gender, names, calc_peak):
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
    if calc_peak:
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


def plot_name_pct_peak(names_annual_data, names_overall_data, initial_name, gender):
    gender = "males" if gender == "M" else "females"
    year_range = [list(names_overall_data["year"])[i] for i in [0, -1]]
    fig = px.line(
        names_annual_data.query("name==@initial_name"),
        x="year",
        y=["pct_gender", "peak"],
        range_x=(year_range[0] - 1, year_range[1] + 1),
        title=f"% of newborn {gender} named '<b>{initial_name.capitalize()}</b>', by year",
        labels={"value": f"% newborn {gender}", "year": "Year"},
        # hover_data={"variable": False},
    )
    fig.data[1].update(mode="markers", marker_symbol="x", marker_size=10)
    fig.layout.yaxis["rangemode"] = "tozero"
    fig.update_layout(showlegend=False)
    fig.update_layout(hovermode="x unified")
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    # Update hover text
    fig.update_traces(hovertemplate="%{y:.4f}%<extra></extra>")

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
                    "title": f"% of newborn {gender} named '<b>{curr_name.capitalize()}</b>', by year"
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
    st.plotly_chart(fig, use_container_width=True)


def init_header_elements():
    st.title("Explore US baby names (1880 - 2019)")
    st.write(
        "Inspired by certain memes, we started exploring baby names in the US.  \n"
        "Click on the left sidebar to read about the data.  \n"
        "Scroll down to play with the data!"
    )


def init_top_n_names_elements(
    top_10_female_names,
    top_10_female_names_data,
    top_10_male_names,
    top_10_male_names_data,
):
    # Visualize top n names
    st.subheader("In 2050 - Don't be an Olivia?")

    st.write(
        """
        The data above contains baby names which were once top 10 in popularity.  \n
        Select the gender, then the name from the dropdown menu. Or, use the slider to choose the names. The chart shows the rough percentage of newborns in each year with the selected name.
        """
    )
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
            index=list(names_dropdown_mapping["M"].values()).index("oliver"),
        )
    else:
        name_input = st.selectbox(
            "Names",
            options=list(names_dropdown_mapping["F"].keys()),
            index=list(names_dropdown_mapping["F"].values()).index("olivia"),
        )

    name_selected = names_dropdown_mapping[gender_input][name_input]
    if gender_input == "M":
        plot_name_pct_peak(
            top_10_male_names_data, top_10_male_names, name_selected, "M"
        )
    else:
        plot_name_pct_peak(
            top_10_female_names_data, top_10_female_names, name_selected, "F"
        )


def init_explore_name_trends(baby_names, all_names_by_gender):
    st.subheader("Explore the data yourself")
    st.write(
        """
        Use the 2 search boxes below to find names and compare their popularity. You can select multiple names per box.
        """
    )
    female_names_selection = st.multiselect(
        "Female names:", options=all_names_by_gender["F"],
    )
    male_names_selection = st.multiselect(
        "Male names:", options=all_names_by_gender["M"],
    )

    female_chart_data = get_names_data_filled(
        baby_names, "F", female_names_selection, False
    ).copy(deep=True)

    female_chart_data["name"] = female_chart_data["name"].apply(
        lambda name: f"{name} (F)"
    )

    male_chart_data = get_names_data_filled(
        baby_names, "M", male_names_selection, False
    ).copy(deep=True)

    male_chart_data["name"] = male_chart_data["name"].apply(lambda name: f"{name} (M)")

    chart_data = pd.concat([female_chart_data, male_chart_data])
    chart_data = chart_data.pivot(
        index="year", columns="name", values="pct_gender"
    ).reset_index()
    chart_data.columns = [col[0].upper() + col[1:] for col in chart_data.columns]

    year_range = [list(baby_names.index.get_level_values(0))[i] for i in [0, -1]]

    # Empty graph if no names selected
    if len(chart_data) == 0:
        x_curr = list(range(year_range[0], year_range[1]))
        y_curr = [np.nan] * len(x_curr)
        fig = px.line(
            x=x_curr,
            y=y_curr,
            range_x=(year_range[0] - 1, year_range[1] + 1),
            range_y=(0, 3),
            labels={"x": "Year"},
        )
    else:
        fig = px.line(
            chart_data,
            x="Year",
            y=chart_data.columns.drop("Year"),
            range_x=(year_range[0] - 1, year_range[1] + 1),
            labels={"variable": "Name"},
        )
    fig.layout.title = "% of newborn females / males with selected name(s)"
    fig.layout.yaxis["rangemode"] = "tozero"
    fig.layout.yaxis["title"] = "% newborn females / males"
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    fig.update_layout(hovermode="x unified")

    # Update hover text
    fig.update_traces(hovertemplate="%{data.name}:<br>%{y:.4f}%<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="US Baby Names",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="auto",
    )
    init_sidebar_elements()
    init_header_elements()

    # Load data
    baby_names = load_data("baby_names")
    biblical_names = load_data("biblical_names")
    religion_trends = load_data("religion_trends")
    all_names_by_gender = load_data("all_names_by_gender", baby_names)

    # Get data for top n names
    gender, n = "F", 10
    top_10_female_names = get_top_n_names_sorted(baby_names, gender, n)
    top_10_female_names_data = get_names_data_filled(
        baby_names, gender, top_10_female_names["name"], True
    )

    gender, n = "M", 10
    top_10_male_names = get_top_n_names_sorted(baby_names, gender, n)
    top_10_male_names_data = get_names_data_filled(
        baby_names, gender, top_10_male_names["name"], True
    )

    init_top_n_names_elements(
        top_10_female_names,
        top_10_female_names_data,
        top_10_male_names,
        top_10_male_names_data,
    )

    st.markdown("---")

    init_explore_name_trends(baby_names, all_names_by_gender)


if __name__ == "__main__":
    main()
