import functools
import omegaconf
import streamlit as st
import pandas as pd
import plotly.express as px

############
mapping_version = "att_3"
part = "train"
dataset_name = omegaconf.OmegaConf.load('configs/config.yaml')["path"]
mapping = omegaconf.OmegaConf.load(f'configs/mapping/{mapping_version}.yaml')
############

chart = functools.partial(st.plotly_chart, use_container_width=True)


@st.experimental_memo
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Filepath"] = df["Filepath"].apply(lambda x: x.split("/")[-3])
    attributes = [attribute.name for attribute in mapping]
    columns = ["Filepath"] + attributes
    return df[columns].rename(columns={"Filepath": "datasets"})


@st.experimental_memo
def filter_data(
    df: pd.DataFrame, dataset_selections: list[str], attribute_selections: list[str]
) -> pd.DataFrame:
    df = df.copy()
    df = df[df.datasets.isin(dataset_selections)]
    columns = ["datasets"] + attribute_selections
    return df[columns]


def main() -> None:
    st.header("Attributes: наборы данных")
    with st.expander("Info"):
        st.write("Mapping:  " + mapping_version)
        st.write("Part:  " + part)
        st.write("Path:  " + dataset_name)

    df = pd.read_csv(dataset_name)
    df = clean_data(df)

    st.sidebar.subheader("Фильтрация по наборам")
    datasets = list(df.datasets.unique())
    dataset_selections = st.sidebar.multiselect(
        "Укажите наборы данных", options=datasets, default=datasets
    )

    st.sidebar.subheader("Фильтрация по атрибутам")

    attributes = [attribute.name for attribute in mapping]
    attribute_selections = st.sidebar.multiselect(
        "Укажите атрибуты", options=attributes, default=attributes
    )

    st.subheader(f"Всего картинок: {len(df.index)}")

    st.subheader("Соотношение размеров наборов")
    pie = df.groupby(["datasets"])["datasets"].count().to_frame()
    pie["name"] = pie.index
    pie = pie.rename(columns={"datasets": "images count"})
    fig = px.pie(pie, values="images count", names="name")
    chart(fig)

    df = filter_data(df, dataset_selections, attribute_selections)
    for id, attribute in enumerate(attributes):
        st.subheader(f"{attribute}")
        count = df.groupby([attribute])[attribute].count().to_frame()
        count["value"] = count.index
        count["value"] = count["value"].apply(lambda x: mapping[id]["values"][x] if x != "-1" else "Непонятно")
        fig = px.bar(count, y=attribute, x="value", color='value')
        fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
        fig.update(layout_coloraxis_showscale=False)
        chart(fig)


if __name__ == "__main__":
    st.set_page_config(
        "Fidelity Account View by Gerard Bentley",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
