import pandas as pd
import streamlit as st
# import plotly.express as px
# import matplotlib.pyplot as plt

from visualization.visualize import (
    fig_histogram,
    fig_scatter,
    fig_heat_map,
    fig_box_plt,
    fig_area_chart,
    fig_pi_chart,
)


def data_exploration():
    st.markdown("Upload your dataset and explore it interactively with visualizations.")

    tab1, tab2, tab3 = st.tabs(
        ["📂 Upload Data", "📊 Data Exploration", "📈 Data Visualization"]
    )

    with tab1:
        st.header("📂 Upload Your Dataset")
        st.markdown("Upload a CSV file to begin exploring your data.")
        uploaded_file = st.file_uploader("Please upload a CSV file:", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state["df"] = df
                st.success("✅ File uploaded successfully!")
                st.write("### Data Preview")
                st.dataframe(df.head(5), use_container_width=True)
                st.write(
                    f"**Number of Rows:** {df.shape[0]} | **Number of Columns:** {df.shape[1]}"
                )
            except Exception as e:
                st.error(f"⚠️ An error occurred while reading the file: {e}")
        else:
            st.info("📥 Please upload a CSV file to proceed.")

    with tab2:
        st.header("📊 Data Exploration")
        st.markdown("Explore the structure and summary of your dataset.")

        if "df" in st.session_state:
            df = st.session_state["df"]

            st.subheader("🔍 Data Preview")
            with st.expander("📑 Data Preview"):
                st.dataframe(df.head(5))
            with st.expander("📈 Descriptive Statistics", expanded=False):
                st.dataframe(df.describe(), use_container_width=True)

            with st.expander("🚨 Missing Values"):
                missing_values = df.isnull().sum()
                if missing_values.sum() > 0:
                    st.write(missing_values[missing_values > 0])
                else:
                    st.success("No missing values found!")

            st.subheader("📊 Correlation Matrix")
            with st.expander("🔗 Correlation Matrix", expanded=False):
                st.markdown(
                    "The correlation matrix shows the relationships between numerical features."
                )
                num_df = df.select_dtypes(include=["number"])
                corr_matrix = num_df.corr()
                st.dataframe(corr_matrix, use_container_width=True)

            st.subheader("📋 Data Summary")

            with st.expander("Data Summary"):
                st.write(f"**Number of Rows:** {df.shape[0]}")
                st.write(f"**Number of Columns:** {df.shape[1]}")
                st.write("**Column Data Types:**")
                st.write(df.dtypes)
        else:
            st.warning("⚠️ Please upload a dataset in the **Upload Data** tab first.")

    with tab3:
        st.header("📈 Data Visualization")
        st.markdown("Create interactive visualizations to better understand your data.")

        if "df" not in st.session_state:
            st.warning("⚠️ Please upload a dataset in the **Upload Data** tab first.")
            return

        df = st.session_state["df"]

        with st.expander("🔧 Visualization Settings", expanded=True):
            st.markdown("Configure the settings for your visualizations below:")

            col1, col2, col3 = st.columns(3)
            with col1:
                target = st.selectbox(
                    "🎯 Select Target Variable",
                    [None] + df.columns.tolist(),
                    help="Choose the column that represents the target variable. It will be used for correlation, marker size, and box plots. Select 'None' to exclude a target variable.",
                )
            with col2:
                x = st.selectbox(
                    "📊 Select X-Axis Variable",
                    df.columns,
                    help="Choose the column to be used for the X-axis in visualizations.",
                )
            with col3:
                y = st.selectbox(
                    "📈 Select Y-Axis Variable",
                    df.columns,
                    help="Choose the column to be used for the Y-axis in visualizations.",
                )

            col4, col5 = st.columns([2, 1])
            with col4:
                color = st.selectbox(
                    "🎨 Select Color Variable",
                    [None] + df.columns.to_list(),
                    help="Choose the column to color the data points in visualizations.",
                )
            with col5:
                n_bins = st.slider(
                    "🔢 Number of Bins (for Histogram)",
                    min_value=3,
                    max_value=75,
                    value=35,
                    step=1,
                    help="Adjust the number of bins for the histogram visualization.",
                )
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "📊 Histogram",
                "📦 Box Plot",
                "📉 Area Chart",
                "📍 Scatter Plot",
                "🔥 Heatmap",
                "🥧 Pi Chart",
            ]
        )

        with tab1:
            st.header("📊 Histogram")
            st.markdown(
                "Explore the distribution of a numerical variable in your dataset using a histogram. "
                "You must select an X variable (column) to generate the chart. "
                "Optionally, you can group the bars by another column using the **color** option."
            )

            if x:
                st.markdown(f"### Histogram of `{x}`")
                if color:
                    st.markdown(f"Grouped by `{color}`")
                st.plotly_chart(
                    fig_histogram(df=df, x=x, color=color, nbins=n_bins),
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Please select an X variable (column) to generate the histogram."
                )

        with tab2:
            st.header("📦 Box Plot")
            st.markdown(
                "Analyze the distribution of a numerical variable and detect outliers using a box plot. "
                "You must select a **target** variable (numerical column) to generate the chart. "
                "Optionally, you can group the boxes by another column using the **color** option."
            )

            if target:
                st.markdown(f"### Box Plot for `{target}`")
                if color:
                    st.markdown(f"Grouped by `{color}`")
                try:
                    st.plotly_chart(
                        fig_box_plt(df=df, column=target, group_by=color),
                        use_container_width=True,
                    )
                except ValueError as e:
                    st.warning(str(e))
            else:
                st.warning(
                    "Please select a target variable (numerical column) to generate the box plot."
                )

        with tab3:
            st.header("📉 Area Chart")
            st.markdown(
                "Visualize trends over time or across categories using an area chart. "
                "You must select both X and Y variables (columns) to generate the chart. "
                "Optionally, you can group the areas by another column using the **color** option."
            )

            if x and y:
                st.markdown(f"### Area Chart: `{y}` vs `{x}`")
                if color:
                    st.markdown(f"Grouped by `{color}`")
                st.plotly_chart(
                    fig_area_chart(df=df, x=x, y=y, color=color),
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Please select both X and Y variables (columns) to generate the area chart."
                )

        with tab4:
            st.header("📍 Scatter Plot")
            st.markdown(
                "Visualize relationships between two numerical variables in your dataset using a scatter plot. "
                "You must select both X and Y variables (columns) to generate the chart. "
                "Optionally, you can group the points by another column using the **color** option, "
                "and adjust marker size by a selected **target** variable."
            )

            if x and y:
                st.markdown(f"### Scatter Plot: `{x}` vs `{y}`")
                if color:
                    st.markdown(f"Grouped by `{color}`")
                if target:
                    st.markdown(f"Marker size by `{target}`")
                st.plotly_chart(
                    fig_scatter(df=df, x=x, y=y, color=color, size=target),
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Please select both X and Y variables (columns) to generate the scatter plot."
                )

        with tab5:
            st.header("🔥 Heatmap")
            st.markdown(
                "Analyze correlations between numerical variables in your dataset using a heatmap. "
                "If you select a **target** variable, the heatmap will show correlations with that variable. "
                "If no target is selected, the full correlation matrix will be displayed."
            )

            if target:
                st.markdown(f"### Correlation Heatmap with `{target}`")
            else:
                st.markdown("### Full Correlation Matrix Heatmap")

            try:
                st.plotly_chart(
                    fig_heat_map(df=df, target=target), use_container_width=True
                )
            except ValueError as e:
                st.warning(str(e))

        with tab6:
            st.header("🥧 Pie Chart")
            st.markdown(
                "Visualize the distribution of values for a selected column in your dataset as a pie chart. "
                "You must select a **target** variable (column) to generate the chart. "
                "Optionally, you can group the slices by another column using the **color** option."
            )

            if target:
                st.markdown(f"### Pie Chart for `{target}`")
                if color:
                    st.markdown(f"Grouped by `{color}`")
                st.plotly_chart(
                    fig_pi_chart(df=df, target=target, color=color),
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Please select a target variable (column) to generate the pie chart."
                )
                st.plotly_chart(
                    fig_pi_chart(df=df, target=None), use_container_width=True
                )
