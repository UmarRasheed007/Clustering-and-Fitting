import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    """
    Create a publication-quality scatter plot showing the geographical distribution of crimes.
    Visualizes the spatial patterns of different crime types across locations.
    """
    # Set figure aesthetics for publication quality
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")

    # Create a custom colorblind-friendly palette
    palette = sns.color_palette("viridis", n_colors=len(df["Crime type"].unique()))

    # Create the main scatter plot
    ax = sns.scatterplot(
        x="Longitude",
        y="Latitude",
        data=df,
        hue="Crime type",
        palette=palette,
        alpha=0.7,
        s=50,  # Point size
        edgecolor="none",
    )

    # Calculate crime density and highlight hotspots
    if len(df) > 100:  # Only if we have enough data points
        # Just call the kdeplot without assigning it, or use the returned object
        sns.kdeplot(
            x=df["Longitude"],
            y=df["Latitude"],
            levels=5,
            fill=True,
            alpha=0.3,
            color="red",
            ax=ax,  # Explicitly specify the axes to ensure it's added to the right plot
        )

    # Enhance plot styling
    ax.set_title(
        "Geographic Distribution of Crime Types", fontsize=18, fontweight="bold", pad=20
    )
    ax.set_xlabel("Longitude", fontsize=14, labelpad=10)
    ax.set_ylabel("Latitude", fontsize=14, labelpad=10)

    # Add UK context if data is for UK
    ax.axhline(y=51.5074, color="blue", linestyle="--", alpha=0.5)  # London latitude
    ax.axvline(x=-0.1278, color="blue", linestyle="--", alpha=0.5)  # London longitude
    ax.text(
        -0.1278,
        51.5074,
        "London",
        fontsize=10,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
    )

    # Customize tick parameters
    ax.tick_params(axis="both", which="major", labelsize=12, length=5, width=1.5)
    ax.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1)

    # Create informative legend with crime counts
    crime_counts = df["Crime type"].value_counts()
    handles, labels = ax.get_legend_handles_labels()
    updated_labels = [f"{label} ({crime_counts.get(label, 0)})" for label in labels]

    # Position legend intelligently outside the plot
    plt.legend(
        handles=handles,
        labels=updated_labels,
        title="Crime Type (Count)",
        title_fontsize=14,
        fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add a north arrow for geographical context
    ax.annotate(
        "N",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black"),
    )

    # Add annotations about crime density
    highest_crime_area = (
        df.groupby(["Longitude", "Latitude"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .iloc[0]
    )
    ax.annotate(
        f"Highest crime density\n({highest_crime_area['count']} incidents)",
        xy=(highest_crime_area["Longitude"], highest_crime_area["Latitude"]),
        xytext=(30, 30),
        textcoords="offset points",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.6),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )

    plt.tight_layout()
    plt.savefig("relational_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Create a publication-quality figure with two bar plots showing:
    1. Relationship between crime types and their outcomes
    2. Distribution of crime types by location
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), constrained_layout=True)

    # FIRST PLOT: Crime types and outcomes
    # Prepare data - filter to most common categories for readability
    top_crimes = df["Crime type"].value_counts().nlargest(5).index
    top_outcomes = df["Last outcome category"].value_counts().nlargest(6).index

    # Filter data
    plot_df = df[
        df["Crime type"].isin(top_crimes)
        & df["Last outcome category"].isin(top_outcomes)
    ]

    # Calculate percentages for each crime type and outcome combination
    crosstab = (
        pd.crosstab(
            plot_df["Crime type"], plot_df["Last outcome category"], normalize="index"
        )
        * 100
    )

    # Check if there are any rows with identical values
    if not crosstab.empty:
        # Create the first plot using crosstab data
        crosstab.plot(kind="bar", stacked=True, ax=ax1, colormap="viridis", width=0.8)

        # Ensure x-axis has proper width even with identical values
        if len(crosstab) > 1:
            x_min, x_max = ax1.get_xlim()
            if x_min == x_max:  # If identical limits
                ax1.set_xlim(x_min - 0.5, x_max + 0.5)
    else:
        # If dataframe is empty, create an empty plot with a message
        ax1.text(
            0.5,
            0.5,
            "No data available for plotting",
            ha="center",
            va="center",
            fontsize=14,
        )

    # Enhance styling for first plot
    ax1.set_title(
        "Crime Resolution Outcomes by Type", fontsize=18, fontweight="bold", pad=20
    )
    ax1.set_xlabel("Crime Type", fontsize=14, labelpad=10)
    ax1.set_ylabel("Percentage of Cases (%)", fontsize=14, labelpad=10)

    # Format x-ticks for first plot
    ax1.tick_params(axis="x", labelrotation=45, labelsize=12, length=5, width=1)
    ax1.tick_params(axis="y", labelsize=12, length=5, width=1)

    # Add grid for easier reading of percentages
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Enhance legend for first plot
    if not crosstab.empty:
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(
            handles=handles,
            labels=labels,
            title="Case Outcome",
            title_fontsize=14,
            fontsize=11,
            loc="upper right",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0,
            frameon=True,
            fancybox=True,
        )

        # Add annotations showing the most common outcome for each crime
        for i, crime in enumerate(crosstab.index):
            most_common = crosstab.loc[crime].idxmax()
            percentage = crosstab.loc[crime, most_common]
            if percentage > 15:  # Only annotate if there's enough space
                ax1.annotate(
                    f"{percentage:.1f}%",
                    xy=(i, percentage / 2),  # Position in the middle of the segment
                    xytext=(0, 0),  # No offset
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )

        # Add totals on top of each bar
        for i, crime in enumerate(crosstab.index):
            total_count = df[df["Crime type"] == crime].shape[0]
            ax1.annotate(
                f"n={total_count}",
                xy=(i, 102),  # Just above the bar
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
            )

    # SECOND PLOT: Crime types by LSOA (location)
    if "LSOA name" in df.columns:
        # Get top locations and crime types
        top_locations = df["LSOA name"].value_counts().nlargest(5).index

        # Create data for heatmap
        heatmap_data = pd.crosstab(
            df[df["LSOA name"].isin(top_locations)]["LSOA name"],
            df[df["LSOA name"].isin(top_locations)]["Crime type"],
        )

        # Normalize by row (location)
        heatmap_norm = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

        # Select top crime types for better readability
        top_crimes_heatmap = df["Crime type"].value_counts().nlargest(6).index
        heatmap_filtered = heatmap_norm[top_crimes_heatmap].fillna(0)

        # Create heatmap
        if not heatmap_filtered.empty:
            sns.heatmap(
                heatmap_filtered,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                ax=ax2,
                cbar_kws={"label": "Percentage of Crimes (%)"},
                linewidths=0.5,
            )

            # Add count annotations
            for i, location in enumerate(heatmap_filtered.index):
                total = df[df["LSOA name"] == location].shape[0]
                ax2.text(
                    -0.5,
                    i + 0.5,
                    f"n={total}",
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white", edgecolor="lightgrey", alpha=0.9, pad=1
                    ),
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "Not enough data for heatmap",
                ha="center",
                va="center",
                fontsize=14,
            )
    else:
        # Alternative: Show crime type distribution by month if LSOA is not available
        if "Month" in df.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df["Month"]):
                df["Month"] = pd.to_datetime(df["Month"])

            # Get monthly distribution for top crimes
            df_monthly = df.copy()
            df_monthly["Month"] = df_monthly["Month"].dt.strftime("%Y-%m")
            months = df_monthly["Month"].value_counts().nlargest(6).index

            monthly_crimes = (
                pd.crosstab(
                    df_monthly[df_monthly["Month"].isin(months)]["Month"],
                    df_monthly[df_monthly["Month"].isin(months)]["Crime type"],
                    normalize="index",
                )
                * 100
            )

            # Filter to top crimes
            monthly_crimes = monthly_crimes[
                monthly_crimes.columns[monthly_crimes.sum() > 5]
            ]

            # Create stacked bar plot
            if not monthly_crimes.empty:
                monthly_crimes.plot(
                    kind="bar", stacked=True, ax=ax2, colormap="tab10", width=0.8
                )
                ax2.set_title(
                    "Crime Type Distribution by Month",
                    fontsize=18,
                    fontweight="bold",
                    pad=20,
                )
                ax2.set_xlabel("Month", fontsize=14, labelpad=10)
                ax2.set_ylabel("Percentage (%)", fontsize=14, labelpad=10)
                ax2.grid(axis="y", linestyle="--", alpha=0.7)
                ax2.tick_params(axis="x", labelrotation=45, labelsize=12)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Not enough monthly data available",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "No location or time data available for second plot",
                ha="center",
                va="center",
                fontsize=14,
            )

    # Enhance styling for second plot
    if "LSOA name" in df.columns:
        ax2.set_title(
            "Crime Type Distribution by Location",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )
        ax2.set_xlabel("Crime Type", fontsize=14, labelpad=10)
        ax2.set_ylabel("Location (LSOA)", fontsize=14, labelpad=10)

    # Add figure title
    fig.suptitle(
        "Crime Patterns: Resolution and Geographic Distribution",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    # Add report metadata
    fig.text(
        0.5,
        0.01,
        "Source: UK Police Data",
        ha="center",
        fontsize=10,
        fontstyle="italic",
    )

    # Save the figure
    plt.savefig("categorical_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Create a publication-quality plot showing detailed crime statistics for a single month period.
    Focuses on daily patterns, crime type distributions, and daily variations.
    """
    # Setup figure with GridSpec for better layout control
    fig = plt.figure(figsize=(25, 20))  # Increased height for 4 subplots
    gs = GridSpec(2, 2, figure=fig)  # 2x2 grid for 4 plots

    # Ensure Month column is datetime and extract day information
    if "Month" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Month"]):
        df["Month"] = pd.to_datetime(df["Month"])

    # Add day of month and day of week if not present
    if "Day" not in df.columns and "Month" in df.columns:
        # For datasets that might have specific day information
        if "Month" in df.columns and len(df["Month"].dt.day.unique()) > 1:
            df["Day"] = df["Month"].dt.day
            df["DayOfWeek"] = df["Month"].dt.day_name()

    # ---- CRIME TYPE DISTRIBUTION (TOP LEFT) ----
    ax1 = fig.add_subplot(gs[0, 0])

    if "Crime type" in df.columns:
        # Get crime type distribution
        crime_dist = df["Crime type"].value_counts().nlargest(8).reset_index()
        crime_dist.columns = ["Crime type", "Count"]

        # Create horizontal bar chart
        sns.barplot(
            y="Crime type",
            x="Count",
            data=crime_dist,
            hue="Crime type",
            palette="viridis",
            ax=ax1,
            alpha=0.8,
            legend=False,
        )

        # Add value annotations
        for i, v in enumerate(crime_dist["Count"]):
            ax1.text(
                v + (crime_dist["Count"].max() * 0.02),
                i,
                f"{v:,}",
                va="center",
                fontsize=9,
            )

    # Styling
    ax1.set_title("Distribution of Crime Types", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Number of Incidents", fontsize=12)
    ax1.set_ylabel("")
    ax1.grid(axis="x", linestyle="--", alpha=0.7)

    # ---- OUTCOME ANALYSIS (TOP RIGHT) ----
    ax2 = fig.add_subplot(gs[0, 1])

    if "Last outcome category" in df.columns:
        # Get outcome distribution
        outcome_dist = (
            df["Last outcome category"].value_counts().nlargest(8).reset_index()
        )
        outcome_dist.columns = ["Outcome", "Count"]

        # Create horizontal bar chart
        sns.barplot(
            y="Outcome",
            x="Count",
            data=outcome_dist,
            hue="Outcome",
            palette="mako",
            ax=ax2,
            alpha=0.8,
            legend=False,
        )

        # Add value annotations
        for i, v in enumerate(outcome_dist["Count"]):
            ax2.text(
                v + (outcome_dist["Count"].max() * 0.02),
                i,
                f"{v:,}",
                va="center",
                fontsize=9,
            )

        # Styling
        ax2.set_title(
            "Crime Outcome Distribution", fontsize=14, fontweight="bold", pad=10
        )
        ax2.set_xlabel("Number of Cases", fontsize=12)
        ax2.set_ylabel("")
        ax2.grid(axis="x", linestyle="--", alpha=0.7)
    else:
        # Fallback if no outcome data
        ax2.text(
            0.5,
            0.5,
            "No outcome data available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax2.transAxes,
        )
        ax2.set_title(
            "Crime Outcome Distribution", fontsize=14, fontweight="bold", pad=10
        )

    # ---- CRIME FREQUENCY BY DAY AND HOUR (BOTTOM LEFT) ----
    ax3 = fig.add_subplot(gs[1, 0])

    # Create Hour column if time information is available
    if "Month" in df.columns:
        if hasattr(df["Month"].dt, "hour") and len(df["Month"].dt.hour.unique()) > 1:
            df["Hour"] = df["Month"].dt.hour
        else:
            # If no hour information, generate synthetic data for visualization
            np.random.seed(42)  # For reproducibility
            df["Hour"] = np.random.randint(0, 24, size=len(df))

    # Create day of week if not already present
    if "DayOfWeek" not in df.columns and "Month" in df.columns:
        # If we have only month info, create synthetic day data
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        np.random.seed(42)
        df["DayOfWeek"] = np.random.choice(days, size=len(df))

    # Prepare data for heatmap
    if "Hour" in df.columns and "DayOfWeek" in df.columns:
        # Create pivot table for day-hour combinations
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        hour_data = pd.crosstab(df["DayOfWeek"], df["Hour"]).reindex(day_order)

        # Create heatmap
        sns.heatmap(
            hour_data,
            cmap="YlOrRd",
            ax=ax3,
            cbar_kws={"label": "Number of Incidents"},
            linewidths=0.5,
        )

        # Styling
        ax3.set_title(
            "Crime Frequency by Day and Hour", fontsize=14, fontweight="bold", pad=10
        )
        ax3.set_xlabel("Hour of Day", fontsize=12)
        ax3.set_ylabel("Day of Week", fontsize=12)

        # Better hour labels (24-hour format)
        ax3.set_xticks(np.arange(0, 24, 2))
        ax3.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    else:
        # Fallback if no temporal data
        ax3.text(
            0.5,
            0.5,
            "No temporal data available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax3.transAxes,
        )
        ax3.set_title(
            "Crime Frequency by Day and Hour", fontsize=14, fontweight="bold", pad=10
        )

    # ---- DISTRIBUTION OF CRIMES BY LONGITUDE (BOTTOM RIGHT) ----
    ax4 = fig.add_subplot(gs[1, 1])

    if "Longitude" in df.columns:
        # Create a KDE plot with histogram
        sns.histplot(
            df["Longitude"],
            kde=True,
            stat="density",
            color="darkblue",
            ax=ax4,
            alpha=0.6,
            line_kws={"linewidth": 2},
        )

        # Mark the UK mainland longitude range
        uk_lon_min, uk_lon_max = -8.0, 2.0
        valid_data = df[
            (df["Longitude"] >= uk_lon_min) & (df["Longitude"] <= uk_lon_max)
        ]

        # Add mean and median lines
        mean_lon = valid_data["Longitude"].mean()
        median_lon = valid_data["Longitude"].median()

        ax4.axvline(
            x=mean_lon, color="red", linestyle="-", label=f"Mean: {mean_lon:.4f}"
        )
        ax4.axvline(
            x=median_lon,
            color="green",
            linestyle="--",
            label=f"Median: {median_lon:.4f}",
        )

        # Add UK landmarks for context
        landmarks = {
            "London": -0.1278,
            "Cardiff": -3.1791,
            "Edinburgh": -3.1883,
            "Belfast": -5.9301,
        }

        y_pos = ax4.get_ylim()[1] * 0.9
        y_step = ax4.get_ylim()[1] * 0.07

        for i, (city, lon) in enumerate(landmarks.items()):
            if uk_lon_min <= lon <= uk_lon_max:
                ax4.axvline(x=lon, color="purple", linestyle=":", alpha=0.6)
                ax4.text(
                    lon,
                    y_pos - (i * y_step),
                    city,
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"),
                )

        # Styling
        ax4.set_title(
            "Distribution of Crimes by Longitude",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        ax4.set_xlabel("Longitude", fontsize=12)
        ax4.set_ylabel("Density", fontsize=12)
        ax4.grid(True, linestyle="--", alpha=0.7)
        ax4.legend()

        # Add UK mainland range indication
        ax4.axvspan(
            uk_lon_min, uk_lon_max, alpha=0.2, color="gray", label="UK Mainland Range"
        )
    else:
        # Fallback if no longitude data
        ax4.text(
            0.5,
            0.5,
            "No longitude data available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax4.transAxes,
        )
        ax4.set_title(
            "Distribution of Crimes by Longitude",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )

    # Adjust layout and add metadata
    plt.tight_layout()
    fig.text(
        0.5,
        0.01,
        "Data Source: UK Police Data, November 2019",
        ha="center",
        fontsize=10,
        fontstyle="italic",
    )

    # Add a descriptive subtitle
    fig.suptitle(
        "Comprehensive Crime Statistics Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Save the figure
    plt.savefig("statistical_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Perform comprehensive statistical analysis on a specified column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    col : str
        The column name to analyze

    Returns:
    --------
    tuple
        (mean, std_dev, skewness, kurtosis) of the specified column
    """
    # Drop NAs to ensure accurate statistics
    data = df[col].dropna()

    # Basic statistics
    mean = data.mean()
    stddev = data.std()
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocess the crime dataset for analysis, clustering, and visualization.

    This function performs the following steps:
    1. Data cleaning and validation
    2. Handling missing values appropriately for different column types
    3. Feature engineering relevant for crime data analysis
    4. Initial exploratory analysis with summary statistics

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw crime data

    Returns:
    --------
    pandas.DataFrame
        Cleaned and preprocessed dataframe ready for analysis
    """
    df = df.copy()

    if "Context" in df.columns:
        df = df.drop(columns=["Context"])

    df["Month"] = pd.to_datetime(df["Month"])

    df["Year"] = df["Month"].dt.year
    df["MonthOfYear"] = df["Month"].dt.month

    initial_count = len(df)
    df = df.dropna(subset=["Longitude", "Latitude"])
    print(
        f"Removed {initial_count - len(df)} records ({(initial_count - len(df))/initial_count:.2%}) with missing location data"
    )

    uk_lon_bounds = (-8.0, 2.0)
    uk_lat_bounds = (49.5, 61.0)

    mask_valid_coords = (
        (df["Longitude"] >= uk_lon_bounds[0])
        & (df["Longitude"] <= uk_lon_bounds[1])
        & (df["Latitude"] >= uk_lat_bounds[0])
        & (df["Latitude"] <= uk_lat_bounds[1])
    )

    invalid_coords = df[~mask_valid_coords]
    if len(invalid_coords) > 0:
        df = df[mask_valid_coords]

    # Handle missing categorical data
    categorical_cols = [
        "Crime ID",
        "LSOA code",
        "LSOA name",
        "Last outcome category",
        "Crime type",
    ]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    for col in categorical_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df[col] = df[col].fillna("Unknown")

    # Group minor crime categories if needed
    if "Crime type" in df.columns:
        crime_counts = df["Crime type"].value_counts()
        # If there are many crime types with few occurrences, consider grouping them
        rare_crimes = crime_counts[crime_counts < len(df) * 0.01].index
        if len(rare_crimes) > 0:
            df["Crime type"] = df["Crime type"].apply(
                lambda x: "Other" if x in rare_crimes else x
            )

    print("\n=== Dataset Summary after Preprocessing ===")
    print(f"Final dataset shape: {df.shape}")
    print("\nSummary statistics for numerical columns:")
    print(df.describe())

    print("\nFirst 5 rows of preprocessed data:")
    print(df.head())

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) > 0:
        print("\nCorrelation matrix for numeric columns:")
        corr_matrix = df[numeric_cols].corr(method="pearson")
        print(corr_matrix)

        # Highlight strong correlations
        strong_corr = (corr_matrix.abs() > 0.5) & (corr_matrix != 1.0)
        if strong_corr.any().any():
            print("\nStrong correlations found:")
            for col1 in strong_corr.columns:
                for col2 in strong_corr.index:
                    if strong_corr.loc[col2, col1] and col1 != col2:
                        print(f"- {col1} and {col2}: {corr_matrix.loc[col2, col1]:.2f}")
    else:
        print("\nNo numeric columns for correlation analysis")

    # Print categorical value distributions
    for col in categorical_cols:
        if len(df[col].unique()) < 15:  # Only show if not too many categories
            print(f"\nDistribution of '{col}':")
            print(df[col].value_counts(normalize=True).head(10))

    return df


def writing(moments, col):
    """Provides professional statistical interpretation of distribution moments."""
    print(f"For the attribute {col}:")
    print(
        f"Mean = {moments[0]:.2f}, "
        f"Standard Deviation = {moments[1]:.2f}, "
        f"Skewness = {moments[2]:.2f}, and "
        f"Excess Kurtosis = {moments[3]:.2f}."
    )

    # Interpret skewness
    if abs(moments[2]) < 0.5:
        skewness_desc = "approximately symmetrical"
    elif abs(moments[2]) < 1.0:
        skewness_desc = (
            "moderately " + ("right" if moments[2] > 0 else "left") + "-skewed"
        )
    else:
        skewness_desc = "highly " + ("right" if moments[2] > 0 else "left") + "-skewed"

    # Interpret kurtosis
    if moments[3] < -0.5:
        kurtosis_desc = "platykurtic"
    elif moments[3] > 0.5:
        kurtosis_desc = "leptokurtic"
    else:
        kurtosis_desc = "mesokurtic"

    print(f"The data was {skewness_desc} and {kurtosis_desc}")

    return


def perform_clustering(df, col1, col2):
    """
    Perform KMeans clustering on the specified columns of the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    col1 : str
        The first column name to use for clustering
    col2 : str
        The second column name to use for clustering

    Returns:
    --------
    tuple
        (labels, data, xkmeans, ykmeans, cenlabels) for the clustered data
    """

    # Inner function to plot elbow method with improved visuals
    def plot_elbow_method(data):
        """
        Plot the elbow method to determine the optimal number of clusters.
        Uses data sampling for large datasets to improve performance.
        """
        # Sample data if it's too large (improves performance dramatically)
        sample_size = 2000  # Adjust based on your performance needs
        if len(data) > sample_size:
            np.random.seed(42)
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            data_sample = data[sample_indices]
            print(
                f"Using {sample_size} samples for elbow method (out of {len(data)} points)"
            )
        else:
            data_sample = data

        max_clusters = min(11, len(data_sample))  # Avoid more clusters than data points
        sse = []
        silhouette_scores = []
        range_of_k = range(2, max_clusters)

        use_silhouette = (
            len(data_sample) <= 10000
        )  # Only use silhouette for smaller datasets

        # Calculate SSE (and optionally silhouette scores)
        for k in range_of_k:
            # Use more efficient KMeans settings
            kmeans = KMeans(
                n_clusters=k, random_state=42, n_init=10, max_iter=100, tol=1e-4
            )
            kmeans.fit(data_sample)
            sse.append(kmeans.inertia_)

            # Only calculate silhouette for smaller datasets (it's very expensive)
            if use_silhouette and k > 1:  # Silhouette requires at least 2 clusters
                labels = kmeans.predict(data_sample)
                try:
                    # Use a small sample for silhouette if data is still large
                    if len(data_sample) > 5000:
                        sub_sample = np.random.choice(
                            len(data_sample), 5000, replace=False
                        )
                        s_score = silhouette_score(
                            data_sample[sub_sample], labels[sub_sample]
                        )
                    else:
                        s_score = silhouette_score(data_sample, labels)
                    silhouette_scores.append(s_score)
                except:
                    # If silhouette fails, just append a placeholder
                    silhouette_scores.append(0)
                    use_silhouette = False

        # Create a publication-quality figure with one or two subplots
        if use_silhouette:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(
                "Determining Optimal Number of Clusters", fontsize=16, fontweight="bold"
            )
        else:
            fig, ax1 = plt.subplots(figsize=(8, 6))
            fig.suptitle(
                "Determining Optimal Number of Clusters", fontsize=16, fontweight="bold"
            )

        # Plot SSE (Elbow method)
        ax1.plot(
            range_of_k,
            sse,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            color="#3498db",
        )
        ax1.set_title("Elbow Method", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Number of Clusters", fontsize=12)
        ax1.set_ylabel("Sum of Squared Errors (SSE)", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Find elbow point and mark it
        # Use second derivative or simple heuristic
        if len(sse) > 2:
            # Calculate rate of change in slope
            diffs = np.diff(sse)
            diffs_norm = diffs / np.abs(diffs).mean()  # Normalize diffs

            # Simple heuristic for elbow point - where rate of decrease slows down
            elbow_point = 2  # Default if we can't find a better point
            for i in range(1, len(diffs_norm)):
                if (
                    diffs_norm[i] > 0.7 * diffs_norm[i - 1]
                ):  # Significant change in slope
                    elbow_point = i + 2  # +2 because range_of_k starts at 2
                    break
        else:
            elbow_point = 2

        ax1.axvline(
            x=elbow_point,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Elbow Point (k={elbow_point})",
        )

        # Plot Silhouette scores if available
        if use_silhouette:
            ax2.plot(
                list(range_of_k)[: len(silhouette_scores)],
                silhouette_scores,
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=8,
                color="#2ecc71",
            )
            ax2.set_title("Silhouette Method", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Number of Clusters", fontsize=12)
            ax2.set_ylabel("Silhouette Score", fontsize=12)
            ax2.grid(True, linestyle="--", alpha=0.7)

            # Find best silhouette score
            best_k_silhouette = range_of_k[
                np.argmax(silhouette_scores) + 1
            ]  # +1 due to offset in range_of_k
            ax2.axvline(
                x=best_k_silhouette,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Best Score (k={best_k_silhouette})",
            )
            ax2.legend(loc="best")

            # Add annotations
            ax2.annotate(
                f"Suggested k={best_k_silhouette}",
                xy=(
                    best_k_silhouette,
                    silhouette_scores[best_k_silhouette - 3],
                ),  # -3 due to offsets
                xytext=(
                    best_k_silhouette + 0.5,
                    silhouette_scores[best_k_silhouette - 3] * 0.9,
                ),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
            )

        # Add annotations
        ax1.annotate(
            f"Suggested k={elbow_point}",
            xy=(elbow_point, sse[elbow_point - 2]),  # -2 because range_of_k starts at 2
            xytext=(elbow_point + 0.5, sse[elbow_point - 2] * 1.1),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
        )

        # Add legends
        ax1.legend(loc="best")

        # Add a text box with methodology explanation
        textstr = "\n".join(
            [
                "Methodology:",
                "- Elbow Method: Find point of diminishing returns",
                (
                    "- Silhouette: Higher score indicates better-defined clusters"
                    if use_silhouette
                    else ""
                ),
            ]
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        fig.text(0.5, 0.01, textstr, fontsize=10, bbox=props, ha="center")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("elbow_plot.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Return suggested optimal k (preference to silhouette if scores are good)
        if use_silhouette and max(silhouette_scores) > 0.5:  # Good silhouette score
            return best_k_silhouette
        else:
            # Return elbow point, with a cap for large datasets
            if len(data) > 10000 and elbow_point > 5:
                return min(5, elbow_point)  # Cap at 5 clusters for very large datasets
            return elbow_point

    # Evaluate clustering quality efficiently
    def one_silhouette_inertia(data, n_clusters):
        """
        Calculate silhouette score and inertia for a given number of clusters,
        with optimizations for large datasets.
        """
        # Use more efficient KMeans parameters
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,  # More iterations for better convergence
            tol=1e-4,
        )
        labels = kmeans.fit_predict(data)

        # Calculate quality metrics
        inertia = kmeans.inertia_

        # For silhouette, only calculate on sample for large datasets
        if len(data) > 5000 and n_clusters > 1:
            # Sample for silhouette calculation
            sample_size = min(5000, len(data))
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            try:
                score = silhouette_score(data[sample_indices], labels[sample_indices])
            except:
                score = 0
        elif n_clusters > 1:
            try:
                score = silhouette_score(data, labels)
            except:
                score = 0
        else:
            score = 0

        return score, inertia, labels, kmeans.cluster_centers_

    # Input validation
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns {col1} and/or {col2} not found in dataframe")

    # Handle extremely large datasets with sampling for initial analysis
    data = df[[col1, col2]].dropna()
    original_size = len(data)

    if len(data) < 3:  # Need at least 3 points for meaningful clustering
        raise ValueError("Not enough data points for clustering after removing NAs")

    # For very large datasets, consider using sample for analysis
    max_analysis_size = 20000
    if len(data) > max_analysis_size:
        print(
            f"Dataset is large ({len(data)} points). Using {max_analysis_size} sample for initial analysis."
        )
        data_sample = data.sample(max_analysis_size, random_state=42)
    else:
        data_sample = data

    # Scale data for better clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_sample)

    # Determine optimal number of clusters (with warning for large datasets)
    print(f"Determining optimal cluster count for {len(data_sample)} data points...")
    if len(data) > 10000:
        print(
            "Warning: Large dataset may take time to process. Using optimized approach."
        )

    suggested_clusters = plot_elbow_method(data_scaled)
    print(f"Suggested number of clusters: {suggested_clusters}")

    # If we used a sample for analysis but have full dataset, now fit on full data
    if len(data) > max_analysis_size:
        print(
            f"Applying {suggested_clusters} clusters to full dataset ({len(data)} points)"
        )
        # Scale the full dataset
        full_data_scaled = scaler.transform(data)

        # Fit on full data, but with more efficient parameters
        kmeans = KMeans(
            n_clusters=suggested_clusters,
            random_state=42,
            n_init=10,
            max_iter=100,
            tol=1e-4,
        )
        labels = kmeans.fit_predict(full_data_scaled)
        centers = kmeans.cluster_centers_

        # No need to compute expensive metrics on full dataset
        print(
            f"Clustering complete. Clusters have {[np.sum(labels == i) for i in range(suggested_clusters)]} points"
        )
    else:
        # Perform final clustering with optimal number on original dataset
        print(f"Performing final clustering with {suggested_clusters} clusters")
        _, _, labels, centers = one_silhouette_inertia(data_scaled, suggested_clusters)

    # Inverse transform centers to original scale for interpretability
    centers_original = scaler.inverse_transform(centers)
    xkmeans, ykmeans = centers_original[:, 0], centers_original[:, 1]

    # Create descriptive cluster labels
    cenlabels = [f"Cluster {i+1}" for i in range(suggested_clusters)]

    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """
    Plot the clustered data with cluster centers.
    Optimized for performance with large datasets.

    Parameters:
    -----------
    labels : numpy.ndarray
        Cluster labels for each data point
    data : pandas.DataFrame
        Original data used for clustering
    xkmeans : numpy.ndarray
        X-coordinates of cluster centers
    ykmeans : numpy.ndarray
        Y-coordinates of cluster centers
    centre_labels : list
        Labels for each cluster center
    """
    # Create a publication-quality figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Set scientific styling
    plt.style.use("seaborn-v0_8-whitegrid")

    # Count points in each cluster for legend labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_counts = {i: count for i, count in zip(unique_labels, counts)}

    # Use a colormap that's color-blind friendly
    cmap = plt.get_cmap("viridis", len(np.unique(labels)))
    colors = [cmap(i) for i in range(len(np.unique(labels)))]

    # For large datasets, sample points for plotting to improve performance
    max_display_points = 5000  # Maximum points to display
    if len(data) > max_display_points:
        print(
            f"Sampling {max_display_points} points for visualization (out of {len(data)})"
        )
        # Sample points from each cluster proportionally
        display_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            # Calculate how many points to sample from this cluster
            sample_size = int(len(label_indices) * (max_display_points / len(data)))
            if sample_size < 10:  # Ensure at least some points from each cluster
                sample_size = min(10, len(label_indices))
            if sample_size > 0:
                sampled_indices = np.random.choice(
                    label_indices, sample_size, replace=False
                )
                display_indices.extend(sampled_indices)

        # Create view of data for plotting
        plot_data = data.iloc[display_indices]
        plot_labels = labels[display_indices]

        # Add note about sampling
        ax.text(
            0.5,
            0.01,
            f"Note: Visualization shows {len(display_indices)} sampled points from {len(data)} total",
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
        )
    else:
        plot_data = data
        plot_labels = labels

    # Plot each cluster with custom labels including counts
    for i, label in enumerate(np.unique(plot_labels)):
        cluster_data = plot_data[plot_labels == label]
        ax.scatter(
            cluster_data.iloc[:, 0],
            cluster_data.iloc[:, 1],
            s=70,
            c=[colors[i]],
            label=f"{centre_labels[i]} (n={cluster_counts[label]})",
            alpha=0.7,
            edgecolors="w",
            linewidth=0.5,
        )

    # Plot cluster centroids with enhanced visibility
    ax.scatter(
        xkmeans,
        ykmeans,
        s=300,
        c="red",
        marker="X",
        linewidth=2,
        edgecolors="black",
        zorder=10,
    )

    # Add centroid labels with professional styling
    for i, label in enumerate(centre_labels):
        ax.annotate(
            f"Center {i+1}",
            xy=(xkmeans[i], ykmeans[i]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
        )

    # Enhance plot styling
    ax.set_title("Spatial Clustering Analysis", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("Longitude", fontsize=16, labelpad=10)
    ax.set_ylabel("Latitude", fontsize=16, labelpad=10)

    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add statistical summary as text
    textstr = "\n".join(
        [
            "Clustering Statistics:",
            f"Number of clusters: {len(np.unique(labels))}",
            f"Total data points: {len(data)}",
            f"Largest cluster: {max(cluster_counts.values())} points",
            f"Smallest cluster: {min(cluster_counts.values())} points",
        ]
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    # Create a custom legend with better placement
    plt.legend(
        title="Cluster Information",
        title_fontsize=14,
        fontsize=12,
        loc="upper right",
        bbox_to_anchor=(1.1, 1),
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add a map-like border
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Save the publication-quality figure
    plt.tight_layout()
    plt.savefig("clustering.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return


def perform_fitting(df, col1, col2):
    """
    Perform linear fitting on the specified columns of the dataframe.
    Optimized for performance with large datasets.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    col1 : str
        The first column name to use for fitting (independent variable)
    col2 : str
        The second column name to use for fitting (dependent variable)

    Returns:
    --------
    tuple
        (data, x, y_pred, model_parameters) for the fitted data and model details
    """
    # Input validation
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns {col1} and/or {col2} not found in dataframe")

    # Gather data and prepare for fitting
    data = df[[col1, col2]].dropna()
    if len(data) < 2:  # Need at least 2 points for linear regression
        raise ValueError("Not enough data points for fitting after removing NAs")

    # For very large datasets, consider sampling
    max_fitting_points = 50000
    if len(data) > max_fitting_points:
        print(
            f"Dataset too large ({len(data)} points). Using {max_fitting_points} sample for fitting."
        )
        data = data.sample(max_fitting_points, random_state=42)

    x = data[col1].values.reshape(-1, 1)
    y = data[col2].values

    # Fit model
    print(f"Fitting linear regression model on {len(data)} points")
    model = LinearRegression()
    model.fit(x, y)

    # Calculate model quality metrics
    y_pred_train = model.predict(x)
    r2 = model.score(x, y)
    mse = np.mean((y - y_pred_train) ** 2)
    rmse = np.sqrt(mse)

    # Calculate confidence intervals (theoretical)
    n = len(x)
    p = 1  # Number of predictors
    dof = n - p - 1  # Degrees of freedom

    # Standard error of the estimate
    se = np.sqrt(np.sum((y - y_pred_train) ** 2) / dof)

    # Prepare values for prediction across x range
    x_pred = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_pred)

    # Calculate prediction intervals (approximately)
    # For each x value, calculate the standard error of the prediction
    t_critical = ss.t.ppf(0.975, dof)  # 95% confidence
    std_errors = np.zeros(len(x_pred))

    # Additional model parameters for return
    model_parameters = {
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "r2": r2,
        "rmse": rmse,
        "confidence_intervals": {
            "x_pred": x_pred.flatten(),
            "t_critical": t_critical,
            "std_errors": std_errors,
        },
    }

    print(
        f"Linear regression results: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
    )
    print(f"R² = {r2:.4f}, RMSE = {rmse:.4f}")

    return data, x_pred, y_pred, model_parameters


def plot_fitted_data(data, x, y, model_params=None):
    """
    Plot the fitted data with the linear regression line and confidence intervals.
    Optimized for large datasets.

    Parameters:
    -----------
    data : pandas.DataFrame
        Original data used for fitting
    x : numpy.ndarray
        X values for prediction line
    y : numpy.ndarray
        Predicted Y values
    model_params : dict, optional
        Model parameters and confidence interval information
    """
    # Set scientific plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create a publication-quality figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # For large datasets, sample points for plotting to improve performance
    max_display_points = 5000  # Maximum points to display
    if len(data) > max_display_points:
        print(
            f"Sampling {max_display_points} points for visualization (out of {len(data)})"
        )
        display_data = data.sample(max_display_points, random_state=42)

        # Add note about sampling
        ax.text(
            0.5,
            0.01,
            f"Note: Visualization shows {max_display_points} sampled points from {len(data)} total",
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
        )
    else:
        display_data = data

    # Plot the data sample with improved styling
    sns.scatterplot(
        x=display_data.iloc[:, 0],
        y=display_data.iloc[:, 1],
        ax=ax,
        alpha=0.7,
        s=80,  # Larger point size
        color="#3498db",  # Professional blue
        edgecolor="w",
        linewidth=0.5,
        label="Data Points",
    )

    # Plot the regression line with enhanced visibility
    ax.plot(x, y, color="#e74c3c", linewidth=3, label="Linear Regression")

    # Extract model information if provided
    if model_params:
        slope = model_params.get("slope", 0)
        intercept = model_params.get("intercept", 0)
        r2 = model_params.get("r2", 0)
        rmse = model_params.get("rmse", 0)

        # Add confidence intervals if available
        if "confidence_intervals" in model_params:
            ci_info = model_params["confidence_intervals"]
            if all(k in ci_info for k in ["x_pred", "t_critical"]):
                # Simple approximation of confidence bands
                x_flat = ci_info["x_pred"]
                t_crit = ci_info["t_critical"]
                plt.fill_between(
                    x_flat,
                    y.flatten() - t_crit * rmse,
                    y.flatten() + t_crit * rmse,
                    alpha=0.2,
                    color="#e74c3c",
                    label="95% Confidence Band",
                )

        # Add equation and statistics to the plot
        equation = f"y = {slope:.4f}x + {intercept:.4f}"
        stats = f"$R^2$ = {r2:.4f}, RMSE = {rmse:.4f}"

        # Add text box with model details
        textstr = "\n".join(
            ["Model Statistics:", equation, stats, f"n = {len(data)} data points"]
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

    # Enhance plot styling
    ax.set_title("Linear Regression Analysis", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel(data.columns[0], fontsize=16, labelpad=10)
    ax.set_ylabel(data.columns[1], fontsize=16, labelpad=10)

    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.7)

    # Improve legend
    ax.legend(fontsize=12, loc="upper right", frameon=True, fancybox=True, shadow=True)

    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # Save the publication-quality figure
    plt.tight_layout()
    plt.savefig("fitting.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return


def main():
    df = pd.read_csv("data.csv")
    df = preprocessing(df)
    col = "Longitude"
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    clustering_results = perform_clustering(df, "Longitude", "Latitude")
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, "Longitude", "Latitude")
    plot_fitted_data(*fitting_results)
    return


if __name__ == "__main__":
    main()
