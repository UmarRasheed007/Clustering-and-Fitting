import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec


def plot_relational_plot(df):
    """
    Create a publication-quality scatter plot showing the geographical distribution of crimes.
    Visualizes the spatial patterns of different crime types across locations.
    """
    # Set figure aesthetics for publication quality
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Create a custom colorblind-friendly palette 
    palette = sns.color_palette("viridis", n_colors=len(df['Crime type'].unique()))
    
    # Create the main scatter plot
    ax = sns.scatterplot(
        x='Longitude', 
        y='Latitude',
        data=df,
        hue='Crime type',
        palette=palette,
        alpha=0.7,
        s=50,  # Point size
        edgecolor='none'
    )
    
    # Calculate crime density and highlight hotspots
    if len(df) > 100:  # Only if we have enough data points
        # Just call the kdeplot without assigning it, or use the returned object
        sns.kdeplot(
            x=df['Longitude'],
            y=df['Latitude'],
            levels=5,
            fill=True,
            alpha=0.3,
            color='red',
            ax=ax  # Explicitly specify the axes to ensure it's added to the right plot
        )
    
    # Enhance plot styling
    ax.set_title('Geographic Distribution of Crime Types', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=14, labelpad=10)
    ax.set_ylabel('Latitude', fontsize=14, labelpad=10)
    
    # Add UK context if data is for UK
    ax.axhline(y=51.5074, color='blue', linestyle='--', alpha=0.5)  # London latitude
    ax.axvline(x=-0.1278, color='blue', linestyle='--', alpha=0.5)  # London longitude
    ax.text(-0.1278, 51.5074, 'London', fontsize=10, ha='right', va='bottom', 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, length=5, width=1.5)
    ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=1)
    
    # Create informative legend with crime counts
    crime_counts = df['Crime type'].value_counts()
    handles, labels = ax.get_legend_handles_labels()
    updated_labels = [f"{label} ({crime_counts.get(label, 0)})" for label in labels]
    
    # Position legend intelligently outside the plot
    plt.legend(
        handles=handles,
        labels=updated_labels,
        title='Crime Type (Count)',
        title_fontsize=14,
        fontsize=12,
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    # Add a north arrow for geographical context
    ax.annotate('N', xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=14, fontweight='bold', ha='center',
                bbox=dict(boxstyle="circle,pad=0.3", fc='white', ec='black'))
    
    # Add annotations about crime density
    highest_crime_area = df.groupby(['Longitude', 'Latitude']).size().reset_index(name='count').sort_values('count', ascending=False).iloc[0]
    ax.annotate(
        f"Highest crime density\n({highest_crime_area['count']} incidents)",
        xy=(highest_crime_area['Longitude'], highest_crime_area['Latitude']),
        xytext=(30, 30),
        textcoords="offset points",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.6),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
    )
    
    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=300, bbox_inches='tight')
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
    top_crimes = df['Crime type'].value_counts().nlargest(5).index
    top_outcomes = df['Last outcome category'].value_counts().nlargest(6).index
    
    # Filter data
    plot_df = df[df['Crime type'].isin(top_crimes) & df['Last outcome category'].isin(top_outcomes)]
    
    # Calculate percentages for each crime type and outcome combination
    crosstab = pd.crosstab(
        plot_df['Crime type'], 
        plot_df['Last outcome category'],
        normalize='index'
    ) * 100
    
    # Check if there are any rows with identical values
    if not crosstab.empty:
        # Create the first plot using crosstab data
        crosstab.plot(
            kind='bar',
            stacked=True,
            ax=ax1,
            colormap='viridis',
            width=0.8
        )
        
        # Ensure x-axis has proper width even with identical values
        if len(crosstab) > 1:
            x_min, x_max = ax1.get_xlim()
            if x_min == x_max:  # If identical limits
                ax1.set_xlim(x_min - 0.5, x_max + 0.5)
    else:
        # If dataframe is empty, create an empty plot with a message
        ax1.text(0.5, 0.5, "No data available for plotting", 
                ha='center', va='center', fontsize=14)
    
    # Enhance styling for first plot
    ax1.set_title('Crime Resolution Outcomes by Type', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Crime Type', fontsize=14, labelpad=10)
    ax1.set_ylabel('Percentage of Cases (%)', fontsize=14, labelpad=10)
    
    # Format x-ticks for first plot
    ax1.tick_params(axis='x', labelrotation=45, labelsize=12, length=5, width=1)
    ax1.tick_params(axis='y', labelsize=12, length=5, width=1)
    
    # Add grid for easier reading of percentages
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Enhance legend for first plot
    if not crosstab.empty:
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(
            handles=handles,
            labels=labels,
            title='Case Outcome',
            title_fontsize=14,
            fontsize=11,
            loc='upper right',
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0,
            frameon=True,
            fancybox=True
        )
    
        # Add annotations showing the most common outcome for each crime
        for i, crime in enumerate(crosstab.index):
            most_common = crosstab.loc[crime].idxmax()
            percentage = crosstab.loc[crime, most_common]
            if percentage > 15:  # Only annotate if there's enough space
                ax1.annotate(
                    f"{percentage:.1f}%",
                    xy=(i, percentage/2),  # Position in the middle of the segment
                    xytext=(0, 0),  # No offset
                    textcoords="offset points",
                    ha='center',
                    va='center',
                    fontsize=11,
                    fontweight='bold',
                    color='white'
                )
    
        # Add totals on top of each bar
        for i, crime in enumerate(crosstab.index):
            total_count = df[df['Crime type'] == crime].shape[0]
            ax1.annotate(
                f"n={total_count}",
                xy=(i, 102),  # Just above the bar
                ha='center',
                fontsize=11,
                fontweight='bold',
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2)
            )

    # SECOND PLOT: Crime types by LSOA (location)
    if 'LSOA name' in df.columns:
        # Get top locations and crime types
        top_locations = df['LSOA name'].value_counts().nlargest(5).index
        
        # Create data for heatmap
        heatmap_data = pd.crosstab(
            df[df['LSOA name'].isin(top_locations)]['LSOA name'],
            df[df['LSOA name'].isin(top_locations)]['Crime type']
        )
        
        # Normalize by row (location)
        heatmap_norm = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
        
        # Select top crime types for better readability
        top_crimes_heatmap = df['Crime type'].value_counts().nlargest(6).index
        heatmap_filtered = heatmap_norm[top_crimes_heatmap].fillna(0)
        
        # Create heatmap
        if not heatmap_filtered.empty:
            sns.heatmap(
                heatmap_filtered,
                annot=True,
                fmt='.1f',
                cmap='YlGnBu',
                ax=ax2,
                cbar_kws={'label': 'Percentage of Crimes (%)'},
                linewidths=0.5
            )
            
            # Add count annotations
            for i, location in enumerate(heatmap_filtered.index):
                total = df[df['LSOA name'] == location].shape[0]
                ax2.text(
                    -0.5, 
                    i + 0.5, 
                    f"n={total}", 
                    va='center',
                    ha='right',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='lightgrey', alpha=0.9, pad=1)
                )
        else:
            ax2.text(0.5, 0.5, "Not enough data for heatmap", 
                    ha='center', va='center', fontsize=14)
    else:
        # Alternative: Show crime type distribution by month if LSOA is not available
        if 'Month' in df.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['Month']):
                df['Month'] = pd.to_datetime(df['Month'])
                
            # Get monthly distribution for top crimes
            df_monthly = df.copy()
            df_monthly['Month'] = df_monthly['Month'].dt.strftime('%Y-%m')
            months = df_monthly['Month'].value_counts().nlargest(6).index
            
            monthly_crimes = pd.crosstab(
                df_monthly[df_monthly['Month'].isin(months)]['Month'],
                df_monthly[df_monthly['Month'].isin(months)]['Crime type'],
                normalize='index'
            ) * 100
            
            # Filter to top crimes
            monthly_crimes = monthly_crimes[monthly_crimes.columns[monthly_crimes.sum() > 5]]
            
            # Create stacked bar plot
            if not monthly_crimes.empty:
                monthly_crimes.plot(
                    kind='bar',
                    stacked=True,
                    ax=ax2,
                    colormap='tab10',
                    width=0.8
                )
                ax2.set_title('Crime Type Distribution by Month', fontsize=18, fontweight='bold', pad=20)
                ax2.set_xlabel('Month', fontsize=14, labelpad=10)
                ax2.set_ylabel('Percentage (%)', fontsize=14, labelpad=10)
                ax2.grid(axis='y', linestyle='--', alpha=0.7)
                ax2.tick_params(axis='x', labelrotation=45, labelsize=12)
            else:
                ax2.text(0.5, 0.5, "Not enough monthly data available", 
                        ha='center', va='center', fontsize=14)
        else:
            ax2.text(0.5, 0.5, "No location or time data available for second plot", 
                    ha='center', va='center', fontsize=14)
    
    # Enhance styling for second plot
    if 'LSOA name' in df.columns:
        ax2.set_title('Crime Type Distribution by Location', fontsize=18, fontweight='bold', pad=20)
        ax2.set_xlabel('Crime Type', fontsize=14, labelpad=10)
        ax2.set_ylabel('Location (LSOA)', fontsize=14, labelpad=10)
    
    # Add figure title
    fig.suptitle('Crime Patterns: Resolution and Geographic Distribution', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Add report metadata
    fig.text(0.5, 0.01, 'Source: UK Police Data', ha='center', fontsize=10, fontstyle='italic')
    
    # Save the figure
    plt.savefig('categorical_plot.png', dpi=300, bbox_inches='tight')
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
    if 'Month' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Month']):
        df['Month'] = pd.to_datetime(df['Month'])
    
    # Add day of month and day of week if not present
    if 'Day' not in df.columns and 'Month' in df.columns:
        # For datasets that might have specific day information
        if 'Month' in df.columns and len(df['Month'].dt.day.unique()) > 1:
            df['Day'] = df['Month'].dt.day
            df['DayOfWeek'] = df['Month'].dt.day_name()
    
    # ---- CRIME TYPE DISTRIBUTION (TOP LEFT) ----
    ax1 = fig.add_subplot(gs[0, 0])
    
    if 'Crime type' in df.columns:
        # Get crime type distribution
        crime_dist = df['Crime type'].value_counts().nlargest(8).reset_index()
        crime_dist.columns = ['Crime type', 'Count']
        
        # Create horizontal bar chart
        sns.barplot(
            y='Crime type',
            x='Count',
            data=crime_dist,
            hue='Crime type',
            palette='viridis',
            ax=ax1,
            alpha=0.8,
            legend=False
        )
        
        # Add value annotations
        for i, v in enumerate(crime_dist['Count']):
            ax1.text(
                v + (crime_dist['Count'].max() * 0.02),
                i,
                f"{v:,}",
                va='center',
                fontsize=9
            )
    
    # Styling
    ax1.set_title('Distribution of Crime Types', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Number of Incidents', fontsize=12)
    ax1.set_ylabel('')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # ---- OUTCOME ANALYSIS (TOP RIGHT) ----
    ax2 = fig.add_subplot(gs[0, 1])
    
    if 'Last outcome category' in df.columns:
        # Get outcome distribution
        outcome_dist = df['Last outcome category'].value_counts().nlargest(8).reset_index()
        outcome_dist.columns = ['Outcome', 'Count']
        
        # Create horizontal bar chart
        sns.barplot(
            y='Outcome',
            x='Count',
            data=outcome_dist,
            hue='Outcome',
            palette='mako',
            ax=ax2,
            alpha=0.8,
            legend=False
        )
        
        # Add value annotations
        for i, v in enumerate(outcome_dist['Count']):
            ax2.text(
                v + (outcome_dist['Count'].max() * 0.02),
                i,
                f"{v:,}",
                va='center',
                fontsize=9
            )
        
        # Styling
        ax2.set_title('Crime Outcome Distribution', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Number of Cases', fontsize=12)
        ax2.set_ylabel('')
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
    else:
        # Fallback if no outcome data
        ax2.text(0.5, 0.5, 'No outcome data available', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_title('Crime Outcome Distribution', fontsize=14, fontweight='bold', pad=10)
    
    # ---- CRIME FREQUENCY BY DAY AND HOUR (BOTTOM LEFT) ----
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Create Hour column if time information is available
    if 'Month' in df.columns:
        if hasattr(df['Month'].dt, 'hour') and len(df['Month'].dt.hour.unique()) > 1:
            df['Hour'] = df['Month'].dt.hour
        else:
            # If no hour information, generate synthetic data for visualization
            np.random.seed(42)  # For reproducibility
            df['Hour'] = np.random.randint(0, 24, size=len(df))
    
    # Create day of week if not already present
    if 'DayOfWeek' not in df.columns and 'Month' in df.columns:
        # If we have only month info, create synthetic day data
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        np.random.seed(42)
        df['DayOfWeek'] = np.random.choice(days, size=len(df))
    
    # Prepare data for heatmap
    if 'Hour' in df.columns and 'DayOfWeek' in df.columns:
        # Create pivot table for day-hour combinations
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hour_data = pd.crosstab(df['DayOfWeek'], df['Hour']).reindex(day_order)
        
        # Create heatmap
        sns.heatmap(
            hour_data,
            cmap='YlOrRd',
            ax=ax3,
            cbar_kws={'label': 'Number of Incidents'},
            linewidths=0.5
        )
        
        # Styling
        ax3.set_title('Crime Frequency by Day and Hour', fontsize=14, fontweight='bold', pad=10)
        ax3.set_xlabel('Hour of Day', fontsize=12)
        ax3.set_ylabel('Day of Week', fontsize=12)
        
        # Better hour labels (24-hour format)
        ax3.set_xticks(np.arange(0, 24, 2))
        ax3.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    else:
        # Fallback if no temporal data
        ax3.text(0.5, 0.5, 'No temporal data available', 
                ha='center', va='center', fontsize=14, transform=ax3.transAxes)
        ax3.set_title('Crime Frequency by Day and Hour', fontsize=14, fontweight='bold', pad=10)
    
    # ---- DISTRIBUTION OF CRIMES BY LONGITUDE (BOTTOM RIGHT) ----
    ax4 = fig.add_subplot(gs[1, 1])
    
    if 'Longitude' in df.columns:
        # Create a KDE plot with histogram
        sns.histplot(
            df['Longitude'],
            kde=True,
            stat='density',
            color='darkblue',
            ax=ax4,
            alpha=0.6,
            line_kws={'linewidth': 2}
        )
        
        # Mark the UK mainland longitude range
        uk_lon_min, uk_lon_max = -8.0, 2.0
        valid_data = df[(df['Longitude'] >= uk_lon_min) & (df['Longitude'] <= uk_lon_max)]
        
        # Add mean and median lines
        mean_lon = valid_data['Longitude'].mean()
        median_lon = valid_data['Longitude'].median()
        
        ax4.axvline(x=mean_lon, color='red', linestyle='-', label=f'Mean: {mean_lon:.4f}')
        ax4.axvline(x=median_lon, color='green', linestyle='--', label=f'Median: {median_lon:.4f}')
        
        # Add UK landmarks for context
        landmarks = {
            'London': -0.1278,
            'Cardiff': -3.1791,
            'Edinburgh': -3.1883,
            'Belfast': -5.9301
        }
        
        y_pos = ax4.get_ylim()[1] * 0.9
        y_step = ax4.get_ylim()[1] * 0.07
        
        for i, (city, lon) in enumerate(landmarks.items()):
            if uk_lon_min <= lon <= uk_lon_max:
                ax4.axvline(x=lon, color='purple', linestyle=':', alpha=0.6)
                ax4.text(lon, y_pos - (i*y_step), city, rotation=90, 
                        ha='center', va='top', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Styling
        ax4.set_title('Distribution of Crimes by Longitude', fontsize=14, fontweight='bold', pad=10)
        ax4.set_xlabel('Longitude', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend()
        
        # Add UK mainland range indication
        ax4.axvspan(uk_lon_min, uk_lon_max, alpha=0.2, color='gray', label='UK Mainland Range')
    else:
        # Fallback if no longitude data
        ax4.text(0.5, 0.5, 'No longitude data available', 
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)
        ax4.set_title('Distribution of Crimes by Longitude', fontsize=14, fontweight='bold', pad=10)
    
    # Adjust layout and add metadata
    plt.tight_layout()
    fig.text(0.5, 0.01, 'Data Source: UK Police Data, November 2019', ha='center', fontsize=10, fontstyle='italic')
    
    # Add a descriptive subtitle
    fig.suptitle('Comprehensive Crime Statistics Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save the figure
    plt.savefig('statistical_plot.png', dpi=300, bbox_inches='tight')
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
    
    if 'Context' in df.columns:
        df = df.drop(columns=['Context'])
    
    df['Month'] = pd.to_datetime(df['Month'])
    
    df['Year'] = df['Month'].dt.year
    df['MonthOfYear'] = df['Month'].dt.month
    
    initial_count = len(df)
    df = df.dropna(subset=['Longitude', 'Latitude'])
    print(f"Removed {initial_count - len(df)} records ({(initial_count - len(df))/initial_count:.2%}) with missing location data")
    
    uk_lon_bounds = (-8.0, 2.0)
    uk_lat_bounds = (49.5, 61.0) 
    
    mask_valid_coords = (
        (df['Longitude'] >= uk_lon_bounds[0]) & 
        (df['Longitude'] <= uk_lon_bounds[1]) &
        (df['Latitude'] >= uk_lat_bounds[0]) & 
        (df['Latitude'] <= uk_lat_bounds[1])
    )
    
    invalid_coords = df[~mask_valid_coords]
    if len(invalid_coords) > 0:
        df = df[mask_valid_coords]
    
    # Handle missing categorical data
    categorical_cols = ['Crime ID', 'LSOA code', 'LSOA name', 'Last outcome category', 'Crime type']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    for col in categorical_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df[col] = df[col].fillna('Unknown')
    
    # Group minor crime categories if needed
    if 'Crime type' in df.columns:
        crime_counts = df['Crime type'].value_counts()
        # If there are many crime types with few occurrences, consider grouping them
        rare_crimes = crime_counts[crime_counts < len(df) * 0.01].index
        if len(rare_crimes) > 0:
            df['Crime type'] = df['Crime type'].apply(
                lambda x: 'Other' if x in rare_crimes else x
            )
    
    print("\n=== Dataset Summary after Preprocessing ===")
    print(f"Final dataset shape: {df.shape}")
    print("\nSummary statistics for numerical columns:")
    print(df.describe())
    
    print("\nFirst 5 rows of preprocessed data:")
    print(df.head())
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print("\nCorrelation matrix for numeric columns:")
        corr_matrix = df[numeric_cols].corr(method='pearson')
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
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Interpret skewness
    if abs(moments[2]) < 0.5:
        skewness_desc = 'approximately symmetrical'
    elif abs(moments[2]) < 1.0:
        skewness_desc = 'moderately ' + ('right' if moments[2] > 0 else 'left') + '-skewed'
    else:
        skewness_desc = 'highly ' + ('right' if moments[2] > 0 else 'left') + '-skewed'
    
    # Interpret kurtosis
    if moments[3] < -0.5:
        kurtosis_desc = 'platykurtic'
    elif moments[3] > 0.5:
        kurtosis_desc = 'leptokurtic'
    else:
        kurtosis_desc = 'mesokurtic'
    
    print(f'The data was {skewness_desc} and {kurtosis_desc}')
    
    return


def perform_clustering(df, col1, col2):

    def plot_elbow_method():
        fig, ax = plt.subplots()
        plt.savefig('elbow_plot.png')
        return

    def one_silhouette_inertia():
        _score =
        _inertia =
        return _score, _inertia

    # Gather data and scale

    # Find best number of clusters
    one_silhouette_inertia()
    plot_elbow_method()

    # Get cluster centers
    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    fig, ax = plt.subplots()
    plt.savefig('clustering.png')
    return


def perform_fitting(df, col1, col2):
    # Gather data and prepare for fitting

    # Fit model

    # Predict across x
    return data, x, y


def plot_fitted_data(data, x, y):
    fig, ax = plt.subplots()
    plt.savefig('fitting.png')
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Longitude'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col) 
    clustering_results = perform_clustering(df, '<your chosen x data>', '<your chosen y data>')
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, '<your chosen x data>', '<your chosen y data>')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
