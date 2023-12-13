# -*- Assignment-2 ads1 - Statistical and Trends -*-
"""
Created on Wed Dec 13 15:06:18 2023

@author: Shilpa
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import skew, kurtosis


def read_world_bank_data(filename):
    """
    Read world bank data from a CSV file, transpose the dataframe, and clean it by 
    dropping NaN values.

    Variables:
    file(str): The path to the CSV file containing the world bank data.

    Returns:
    tuple: A tuple containing two dataframes - the original data and the 
    cleaned and transposed data.
    """
    # Reading the data from the provided CSV file
    original_data = pd.read_csv(filename)

    # Transpose the dataframe
    data_transposed = original_data.transpose()

    # Clean the transposed dataframe by dropping NaN values
    data_transposed_cleaned = data_transposed.dropna()

    # Return both the original and cleaned transposed dataframes
    return original_data, data_transposed_cleaned


file_path = 'world-data.csv'
original_data, transposed_data = read_world_bank_data(file_path)

# Display the original and cleaned transposed dataframes
print('Original Data:')
print(original_data.head())

print('\nCleaned and Transposed Data:')
print(transposed_data.head())


def plot_renewable_electricity(original_data, countries, indicator, years):
    """
    Plot a line plot with markers showing renewable electricity output as a 
    percentage of total electricity
    for selected countries over specified years.

    Variables:
    original_data (DataFrame): The original dataset containing electricity data.
    countries (list): A list of countries to include in the plot.
    indicator (str): The indicator for renewable electricity output percentage.
    years (list): A list of years to include in the plot.

    Returns:
    Output: Displays the line plot with markers.
    """
    # Ensure indicator is a list
    if not isinstance(indicator, list):
        indicator = [indicator]

    # Filtering the data for the selected countries and years
    selected_data = original_data[(original_data['Country Name'].isin(countries)) 
                                  & (original_data['Indicator Name'].isin(indicator))][['Country Name'] + years].fillna(0)

    # Checking if the filtered data is empty
    if selected_data.empty:
        print('No data to plot.')
        return

    # Melt the DataFrame to make it suitable for Seaborn lineplot
    melted_data = selected_data.melt(
        id_vars = ['Country Name'], var_name = 'Year', value_name = indicator[0])

    # Plot a line plot with markers for each point
    plt.figure(figsize = (12, 6))
    sns.set(style = 'darkgrid')
    sns.set_palette("muted")  # Use vibrant colors
    lineplot = sns.lineplot(
        x = 'Year', y = indicator[0], hue = 'Country Name', markers = True, 
        style = 'Country Name', data = melted_data)
    # Calculate the mean value for each country
    country_means = melted_data.groupby('Country Name')[indicator[0]].mean()
    # Create new labels with the mean value next to the country name for the legend
    labels = [f"{country}: {mean:.2f}%" for country,
              mean in country_means.iteritems()]

    # Seting the new labels to the legend
    handles, _ = lineplot.get_legend_handles_labels()
    plt.legend(handles, labels, title = 'Country Name', loc = "upper left")

    plt.title('Renewable Electricity Output as % of Total Electricity')
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()


# Specified years, countries, indicators for the Line Plot
# Convert years to strings if necessary
selected_years = [str(year) for year in range(2010, 2020)]
selected_countries = ['Australia', 'Canada',
                      'China', 'Germany', 'United States']

selected_indicator = "Renewable electricity output (% of total electricity output)"

# Calling the function to create the plot with vibrant colors and marker style
plot_renewable_electricity(
    original_data, selected_countries, selected_indicator, selected_years)


def bar_plot(original_data, countries, indicator_name, years):
    """
    Create a bar plot to visualize the specified indicator for selected 
    countries over the given years.
    
     Variables:
    - original_data (pd.DataFrame): The original dataset containing relevant information.
    - countries (list): A list of country names to be included in the plot.
    - indicator_name (str): The name of the indicator to be visualized.
    - years (list): A list of years for which the data will be plotted.

    Returns:
    Output: Displays the Bar plot.
    """
    # Filtering the data for the selected countries and years
    data_filtered = original_data[(original_data['Country Name'].isin(countries)) & (
        original_data['Indicator Name'] == indicator_name)][['Country Name'] + years].fillna(0)

    # Melting the DataFrame to make it suitable for Seaborn barplot
    data_melted = data_filtered.melt(
        id_vars = ['Country Name'], var_name = 'Year', value_name = indicator_name)

    # Plotting
    plt.figure(figsize = (14, 7))
    sns.set(style = "whitegrid", palette = "Set1")
    barplot = sns.barplot(x = 'Year', y = indicator_name,
                          hue = 'Country Name', data = data_melted)

    # Adding text labels into the legend
    handles, labels = barplot.get_legend_handles_labels()
    # Calculating the mean value for each country
    country_means = data_melted.groupby(
        'Country Name')[indicator_name].mean().reindex(countries)
    # Creating new labels with the mean value next to the country name
    new_labels = [f'{label}: {country_means[label]:.2f}' for label in labels]
    # Seting the new labels to the legend inside the plot at the upper left location
    plt.legend(handles, new_labels, title = 'Country Name',
               loc = "upper left", bbox_to_anchor = (0, 1))

    plt.title(f'{indicator_name} by Country from 2010-2019', fontsize = 16)
    plt.xlabel('Year', fontsize = 13)
    plt.ylabel(f'{indicator_name}', fontsize = 13)
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()


# Specified years, countries, indicators for the Bar Plot
years_for_plot = ['2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019']
selected_countries = ['Australia', 'Canada',
                      'China', 'Germany', 'United States']
selected_indicator = "CO2 emissions (kt)"

# Calling the function to plot the data for Bar Plot
bar_plot(original_data, selected_countries, selected_indicator, years_for_plot)


def generate_pie_chart(original_data, countries, indicator, selected_year):
    """
    Generate a pie chart to visualize the distribution of a specific indicator 
    across selected countries for a given year.
    
    Variables:
    - original_data (pd.DataFrame): The original dataset containing relevant information.
    - countries (list): A list of country names to be included in the pie chart.
    - indicator (str): The name of the indicator to be visualized.
    - selected_year (str): The year for which the data will be visualized.

    Returns:
    Output: Displays the Pie Chart.
   """
   # Filtering data for the specified countries and indicator
    filtered_data = original_data[(original_data['Country Name'].isin(
        countries)) & (original_data['Indicator Name'] == indicator)]

    data_pie = filtered_data[['Country Name', selected_year]].dropna()

    # Creating the pie chart
    plt.figure(figsize = (12, 6))
    plt.pie(data_pie[selected_year], labels = data_pie['Country Name'],
            autopct = "%1.1f%%", startangle = 150, explode = [0.02]*len(data_pie))
    plt.title(f'Pie Chart of {indicator} in {selected_year}')

    # Show the plot
    plt.show()


# Specified years, countries, indicators for the Pie Plot
selected_countries_pie = ['Australia', 'Canada',
                          'China', 'Germany', 'United States']
selected_indicator_pie = "Electric power consumption (kWh per capita)"
selected_year_pie = '2010'

# Calling the function to plot the pie chart
generate_pie_chart(original_data, selected_countries_pie,
                   selected_indicator_pie, selected_year_pie)


def generate_scatter_plot(original_data, countries_list, indicator_list, selected_years):
    """
   Generate a scatter plot with lines to visualize the trend of selected 
   indicators across multiple countries over specified years.

   Variable:
   - original_data (pd.DataFrame): The original dataset containing relevant information.
   - countries_list (list): A list of country names to be included in the scatter plot.
   - indicator_list (list or str): A list of indicators or a single indicator to be visualized.
   - selected_years (list): A list of years for which the data will be plotted.

   Returns:
   Output: Displays the scatter plot
   """
    # Ensure indicator_list is a list
    if not isinstance(indicator_list, list):
        indicator_list = [indicator_list]

    # Filtering the data for the selected countries and years
    filtered_data = original_data[(original_data['Country Name'].isin(countries_list)) 
                                  & (original_data['Indicator Name'].isin(indicator_list))][['Country Name'] + selected_years].fillna(0)

    # Checking if the filtered data is empty
    if filtered_data.empty:
        print("No data to plot.")
        return

    # Melting the DataFrame to make it suitable for Seaborn scatter plot
    melted_data = filtered_data.melt(
        id_vars = ['Country Name'], var_name = 'Year', value_name = indicator_list[0])

    # Plotting a scatter plot with lines
    plt.figure(figsize = (10, 6))
    sns.set(style = "darkgrid", palette = "muted")
    scatter_plot = sns.scatterplot(
        x = 'Year', y = indicator_list[0], hue = 'Country Name', 
        style = 'Country Name', markers = True, data = melted_data)
    sns.lineplot(x='Year', y=indicator_list[0], hue='Country Name', style='Country Name',
                 markers = False, dashes = True, data = melted_data, legend = False)

    # Adding the average consumption in the legend
    averages = melted_data.groupby('Country Name')[
        indicator_list[0]].mean().to_dict()
    legend_labels = [
        f"{country} ({averages[country]:.2f}%)" for country in countries_list]
    handles, old_labels = scatter_plot.get_legend_handles_labels()
    plt.legend(handles, legend_labels, title = 'Country Name', loc = 'upper left')

    plt.title(f'Renewable Energy Consumption as % of Total Final Energy Consumption')
    plt.xlabel('Year')
    plt.ylabel(indicator_list[0])
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()


# Specified years, countries, indicators for the Scatter Plot
selected_years_scatter = ['2004', '2006', '2008', '2010', '2012', '2014']
selected_countries_scatter = ['Australia',
                              'Canada', 'China', 'Germany', 'United States']
selected_indicator_scatter = "Renewable energy consumption (% of total final energy consumption)"

# Call the function to create the plot
generate_scatter_plot(original_data, selected_countries_scatter,
                      selected_indicator_scatter, selected_years_scatter)


def generate_custom_box_plot(original_data, countries_list, indicator_list, selected_years):
    """
    Generate a custom box plot to visualize the distribution of electricity 
    production from renewable sources across selected countries over specified years.

    Variable:
    - original_data (pd.DataFrame): The original dataset containing relevant information.
    - countries_list (list): A list of country names to be included in the box plot.
    - indicator_list (list): A list of indicators to be visualized.
    - selected_years (list): A list of years for which the data will be plotted.

    Returns:
    Output: Displays the Box plot.
    """
    # Filtering the data for the selected countries and years
    filtered_data = original_data[(original_data['Country Name'].isin(countries_list)) 
                                  & (original_data['Indicator Name'].isin(indicator_list))][['Country Name'] + selected_years].fillna(0)

    # Check if the filtered data is empty
    if filtered_data.empty:
        print("No data to plot.")
        return

    # Melting the DataFrame to make it suitable for Seaborn boxplot
    melted_data = filtered_data.melt(id_vars = ['Country Name'], var_name = 'Year',
                                     value_name = 'Electricity production from renewable sources, excluding hydroelectric (kWh)')

    # Plotting a box plot
    plt.figure(figsize = (12, 6))
    sns.set(style = "whitegrid", palette = "bright")
    sns.boxplot(x = 'Country Name', y = 'Electricity production from renewable sources, excluding hydroelectric (kWh)',
                hue = 'Country Name', data = melted_data)

    # Customizing the legend with percentage labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, [f"{label} ({round(melted_data[melted_data['Country Name'] == label]['Electricity production from renewable sources, excluding hydroelectric (kWh)'].mean()*100, 2)}%)"
                         for label in labels], title = 'Country Name', loc = 'upper left')
    plt.title('Electricity production from renewable sources')
    plt.xlabel('Country Name')
    plt.ylabel('Electricity production from renewable sources')
    plt.xticks(rotation = 45)
    plt.show()
    return


# Specified years, countries, indicators for the Box Plot
selected_years_box = ['2004', '2006', '2008',
                      '2010', '2012', '2014', '2016', '2018']
selected_countries_box = ['Australia', 'Canada',
                          'United States', 'China', 'Germany']
selected_indicator_box = [
    "Electricity production from renewable sources, excluding hydroelectric (kWh)"]

# Calling the function to create the plot
generate_custom_box_plot(original_data, selected_countries_box,
                         selected_indicator_box, selected_years_box)


def generate_custom_heatmap(original_data, selected_countries, selected_year, selected_indicators):
    """
    Generate a custom correlation heatmap to visualize the correlation between 
    selected indicators for specific countries in a given year.

    Parameters:
    - original_data (pd.DataFrame): The original dataset containing relevant information.
    - selected_countries (list): A list of country names to be included in the heatmap.
    - selected_year (str): The year for which the data will be visualized.
    - selected_indicators (list): A list of indicators for which the correlation will be calculated.

    Returns:
    Output: Displays the Heatmmap.
    """
    # Filter for selected countries and indicators
    filtered_data = original_data[(original_data['Country Name'].isin(
        selected_countries)) & (original_data['Indicator Name'].isin(selected_indicators))]

    # Pivot the table to get countries as rows and indicators as columns
    pivoted_data = filtered_data.pivot(
        index = 'Country Name', columns = 'Indicator Name', values = selected_year)

    correlation_matrix = pivoted_data.corr()

    # Plotting the heatmap
    plt.figure(figsize = (10, 8))
    sns.heatmap(correlation_matrix, annot = True,
                cmap = 'coolwarm', fmt = ".2f", linewidths = .5)
    plt.title(f'Correlation Heatmap ({selected_year})')
    plt.xticks(rotation = 45, ha = "right")
    plt.yticks(rotation = 0)
    # It Adjusts the plot to ensure everything fits without overlapping
    plt.tight_layout() 
    plt.show()
    return


# Specified countries, year, and indicators
selected_countries_heatmap = ['Australia',
                              'Canada', 'China', 'Germany', 'United States']
selected_year_heatmap = '2010'
selected_indicators_heatmap = [
    'CO2 emissions (kt)',
    'Renewable electricity output (% of total electricity output)',
    'Electricity production from renewable sources, excluding hydroelectric (kWh)',
    'Electric power consumption (kWh per capita)',
    'Renewable energy consumption (% of total final energy consumption)']

# Call the function to create the correlation heatmap with different variables
generate_custom_heatmap(original_data, selected_countries_heatmap,
                        selected_year_heatmap, selected_indicators_heatmap)

# Specified countries, years, and the indicator for analysis
selected_countries = ['Australia', 'Canada',
                      'China', 'Germany', 'United States']
indicator = 'Renewable energy consumption (% of total final energy consumption)'
selected_years = ['2004', '2006', '2008',
                  '2010', '2012', '2014', '2016', '2018']

# Filter the data for the selected countries and indicator
filtered_data = original_data[(original_data['Country Name'].isin(
    selected_countries)) & (original_data['Indicator Name'] == indicator)]

# Aggregate the data for the specified years
indicator_data = filtered_data[selected_years].dropna().values.flatten()

# Calculate skewness and kurtosis
skewness = skew(indicator_data)
# Using Pearson's definition
kurtosis_value = kurtosis(indicator_data, fisher=False)

# Plot the histogram
plt.figure(figsize = (10, 6))
plt.hist(indicator_data, bins = 30, alpha = 0.7,
         color = 'skyblue', edgecolor = 'black')

# Determine the range for placing text
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# Adding skewness and kurtosis to the plot
plt.text(x_max * 0.7, y_max * 0.9,
         f'Skewness: {skewness:.2f}', fontsize = 12, color = 'black')
plt.text(x_max * 0.7, y_max * 0.85,
         f'Kurtosis: {kurtosis_value:.2f}', fontsize = 12, color = 'black')

# Style the plot
plt.title('Histogram with Skewness and Kurtosis for ' + indicator, fontsize = 15)
plt.xlabel('Values', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.grid(axis = 'y', alpha = 0.75)
plt.show()
