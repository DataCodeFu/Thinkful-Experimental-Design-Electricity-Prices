#### Import Python Libraries and Define Data Processing Functions   ---------------------------------------

# Import Python libraries.
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import csv
import os
from IPython.display import display, HTML
import calendar
import itertools as it


# Setup global settings.
#%matplotlib inline
pd.set_option('display.max_rows', 200)  # Int or None
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 250)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.width', 0)#None)
pd.set_option('display.max_colwidth', None)
pd.set_option('precision', 3)

    
"""
    Read csv file to determine row number of data headers.
        Function call example:
        -
        csv_header_row = get_header_row(full_csv_path)
"""
def get_header_row(full_csv_path):
    file_obj = open(full_csv_path)
    csv_reader_obj = csv.reader(file_obj)
    header_row = 0
    for row in csv_reader_obj:
        if len(row) > 1:
            break
        header_row += 1
    return header_row


"""
    Take a folder path and generate a 2D dictionary of file names and paths of all .csv files 
    within the selected folder.  The keys are the file names with '.csv' extensions in the folder.
    Each item value is a 2-key dictionary holding 'path' and 'name.'  This csv dictionary is later
    used as an input parameter for csv_list_to_dataframes().
        Function call example:
        location = "C:\\dev\Thinkful\\15. Capstone 1 - Experimental Design\\EIA data sets\\EIA whole data sets\\"
        csv_data_set1 = make_csv_dict(location)
"""
def make_csv_dict(location):
    file_list = []
    # r=>root, d=>directories, f=>files
    for r, d, f in os.walk(location):
       for item in f:
          file_list.append(item) if '.csv' in item else next
    csv_data_set = {}
    for item in file_list:
        csv_data_set[item.replace('.csv', '')] = {'path': location, 'name': item}
    return csv_data_set


"""
    Take a structured dictionary of CSV file locations or web addresses and download
    the data into a dictionary of dataframes.  Also checks to see if all the columns
    are matching on the tables or not.  Function call example:
        columns_df = csv_list_to_dataframes(csv_data_set, df_dict)
"""
def csv_list_to_df_dict(csv_data_set, df_dict, na_values):
    # Setup data buckets for loop through csv files.
    summary_df = pd.DataFrame()
    dfs_summary_dict = {}
    summary_df_col_index = []
    col_rename_dict = {}
    column_max_count, max_key = 0, ''
    
    # Loop keys/csv filenames to fill df dictionary and matching column check table.
    for table_name in csv_data_set.keys():
        # Download the csv file into a DataFrame and add it to the df_dict.
        full_csv_path = csv_data_set[table_name]['path'] \
                      + csv_data_set[table_name]['name']
        csv_header_row = get_header_row(full_csv_path)
        data = pd.read_csv(full_csv_path, header = csv_header_row, na_values=na_values)#.fillna(0)
        df_dict[table_name] = data.copy()
        # Collect summary information on the DataFrames.
        col_data = list(df_dict[table_name].columns)
        dfs_summary_dict.update({ table_name: col_data })
        if len(df_dict[table_name].columns) > column_max_count:
            column_max_count = len(df_dict[table_name].columns)
            max_key = table_name
    # Select one of the largest DataFrames to set the column order.
    column_index_key = df_dict[max_key].columns
    # Modify list of column values in dfs_summary_dict > summary_df.
    matches, insertions = 0, 0
    for report, col_data in dfs_summary_dict.items(): 
        col_idx = 0
        for key_idx in range(0, len(column_index_key)):
            no_match = column_index_key[key_idx] != col_data[col_idx]
            if no_match:
                col_data.insert(col_idx, np.nan)
                insertions += 1
                col_idx += 1
            else:
                matches += 1
                col_idx += 1
    equalized_columns = { key: pd.Series(value) for key, value in dfs_summary_dict.items() }
    equalized_columns_df = pd.DataFrame.from_dict(equalized_columns)
    summary_df_col_index += list(equalized_columns_df.columns)
    for i in range(len(summary_df_col_index)):
        col_rename_dict.update({i: summary_df_col_index[i]})
    concat_list = [summary_df, equalized_columns_df]
    summary_df = pd.concat(concat_list, copy=False, axis='columns')[equalized_columns_df.columns]
    summary_df.rename(columns=col_rename_dict, inplace=True)
    equal_columns = summary_df.apply(lambda row: row[0] == row.all(), axis=1)
    summary_df.insert(loc=0, column='_equal_table_columns', value=equal_columns)
    # Copy into a column-only dataframe, match_columns_df, excluding dimension rows.
    match_columns_df = summary_df.reset_index(drop=True).copy(deep=True)
    val_cnts = match_columns_df['_equal_table_columns'].value_counts()    
    summary_df.index.name = 'col_pos'
    summary_df = summary_df.transpose()
    summary_df.index.name = 'table_names'
    summary_df.reset_index(inplace=True)
    print("\n{:,} matching columns and {:,} mis-matching columns (element-wise including NaNs) for all csv tables > dataframes.".format(
        val_cnts.at[True] if True in val_cnts.index else 0,      
        val_cnts.at[False] if False in val_cnts.index else 0))
    print("{:,} max columns across all dataframes.".format(column_max_count))
    return summary_df, column_index_key


"""
    Take a dictionary of dataframes and convert them into a single large dataframe.
    Also checks to see if the columns names and total row numbers are matching on the tables or not.
    Function call example:
        folder_df1 = df_dict_to_single_df(df_dict1)
"""
def df_dict_to_folder_df(df_dict, column_index_key):
    # Concatenate vertically along 0/columns and multi-index on csv names.
    new_row_label = 'csv_table_names'
    df_list = []
    table_row_counts = []
    for key, df in df_dict.items():
        table_row_counts.append(df.shape[0])
        csv_name_col = df.apply(lambda row: key, axis=1)
        df.insert(loc=0, column=new_row_label, value=csv_name_col)
        df_list.append(df) 
    column_index_key = column_index_key.insert(0, new_row_label)
    folder_df = pd.concat(df_list)
    folder_df.reset_index(drop=True, inplace=True)
    folder_df_concat_good = folder_df.shape[0] == sum(table_row_counts)
    if not folder_df_concat_good:
        print("ERROR: Concatenation of DataFrames did not result in the same number of rows.".format(bad_stat))
        print("{:,} total rows in the dictionary of DataFrames.".format(sum(table_row_counts)))
        print("{:,} rows in the vertically concatenated DataFrame result.".format(folder_df.shape[0]))  
    return folder_df, column_index_key


"""
    Drop zero-sum numeric rows from dataframe folder_df.  If non_numeric_cols are not provided
    in a manual list, then Numeric columns are selected by their data type being either 
    int64 or float64.  Function call example:
        non_numeric_cols = ['description', 'units', 'source key']
        drop_zero_sum_numeric_rows(folder_df, non_numeric_cols)
"""
def drop_zero_sum_numeric_rows(folder_df, non_numeric_cols=None):
    if non_numeric_cols == None:
        non_numeric_cols = []
        dtype_dict = dict(folder_df.dtypes)
        for col in dtype_dict:
            if dtype_dict[col] != 'float64' and dtype_dict[col] != 'int64':
                non_numeric_cols.append(col)    
    numeric_df = folder_df.drop(columns=non_numeric_cols)
    zero_rows_df = pd.DataFrame(numeric_df.apply(lambda row: True if np.sum(row) == 0 else False, axis=1),
                                columns=['zero_sum_rows'],
                                index=numeric_df.index)
    zero_rows_only_df = zero_rows_df[zero_rows_df['zero_sum_rows'] == True]
    # Drop rows with zero sum numeric amounts.
    folder_df.drop(index=zero_rows_only_df.index, inplace=True)
    # Count rows and check for potential errors.
    all_row_ct = zero_rows_df.shape[0]
    zero_sum_row_ct = zero_rows_only_df.shape[0]
    remaining_row_ct = folder_df.shape[0]
    if all_row_ct != (zero_sum_row_ct + remaining_row_ct):
        print("Total rows {} less {} non-numeric rows equals {}, but folder_df now has {}".format(
              all_row_ct, zero_sum_row_ct, all_row_ct - zero_sum_row_ct, remaining_row_ct))
    folder_df.reset_index(drop=True, inplace=True)
    folder_df.index.rename('index', inplace=True)
    return folder_df


"""
    Combine DataFrames containing a folder's worth of csv files.  Function call example:
        all_one_df = combine_folder_dfs(folder_df1_non_zero, folder_df2_non_zero, column_index_key1)
"""
def combine_folder_dfs(folder_df1, folder_df2, column_index_key):
    # Combine data set part 1 and part 2 to create one large dataframe of original values.
    all_one_df = pd.concat([folder_df1, folder_df2])[column_index_key]
    row_miscount = (all_one_df.shape[0] - folder_df1.shape[0] - folder_df2.shape[0]) != 0
    if row_miscount:
        print("row_miscount =", row_miscount)
    else:
        # Additional data cleaning items on the combined DataFrame.
        split_description = all_one_df['description'].str.split(pat=': ', expand=True)
        split_description_df = pd.DataFrame(split_description)
        split_description_df.rename(columns={0: 'location', 1: 'sector'}, inplace=True)
        all_one_df.insert(loc=1, column='location', value=split_description_df['location'])
        all_one_df.insert(loc=2, column='sector', value=split_description_df['sector'])
        all_one_df.drop(columns=['description'], inplace=True)
        all_one_df.index.rename('index', inplace=True)
    return all_one_df


def display_column_counts(df, col_count_list):
    df_concat_list = []
    for col in col_count_list:
        temp_df = pd.DataFrame(df[col].value_counts()).reset_index(drop=False)
        temp_df.rename(columns={'index': col, col: col + '_count'}, inplace=True)
        df_concat_list.append(temp_df)
        col_space = pd.DataFrame(temp_df.apply(lambda row: '||||||||', axis='columns'), columns=['||||||||'])
        df_concat_list.append(col_space)
    category_count_df = pd.concat(df_concat_list, axis='columns')
    return category_count_df
    display(category_count_df)


def get_95_ci_by_thinkful(array_1, array_2):
    sample_1_n = array_1.shape[0]
    sample_2_n = array_2.shape[0]
    sample_1_mean = array_1.mean()
    sample_2_mean = array_2.mean()
    sample_1_var = array_1.var()
    sample_2_var = array_2.var()
    mean_difference = sample_2_mean - sample_1_mean
    std_err_difference = math.sqrt((sample_1_var/sample_1_n)+(sample_2_var/sample_2_n))
    margin_of_error = 1.96 * std_err_difference
    ci_lower = mean_difference - margin_of_error
    ci_upper = mean_difference + margin_of_error
    return("The difference in means at the 95% confidence interval (two-tail) is between " \
           + "{:,.1} and {:,.1}.".format(str(ci_lower), str(ci_upper)))


def create_data_and_csv():

	#### Creation of Data Set  -------------------------------------------------------------------------------

	# Data Part 1: Obtained from EIA at https://www.eia.gov/electricity/data/browser/
	# under 'Change data set'.
	location1 = "C:\\dev\Thinkful\\15. Capstone 1 - Experimental Design\\EIA data sets\\EIA whole data sets\\"
	na_values1 = ['', '--', 'NM', 'W']
	df_dict1 = {}
	column_index_key1 = pd.DataFrame()
	csv_data_set1 = make_csv_dict(location1)
	summary_df1, column_index_key1  = csv_list_to_df_dict(csv_data_set1, df_dict1, na_values1)
	folder_df1, column_index_key1 = df_dict_to_folder_df(df_dict1, column_index_key1)
	folder_df1_non_zero = drop_zero_sum_numeric_rows(folder_df1)


	# Data Part 2: Obtained from EIA at https://www.eia.gov/electricity/data/browser/
	# under 'View a pre-generated report'.
	location2 = "C:\\dev\\Thinkful\\15. Capstone 1 - Experimental Design\\EIA data sets\\" \
	          + "EIA pre-generated reports\\Time-series data, monthly 2001-01 to 2020-02\\"
	na_values2 = ['', '--', 'NM', 'W']
	df_dict2 = {}
	csv_data_set2 = make_csv_dict(location2)
	summary_df2, column_index_key2 = csv_list_to_df_dict(csv_data_set2, df_dict2, na_values2)
	folder_df2, column_index_key2 = df_dict_to_folder_df(df_dict2, column_index_key2)
	folder_df2_non_zero = drop_zero_sum_numeric_rows(folder_df2)
	#summary_df2.loc[summary_df2[4].isnull()]    # Tables that do not have the first column.


	# Combine data set part 1 and part 2, clean data, and export to csv.
	if len(column_index_key1) > len(column_index_key2):
	    column_index_key_all = column_index_key1
	else:
	    column_index_key_all = column_index_key1
	all_one_df = combine_folder_dfs(folder_df1_non_zero, folder_df2_non_zero, column_index_key1)
	all_one_df_unpivoted = all_one_df.melt(id_vars=['csv_table_names', 'location', 'sector', 'units', 'source key'], 
	                                       var_name='month_year', 
	                                       value_name='variable_value')
	all_one_df_unpivoted.to_csv('eia_data.csv')
	# print("From (row, column):", all_one_df.shape, "to", all_one_df_unpivoted.shape, "with 6 category columns.")

	#### Import Electricity Price Data and Select Key Report Sources for the Experiment. -------------------------------

	raw_data_df = pd.read_csv('eia_data.csv')
	raw_data_df.set_index([raw_data_df.columns.values[0]], inplace=True)
	raw_data_df.index.names = [None]
	# print("raw_data_df.shape (rows, columns):", raw_data_df.shape, '\n')
	# reports = sorted(raw_data_df['csv_table_names'].unique())
	# print(*reports, sep='\n')

	# -------------------------------

	table_names = {
	    '5.06 Average_retail_price_of_electricity - by state by sector': \
	        'avg_retail_electricity_price',
	    '4.10.a Average_cost_of_fossil_fuels_for_electricity_generation_(per_Btu)_for_coal - by state': \
	        'avg_coal_cost_btu_generation',
	    '4.11.a Average_cost_of_fossil_fuels_for_electricity_generation_(per_Btu)_for_petroleum_liquids by state by sector': \
	        'avg_pet_liq_cost_btu_generation',
	    '4.12.a Average_cost_of_fossil_fuels_for_electricity_generation_(per_Btu)_for_petroleum_coke - by state by sector': \
	        'avg_pet_coke_cost_btu_generation',
	    '4.13.a Average_cost_of_fossil_fuels_for_electricity_generation_(per_Btu)_for_natural_gas - by state': \
	        'avg_nat_gas_cost_btu_generation'
	}
	col_count_list = ['sector', 'location']
	drop_columns = ['csv_table_names', 'units', 'source key']
	table_data = {}

	for table, name in table_names.items():
	    table_attributes = {}
	    
	    temp_filter = (raw_data_df['csv_table_names'].astype(str) == table)
	    temp_df = raw_data_df[temp_filter].reset_index(drop=True)
	    temp_df.name = temp_df['csv_table_names'].iat[0]
	    temp_df.index.name = str(temp_df['csv_table_names'].iat[0]) \
	                       + ' (' + str(temp_df['units'].iat[0]) + ')'
	    temp_df.rename(columns={'variable_value': name}, inplace=True)
	    temp_df.drop(columns=drop_columns, inplace=True)
	    temp_df.dropna(inplace=True)
	    table_attributes['filter'] = temp_filter
	    table_attributes['df'] = temp_df.copy()
	    table_attributes['name'] = name
	    table_attributes['long_name'] = table
	    table_data[name] = table_attributes.copy()

	#print("avg_retail_electricity_price before removing state location data (rows, columns):", table_data['avg_retail_electricity_price']['df'].shape)
	#display_column_counts(table_data['avg_retail_electricity_price']['df'], col_count_list)

	#### Look at Electricity Prices and Clean Up Table -------------------------------------------

	# Create new DataFrame df to hold avg_retail_electricity_price.
	df = table_data['avg_retail_electricity_price']['df'].copy()

	# Separate out year and month into separate columns.
	year_month_columns = df['month_year'].str.split(pat=' ', expand=True)
	year_month_columns_df = pd.DataFrame(year_month_columns)
	year_month_columns_df.rename(columns={0: 'month', 1: 'year'}, inplace=True)
	df.insert(loc=2, column='month', value=year_month_columns_df['month'])
	df.insert(loc=3, column='year', value=year_month_columns_df['year'])
	 
	# Format the month, year, and month_year columns as int64 and datetime data types.
	df['month_year'] = pd.to_datetime(df['month_year'])
	df['year'] = pd.to_numeric(df['year'])
	df['month'] = pd.to_numeric(df['month'].apply(lambda row: list(calendar.month_abbr).index(row)))
	#df['month'] = pd.to_datetime(df['month_year'], format='%m')
	# df['Month'] = pd.to_datetime(df['Month'], format='%m').dt.month_name().str.slice(stop=3)
	df['location'] = df['location'].str.strip()
	df['sector'] = df['sector'].str.strip()
	#display(df.info())
	#display(df.head())

	# --------------------------------------------

	# Simplify electricity price data to aggregated geographic locations.
	aggregated_locations = [ 'United States',
	                         'Pacific Contiguous',
	                         'Pacific Noncontiguous',
	                         'South Atlantic',
	                         'Middle Atlantic',
	                         'West South Central',
	                         'East North Central' ]

	filter_agg_location = ( df['location'].str.contains('United States') ) \
	                    | ( df['location'].str.contains('Pacific Contiguous') ) \
	                    | ( df['location'].str.contains('Pacific Noncontiguous') ) \
	                    | ( df['location'].str.contains('South Atlantic') ) \
	                    | ( df['location'].str.contains('Middle Atlantic') ) \
	                    | ( df['location'].str.contains('West South Central') ) \
	                    | ( df['location'].str.contains('East North Central') )

	# filter_us_all = ( df['sector'].str.contains("all sectors") ) \
	#               & ( df['location'].str.contains("United States") )
	# filter_from_2010 = ( df['year'] >= 2010 )
	# filter_from_2010_no_pnc = ( df['year'] >= 2010 ) \
	#                         & ( ~ df['location'].str.contains('Pacific Noncontiguous') )
	# filter_from_2015 = ( df['year'] >= 2015 )
	# filter_from_2015_no_pnc = ( df['year'] >= 2015 ) \
	#                         & ( ~ df['location'].str.contains('Pacific Noncontiguous') )
	# filter_no_pnc_loc = ( ~ df['location'].str.contains('Pacific Noncontiguous') )

	# Remove overlapping data with location values at the state level, leaving regions.
	df = df.loc[filter_agg_location]
	#print('\n', df['location'].value_counts(), '\n')
	#print(df.info(), '\n')
	#display(df.head())

	# -----------------------------------------

	return df


def graph_categories(df):
	sns.catplot(x="location", y='avg_retail_electricity_price', hue="sector", kind="swarm", legend=False,
	            height=5, aspect=1.3, data=df).set_xticklabels(rotation=45)
	plt.tight_layout()
	plt.legend(loc="upper left")
	plt.show()


def graph_prices(df):
	sns.set(rc={'figure.figsize':(12,8)})
	g = sns.lineplot(x='month_year', y='avg_retail_electricity_price', hue='location', 
	                data=df)
	plt.tight_layout()
	plt.show()


def graph_prices_no_pnc(df):
	filter_no_pnc_loc = ( ~ df['location'].str.contains('Pacific Noncontiguous') )
	sns.set(rc={'figure.figsize':(12,8)})
	g = sns.lineplot(x='month_year', y='avg_retail_electricity_price', hue='location', 
	                data=df.loc[filter_no_pnc_loc])
	plt.tight_layout()
	plt.show()


def graph_boxplot_no_pnc(df):
	filter_no_pnc_loc = ( ~ df['location'].str.contains('Pacific Noncontiguous') )
	sns.set(rc={'figure.figsize':(12,8)})
	sns.boxplot(y='avg_retail_electricity_price', x='year', hue='location', data=df.loc[filter_no_pnc_loc])   # filter_from_2015_no_pnc
	plt.tight_layout()
	plt.show()


def graph_seasonality_changes_last_5_years(df):
	filter_from_2010_no_pnc = ( df['year'] >= 2010 ) \
	                        & ( ~ df['location'].str.contains('Pacific Noncontiguous') )
	fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, squeeze=True, figsize=(12,10))
	g1 = sns.lineplot(x='month', y='avg_retail_electricity_price', hue='year', 
	                  data=df.loc[filter_from_2010_no_pnc].query("location == 'United States'"),
	                  ax=ax[0, 0])
	ax[0, 0].set_title('United States')
	g2 = sns.lineplot(x='month', y='avg_retail_electricity_price', hue='year', 
	                  data=df.loc[filter_from_2010_no_pnc].query("location == 'West South Central'"),
	                  ax=ax[0, 1])
	ax[0, 1].set_title('West South Central')
	g2 = sns.lineplot(x='month', y='avg_retail_electricity_price', hue='year', 
	                  data=df.loc[filter_from_2010_no_pnc].query("location == 'Pacific Contiguous'"),
	                  ax=ax[1, 0])
	ax[1, 0].set_title('Pacific Contiguous')
	g3 = sns.lineplot(x='month', y='avg_retail_electricity_price', hue='year', 
	                  data=df.loc[filter_from_2010_no_pnc].query("location == 'Middle Atlantic'"),
	                  ax=ax[1, 1])
	ax[1, 1].set_title('Middle Atlantic')
	plt.tight_layout()
	plt.show()


def graph_seasonality_by_category(df):
	filter_from_2010_no_pnc = ( df['year'] >= 2010 ) \
                        & ( ~ df['location'].str.contains('Pacific Noncontiguous') )
	fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, squeeze=True, figsize=(12,5))
	g1 = sns.lineplot(x='month', y='avg_retail_electricity_price', hue='sector', 
	                  data=df.loc[filter_from_2010_no_pnc], ax=ax[0])
	g2 = sns.lineplot(x='month', y='avg_retail_electricity_price', hue='location', 
	                  data=df.loc[filter_from_2010_no_pnc], ax=ax[1])
	plt.tight_layout()
	plt.show()


def separate_data_by_location(df):
	# Separating the data into seven (7) DataFrames by location for statistical analysis.
	locations = df['location'].value_counts().index.to_list()
	dfs_by_loc = {}
	for location in locations:
	    dfs_by_loc[location] = df.query("location == '{}'".format(location), inplace=False).reset_index(drop=True)
	print("{:<21} (Rows, Columns)".format("dfs_by_loc keys"))
	print("{:<21} ---------------".format("---------------"))
	[print("{:<21}".format(location), dfs_by_loc[location].shape) for location in locations]
	#display(dfs_by_loc['United States'].head())

	return dfs_by_loc


def energy_prices_by_geography_normality(df, dfs_by_loc):
	hist_axis = df.hist(column='avg_retail_electricity_price', by='location', 
	                    figsize=(12, 7), layout=(2, 4), sharex=False, sharey=True)
	for axis in hist_axis.flatten():
	    axis.set_xlabel("Electricity Prices")
	    axis.set_ylabel("Occurences in Avg. Monthly Pricing")
	    axis.tick_params(axis='x', labelrotation=45)
	plt.tight_layout()
	plt.show()

	# ------------------------------------------------
	# Generate normality statistics in normality_df.
	normality_stats1 = {'location':[], 'Skewness': [], 'Kurtosis': [], 
	                    'Shapiro-Wilk W-Stat':[], 'Shapiro-Wilk P-Stat':[]}
	for location, dframe in dfs_by_loc.items():
	    desc_stats = stats.describe(dframe['avg_retail_electricity_price'])
	    shap_wilk_stats = stats.shapiro(dframe['avg_retail_electricity_price'])
	    normality_stats1['location'].append(location)
	    normality_stats1['Skewness'].append(desc_stats.skewness)
	    normality_stats1['Kurtosis'].append(desc_stats.kurtosis)
	    normality_stats1['Shapiro-Wilk W-Stat'].append(shap_wilk_stats[0])
	    normality_stats1['Shapiro-Wilk P-Stat'].append(shap_wilk_stats[1])
	normality_df1 = pd.DataFrame.from_dict(normality_stats1)
	#normality_df1.apply(lambda x: '%.5f' % x, axis=1)
	#[print("{:<21}".format(location), dfs_by_loc[location].shape) for location in locations]

	# Plot normality statistics in a visual chart.
	skew_min, skew_norm, skew_max = -1, 0, 1
	kurt_min, kurt_norm, kurt_max = -3, 0, 3
	shww_min, shww_norm, shww_max =  0, 1, 1
	shwp_min, shwp_norm, shwp_max =  0.05, 1, 1
	x_count = len(dfs_by_loc.keys())
	fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, squeeze=True, figsize=(12,7))
	g1 = sns.barplot(x='location', y='Skewness', data=normality_df1, ax=ax[0, 0])
	ax[0, 0].set_title('Skewness')
	ax[0, 0].tick_params(axis='x', labelrotation=45)
	ax[0, 0].plot([skew_max for i in range(x_count)])
	ax[0, 0].plot([skew_min for i in range(x_count)])
	g2 = sns.barplot(x='location', y='Kurtosis', data=normality_df1, ax=ax[0, 1])
	ax[0, 1].set_title('Kurtosis')
	ax[0, 1].tick_params(axis='x', labelrotation=45)
	ax[0, 1].plot([kurt_max for i in range(x_count)])
	ax[0, 1].plot([kurt_min for i in range(x_count)])
	g2 = sns.barplot(x='location', y='Shapiro-Wilk W-Stat', data=normality_df1, ax=ax[1, 0])
	ax[1, 0].set_title('Shapiro-Wilk W-Stat')
	ax[1, 0].tick_params(axis='x', labelrotation=45)
	ax[1, 0].plot([shww_max for i in range(x_count)])
	g3 = sns.barplot(x='location', y='Shapiro-Wilk P-Stat', data=normality_df1, ax=ax[1, 1])
	ax[1, 1].set_title('Shapiro-Wilk P-Stat')
	ax[1, 1].tick_params(axis='x', labelrotation=45)
	ax[1, 1].plot([shwp_min for i in range(x_count)])
	plt.tight_layout()
	plt.show()
	#display(normality_df1)


def one_way_ANOVA_by_loc(dfs_by_loc):
	confidence_interval = 0.95
	alpha = 1 - confidence_interval
	df_list_no_us = [df for key, df in dfs_by_loc.items() if key != 'United States']
	df_concat = pd.concat(df_list_no_us, axis='rows').reset_index(drop=True)
	total_rows = [df.shape[0] for df in df_list_no_us]
	if sum(total_rows) != df_concat.shape[0]:
	    [print("{:<25} {:>10}".format(key, str(df.shape))) for key, df in dfs_by_loc.items()]
	    print("{:<25} {:>10}".format("tukey_concat", str(tukey_concat.shape)))
	deg_free_between = len(df_concat['location'].unique()) - 1
	deg_free_within = df_concat['avg_retail_electricity_price'].count() - deg_free_between
	f_crit_val = stats.f.ppf(q=confidence_interval, dfn=deg_free_between, dfd=deg_free_within)
	one_way_anova_result = stats.f_oneway(
	    dfs_by_loc['South Atlantic']['avg_retail_electricity_price'],
	    dfs_by_loc['East North Central']['avg_retail_electricity_price'],
	    dfs_by_loc['Middle Atlantic']['avg_retail_electricity_price'],
	    dfs_by_loc['West South Central']['avg_retail_electricity_price'],
	    dfs_by_loc['Pacific Contiguous']['avg_retail_electricity_price'],
	    dfs_by_loc['Pacific Noncontiguous']['avg_retail_electricity_price']
	)
	print("Degrees of freedom between is {:,} and degrees of freedom within is {:,}.".format(
	    deg_free_between, deg_free_within))
	print("The F-value must be above a critical value of {:,.3}".format(f_crit_val),
	      "based upon a {:.1%} confidence interval.".format(
	          stats.f.cdf(f_crit_val, dfn=deg_free_between, dfd=deg_free_within)))
	print("One-Way ANOVA Test F-value: {:,.1f} \t p-value: {:.3f}".format(
	    one_way_anova_result.statistic, one_way_anova_result.pvalue))


def tukey_hsd_by_loc(dfs_by_loc):
	df_list_no_us = [df for key, df in dfs_by_loc.items() if key != 'United States']
	tukey_concat = pd.concat(df_list_no_us, axis='rows')
	total_rows = [df.shape[0] for df in df_list_no_us]
	if sum(total_rows) != tukey_concat.shape[0]:
	    [print("{:<25} {:>10}".format(key, str(df.shape))) for key, df in dfs_by_loc.items()]
	    print("{:<25} {:>10}".format("tukey_concat", str(tukey_concat.shape)))

	# Run Tukey's Honest Significance Differences Test and input the data, group classifications, and significance level.
	tukey = pairwise_tukeyhsd(endog = tukey_concat['avg_retail_electricity_price'],
	                          groups = tukey_concat['location'],
	                          alpha=0.05)
	tukey_df = pd.DataFrame(tukey.summary(), columns=pd.DataFrame(tukey.summary()).iloc[0])
	tukey_df.drop(0, inplace=True)
	tukey_df.reset_index(drop=True)
	tukey_df.columns = [str(col).strip() for col in tukey_df.columns]
	for col in tukey_df.columns:
	    tukey_df[col] = tukey_df[col].astype(str).str.strip()
	#tukey_df.query('group1 == "Pacific Contiguous" or group2 == "Pacific Contiguous"')
	print("Multiple Comparison of Means - Tukey HSD, FWER=0.05")
	tukey.plot_simultaneous()
	#return tukey_df.to_latex(index=False)
	return tukey_df


def kruskal_wallis_by_loc(dfs_by_loc):
	kruskal_result = stats.kruskal(
	    dfs_by_loc['South Atlantic']['avg_retail_electricity_price'], 
	    dfs_by_loc['East North Central']['avg_retail_electricity_price'],
	    dfs_by_loc['Middle Atlantic']['avg_retail_electricity_price'],
	    dfs_by_loc['West South Central']['avg_retail_electricity_price'],
	    dfs_by_loc['Pacific Contiguous']['avg_retail_electricity_price'],
	    dfs_by_loc['Pacific Noncontiguous']['avg_retail_electricity_price']
	)
	print("\nKruskal-Wallis Test H-value: {:,.1f} \t p-value: {:.3f}\n".format(
	    kruskal_result.statistic, kruskal_result.pvalue))
	var_name = "avg_retail_electricity_price"
	print("{:<19} {:>9}".format("Location", "Median Price"))
	for loc in ["South Atlantic", "East North Central", "Middle Atlantic",
	            "West South Central", "Pacific Contiguous", "Pacific Noncontiguous"]:
	    print("{:<22} {:>9,.1f}".format(loc, dfs_by_loc[loc][var_name].median()))


def mann_whitney_rank_by_loc_pair(dfs_by_loc):
	# Generate Mann-Whitney Rank Test.
	var_name = 'avg_retail_electricity_price'
	location_combos = list(it.combinations(['South Atlantic', 'East North Central', 
	    'Middle Atlantic', 'West South Central', 'Pacific Contiguous', 'Pacific Noncontiguous'], 2))
	mann_whitney1 = {'Location 1':[], 'Location 2':[],
	                 'U Statistic 2001-2020':[], 'P Value 2001-2020': [],
	                 'U Statistic 2010-2020':[], 'P Value 2010-2020': []}
	for loc_combo in location_combos:
	    df_loc1_2001 = dfs_by_loc[loc_combo[0]][var_name]
	    df_loc2_2001 = dfs_by_loc[loc_combo[1]][var_name]
	    df_loc1_2010 = dfs_by_loc[loc_combo[0]].query("year >= 2010")[var_name]
	    df_loc2_2010 = dfs_by_loc[loc_combo[1]].query("year >= 2010")[var_name]
	    mann_whit_result_2001 = stats.mannwhitneyu(df_loc1_2001, df_loc2_2001, alternative='two-sided')
	    mann_whit_result_2010 = stats.mannwhitneyu(df_loc1_2010, df_loc2_2010, alternative='two-sided')
	    mann_whitney1['Location 1'].append(loc_combo[0])
	    mann_whitney1['Location 2'].append(loc_combo[1])
	    mann_whitney1['U Statistic 2001-2020'].append(mann_whit_result_2001.statistic)
	    mann_whitney1['P Value 2001-2020'].append(mann_whit_result_2001.pvalue)
	    mann_whitney1['U Statistic 2010-2020'].append(mann_whit_result_2010.statistic)
	    mann_whitney1['P Value 2010-2020'].append(mann_whit_result_2010.pvalue)
	mann_whitney_df1 = pd.DataFrame.from_dict(mann_whitney1)
	mann_whitney_df1['Location Pair'] = mann_whitney_df1['Location 1'] + " - " + mann_whitney_df1['Location 2']

	# Plot Mann-Whitney Rank statistics in a visual chart.
	#skew_min, skew_norm, skew_max = -1, 0, 1
	alpha = 0.05
	x_count = len(location_combos)
	fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, squeeze=True, figsize=(12,15))
	g1 = sns.barplot(x='Location Pair', y='U Statistic 2001-2020', data=mann_whitney_df1, ax=ax[0])
	ax[0].set_title("Mann-Whitney Rank U Statistic: 2001-2020")
	ax[0].tick_params(axis='x', labelrotation=75)
	g2 = sns.barplot(x='Location Pair', y='P Value 2001-2020', data=mann_whitney_df1, ax=ax[1])
	ax[1].set_title("Mann-Whitney Rank P Value: 2001-2020")
	ax[1].tick_params(axis='x', labelrotation=75)
	ax[1].plot([alpha for i in range(x_count)])
	g2 = sns.barplot(x='Location Pair', y='U Statistic 2010-2020', data=mann_whitney_df1, ax=ax[2])
	ax[2].set_title("Mann-Whitney Rank U Statistic: 2010-2020")
	ax[2].tick_params(axis='x', labelrotation=75)
	g3 = sns.barplot(x='Location Pair', y='P Value 2010-2020', data=mann_whitney_df1, ax=ax[3])
	ax[3].set_title("Mann-Whitney Rank P Value: 2010-2020")
	ax[3].tick_params(axis='x', labelrotation=75)
	ax[3].plot([alpha for i in range(x_count)])
	plt.tight_layout()
	plt.show()
	#display(mann_whitney_df1.drop('Location Pair', axis='columns'))


def energy_prices_by_geography_and_time_normality(df, dfs_by_loc):
	df.query("year >= 2010").hist(column='avg_retail_electricity_price', by='location', figsize=(12, 5), layout=(2, 4))
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.show()

	# Generate normality statistics in normality_df.
	normality_stats2 = {'location':[], 'Skewness': [], 'Kurtosis': [], 
	                   'Shapiro-Wilk W-Stat':[], 'Shapiro-Wilk P-Stat':[]}
	for location, dframe in dfs_by_loc.items():
	    desc_stats = stats.describe(dframe.query("year >= 2010")['avg_retail_electricity_price'])
	    shap_wilk_stats = stats.shapiro(dframe.query("year >= 2010")['avg_retail_electricity_price'])
	    normality_stats2['location'].append(location)
	    normality_stats2['Skewness'].append(desc_stats.skewness)
	    normality_stats2['Kurtosis'].append(desc_stats.kurtosis)
	    normality_stats2['Shapiro-Wilk W-Stat'].append(shap_wilk_stats[0])
	    normality_stats2['Shapiro-Wilk P-Stat'].append(shap_wilk_stats[1])
	normality_df2 = pd.DataFrame.from_dict(normality_stats2)
	#[print("{:<21}".format(location), dfs_by_loc[location].shape) for location in locations]

	# Plot normality statistics in a visual chart.
	skew_min, skew_norm, skew_max = -1, 0, 1
	kurt_min, kurt_norm, kurt_max = -3, 0, 3
	shww_min, shww_norm, shww_max =  0, 1, 1
	shwp_min, shwp_norm, shwp_max =  0.05, 1, 1
	x_count = len(dfs_by_loc.keys())
	fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, squeeze=True, figsize=(12,7))
	g1 = sns.barplot(x='location', y='Skewness', data=normality_df2, ax=ax[0, 0])
	ax[0, 0].set_title('Skewness')
	ax[0, 0].tick_params(axis='x', labelrotation=45)
	ax[0, 0].plot([skew_max for i in range(x_count)])
	ax[0, 0].plot([skew_min for i in range(x_count)])
	g2 = sns.barplot(x='location', y='Kurtosis', data=normality_df2, ax=ax[0, 1])
	ax[0, 1].set_title('Kurtosis')
	ax[0, 1].tick_params(axis='x', labelrotation=45)
	ax[0, 1].plot([kurt_max for i in range(x_count)])
	ax[0, 1].plot([kurt_min for i in range(x_count)])
	g2 = sns.barplot(x='location', y='Shapiro-Wilk W-Stat', data=normality_df2, ax=ax[1, 0])
	ax[1, 0].set_title('Shapiro-Wilk W-Stat')
	ax[1, 0].tick_params(axis='x', labelrotation=45)
	ax[1, 0].plot([shww_max for i in range(x_count)])
	g3 = sns.barplot(x='location', y='Shapiro-Wilk P-Stat', data=normality_df2, ax=ax[1, 1])
	ax[1, 1].set_title('Shapiro-Wilk P-Stat')
	ax[1, 1].tick_params(axis='x', labelrotation=45)
	ax[1, 1].plot([shwp_min for i in range(x_count)])
	plt.tight_layout()
	plt.show()
	#display(normality_df2)


def kruskal_wallis_by_loc_and_time(dfs_by_loc):
	# Generate Kruskal-Wallis H Test.
	var_name = 'avg_retail_electricity_price'
	kruskal_wallis2 = {'Location':[], 'H Statistic 2010':[], 'P Value 2010': [],
	                                 'H Statistic 2015':[], 'P Value 2015': []}
	for location, dframe in dfs_by_loc.items():
	    before_2010, from_2010 = dframe.query("year < 2010"), dframe.query("year >= 2010")
	    before_2015, from_2015 = dframe.query("year < 2015"), dframe.query("year >= 2015")
	    kruskal_result_2010 = stats.kruskal(before_2010[var_name], from_2010[var_name])
	    kruskal_result_2015 = stats.kruskal(before_2015[var_name], from_2015[var_name])
	    kruskal_wallis2['Location'].append(location)
	    kruskal_wallis2['H Statistic 2010'].append(kruskal_result_2010.statistic)
	    kruskal_wallis2['P Value 2010'].append(kruskal_result_2010.pvalue)
	    kruskal_wallis2['H Statistic 2015'].append(kruskal_result_2015.statistic)
	    kruskal_wallis2['P Value 2015'].append(kruskal_result_2015.pvalue)
	kruskal_wallis_df2 = pd.DataFrame.from_dict(kruskal_wallis2)


	# Plot Kruskal-Wallis statistics in a visual chart.
	#skew_min, skew_norm, skew_max = -1, 0, 1
	alpha = 0.05
	x_count = len(dfs_by_loc.keys())
	fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, squeeze=True, figsize=(12,8))
	g1 = sns.barplot(x='Location', y='H Statistic 2010', data=kruskal_wallis_df2, ax=ax[0, 0])
	ax[0, 0].set_title("Kruskal-Wallis H Statistic: 2001-2009 vs. 2010-2020")
	ax[0, 0].tick_params(axis='x', labelrotation=45)
	g2 = sns.barplot(x='Location', y='P Value 2010', data=kruskal_wallis_df2, ax=ax[0, 1])
	ax[0, 1].set_title("Kruskal-Wallis P Value: 2001-2009 vs. 2010-2020")
	ax[0, 1].tick_params(axis='x', labelrotation=45)
	ax[0, 1].plot([alpha for i in range(x_count)])
	g2 = sns.barplot(x='Location', y='H Statistic 2015', data=kruskal_wallis_df2, ax=ax[1, 0])
	ax[1, 0].set_title("Kruskal-Wallis H Statistic: 2010-2014 vs. 2015-2020")
	ax[1, 0].tick_params(axis='x', labelrotation=45)
	g3 = sns.barplot(x='Location', y='P Value 2015', data=kruskal_wallis_df2, ax=ax[1, 1])
	ax[1, 1].set_title("Kruskal-Wallis P Value: 2010-2014 vs. 2015-2020")
	ax[1, 1].tick_params(axis='x', labelrotation=45)
	ax[1, 1].plot([alpha for i in range(x_count)])
	plt.tight_layout()
	plt.show()
	#display(kruskal_wallis_df2)


def mann_whitney_by_loc_pair_and_time(dfs_by_loc):
	# Generate Mann-Whitney Rank Test.
	var_name = 'avg_retail_electricity_price'
	mann_whitney2 = {'Location':[], 'U Statistic 2010':[], 'P Value 2010': [],
	                                'U Statistic 2015':[], 'P Value 2015': []}
	for location, dframe in dfs_by_loc.items():
	    before_2010, from_2010 = dframe.query("year < 2010"), dframe.query("year >= 2010")
	    before_2015, from_2015 = dframe.query("year < 2015"), dframe.query("year >= 2015")
	    mann_whit_result_2010 = stats.mannwhitneyu(before_2010[var_name], from_2010[var_name], alternative='two-sided')
	    mann_whit_result_2015 = stats.mannwhitneyu(before_2015[var_name], from_2015[var_name], alternative='two-sided')
	    mann_whitney2['Location'].append(location)
	    mann_whitney2['U Statistic 2010'].append(mann_whit_result_2010.statistic)
	    mann_whitney2['P Value 2010'].append(mann_whit_result_2010.pvalue)
	    mann_whitney2['U Statistic 2015'].append(mann_whit_result_2015.statistic)
	    mann_whitney2['P Value 2015'].append(mann_whit_result_2015.pvalue)
	mann_whitney_df2 = pd.DataFrame.from_dict(mann_whitney2)


	# Plot Mann-Whitney Rank statistics in a visual chart.
	#skew_min, skew_norm, skew_max = -1, 0, 1
	alpha = 0.05
	x_count = len(dfs_by_loc.keys())
	fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, squeeze=True, figsize=(10,8))
	g1 = sns.barplot(x='Location', y='U Statistic 2010', data=mann_whitney_df2, ax=ax[0, 0])
	ax[0, 0].set_title("Mann-Whitney Rank U Statistic: 2001-2009 vs. 2010-2020")
	ax[0, 0].tick_params(axis='x', labelrotation=45)
	g2 = sns.barplot(x='Location', y='P Value 2010', data=mann_whitney_df2, ax=ax[0, 1])
	ax[0, 1].set_title("Mann-Whitney Rank P Value: 2001-2009 vs. 2010-2020")
	ax[0, 1].tick_params(axis='x', labelrotation=45)
	ax[0, 1].plot([alpha for i in range(x_count)])
	g2 = sns.barplot(x='Location', y='U Statistic 2015', data=mann_whitney_df2, ax=ax[1, 0])
	ax[1, 0].set_title("Mann-Whitney Rank U Statistic: 2010-2014 vs. 2015-2020")
	ax[1, 0].tick_params(axis='x', labelrotation=45)
	g3 = sns.barplot(x='Location', y='P Value 2015', data=mann_whitney_df2, ax=ax[1, 1])
	ax[1, 1].set_title("Mann-Whitney Rank P Value: 2010-2014 vs. 2015-2020")
	ax[1, 1].tick_params(axis='x', labelrotation=45)
	ax[1, 1].plot([alpha for i in range(x_count)])
	plt.tight_layout()
	plt.show()
	#display(mann_whitney_df2)


def analyze_95_ci_medians(dfs_by_loc):
	# Pacific Contiguous vs. Noncontiguous and Atlantic regions (higher power prices)

	# From https://medium.com/@wenjun.sarah.sun/bootstrap-confidence-interval-in-python-3fe8d5a6fd56
	def bootstrap_ci(df, variable, classes, repetitions = 1000, alpha = 0.05, random_state=None): 
	    
	    df = df[[variable, classes]]
	    bootstrap_sample_size = len(df) 
	    
	    mean_diffs = []
	    for i in range(repetitions):
	        bootstrap_sample = df.sample(n=bootstrap_sample_size, replace=True, random_state=random_state)
	        mean_diff = bootstrap_sample.groupby(classes).mean().iloc[1,0] - bootstrap_sample.groupby(classes).mean().iloc[0,0]
	        mean_diffs.append(mean_diff)
	    # confidence interval
	    left = np.percentile(mean_diffs, alpha/2*100)
	    right = np.percentile(mean_diffs, 100-alpha/2*100)
	    # point estimate
	    point_est = df.groupby(classes).mean().iloc[1,0] - df.groupby(classes).mean().iloc[0,0]
	    return [round(left, 2), round(point_est, 2), round(right, 2)]
	    print('Point estimate of difference between means:', round(point_est,2))
	    print((1-alpha)*100,'%','confidence interval for the difference between means:', (round(left,2), round(right,2)))

	# ci_list = []#[loc for loc in dfs_by_loc.keys() if loc != 'United States']
	# for loc, df in dfs_by_loc.items():
	#     if loc == 'United States':
	#         continue
	#     boot_df = pd.concat( [dfs_by_loc['United States'], dfs_by_loc[loc]] )
	#     boot_df = boot_df.loc[:, ['location', 'avg_retail_electricity_price']].reset_index(drop=True)
	#     temp_list = ['United States', loc]
	#     temp_list2 = bootstrap_ci(boot_df, 'avg_retail_electricity_price', 'location')
	#     temp_list.extend(temp_list2)
	#     ci_list.append(temp_list)


	ci_dict = {'location1': [], 'location2': [], 
	           'lower': [], 'meandiff': [], 'upper': []}
	var_name = 'avg_retail_electricity_price'
	for loc, df in dfs_by_loc.items():
	    if loc == 'United States':
	        continue
	    boot_df = pd.concat( [dfs_by_loc['United States'], dfs_by_loc[loc]] )
	    boot_df = boot_df.loc[:, ['location', var_name]].reset_index(drop=True)
	    ci_dict['location1'].append('United States')
	    ci_dict['location2'].append(loc)
	    temp_list = bootstrap_ci(boot_df, 'avg_retail_electricity_price', 'location')
	    us_median = dfs_by_loc['United States'][var_name].median()
	    if ( us_median < dfs_by_loc[loc][var_name].median() ) and ( temp_list[1] < 0 ):
	        temp_list = [num * -1 for num in temp_list][::-1]
	    elif ( us_median > dfs_by_loc[loc][var_name].median() ) and ( temp_list[1] > 0 ):
	        temp_list = [num * -1 for num in temp_list][::-1]
	    ci_dict['lower'].append(temp_list[0])
	    ci_dict['meandiff'].append(temp_list[1])
	    ci_dict['upper'].append(temp_list[2])

	ci_df = pd.DataFrame.from_dict(ci_dict)
	# ci_df.drop('location1', axis='columns').set_index('location2').plot.bar()
	# ci_df.head(10)

	# Modified, from https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
	def add_value_labels(ax, spacing, position):
	    """Add labels to the end of each bar in a bar chart.

	    Arguments:
	        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
	            of the plot to annotate.
	        spacing (int): The distance between the labels and the bars.
	    """

	    # For each bar: Place a label
	    for rect in ax.patches:
	        
	        # Get X and Y placement of label from rect.
	        if position == 'down':
	            y_value = rect.get_y()
	        else:
	            y_value_min = rect.get_y()
	            y_value = rect.get_y() + rect.get_height()
	        x_value = rect.get_x() + rect.get_width() / 2
	        # Number of points between bar and label. Change to your liking.
	        space = spacing
	        # Vertical alignment for positive values
	        va = 'bottom'
	        # If value of bar is negative: Place label below bar
	        if position == 'down':          #y_value < 0 or 
	            # Invert space to place label below
	            space *= -1
	            # Vertically align label at top
	            va = 'top'
	        # Use Y value as label and format number with one decimal place
	        label = "{:.2f} ~ {:.2f}".format(y_value_min, y_value)
	        # Create annotation
	        ax.annotate(
	            label,                      # Use `label` as label
	            (x_value, y_value),         # Place label at end of the bar
	            xytext=(0, space),          # Vertically shift label by `space`
	            textcoords="offset points", # Interpret `xytext` as offset in points
	            ha='center',                # Horizontally center label
	            va=va)                      # Vertically align label differently for
	                                        # positive and negative values.




	df2 = ci_df.drop('location1', axis='columns').set_index('location2')
	fig, ax = plt.subplots(figsize = (5,7))
	xlablels = df2.index

	ax.bar(xlablels, df2['upper'] - df2['lower'], bottom=df2['lower'])
	add_value_labels(ax, 3, 'up')
	ax.set_xticklabels(xlablels, rotation=45)
	#plt.xticks(rotation=30)
	#ax.tick_params(axis='x', labelrotation=45)
	fig.tight_layout()
	x_count = df2.shape[0]
	ax.plot([0 for i in range(x_count)])
	plt.title('Median Bootstrapped 95% Confidence Intervals Versus United States')
	plt.show()
	df2.head(10)