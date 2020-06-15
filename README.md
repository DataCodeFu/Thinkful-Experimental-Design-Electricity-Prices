# Regional Retail Electricity Prices: Target Markets with Higher Sustained Prices and 20-year Trends from January 2001 to February 2020  
  
[Open the Presentation Summary](https://github.com/DataCodeFu/Thinkful-Experimental-Design-Electricity-Prices/blob/master/Experimental%20Design%20-%20Presentation%20Summary.pdf)

## ReadMe Table of Contents
* [Project Information](#project-information)
* [Technologies](#technologies)

## Project Information
This project utilized statistical methods to analyze initial impressions of retail electricity price data by US region over the past 20 years and determine if significant differences exist between average regional pricing and US national pricing, split time periods of prices within regions, and seasonality factors by region.  Statistical methods include parametric tests such as the __One-way ANOVA__ and __Tukey's honest significant differences (HSD) test__, as well as non-parametric tests such as the __Kruskal-Wallis H test__ (one-way ANOVA on ranks) and __Mann-Whitney U test__.  The data was pulled from the Energy Information Administration (EIA) and includes 7,844 observations split into six different regional prices and a US national average price.

The power generation industry operates on long-term contracts from ten to thirty years in tenure, is a rapidly changing industry technologically and in regulation, and is exposed to significant fluctations in energy prices in merchant markets. When evaluating the regional strategy approach for a national, utility-scale power generator, we wanted to begin to understand if any particular region may show additional promise with consistently higher power prices, what its stability over time has been, and its seasonality throughout the year.  This is a starting point for further strategic analysis, as retail power prices represent a total potential economic value for the generator, utility, transmission operator, and regulatory costs.

__Retail electricity prices show consistenly higher prices__ (a significant positive difference) __in the Pacific Noncontiguous__ (_+$11.08 to $11.79_), __Middle Atlantic__ (_+$2.06 to $2.45_), and __Pacific Contiguous__ (_+$1.03 to $1.39_) regions based on bootstrapped confidence intervals of medians at 95% confidence on data spanning twenty (20) years.  __All regions showed significant differences in prices between ten-year and five-year period comparisons__, indicating no safe haven of stability in the power markets.  __The Pacific Contiguous region displayed the strongest seasonality in power prices around the summer months__ between May and October.  Pricing differences between industry sectors appears to be fixed and based on the economics of electricity transmission and distribution operations as they relate to purchase quantities.
	
## Technologies
Project is created with the following Python technologies and libraries:
 * Python version: 3.7.7
 * Numpy version: 1.18.1
 * Pandas version: 1.0.3
 * SciPy version: 1.4.1
 * StatsModels version: 0.11.0
 * Matplotlib version: 3.1.3
 * Seaborn version: 0.10.1
