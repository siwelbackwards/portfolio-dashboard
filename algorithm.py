import pandas_datareader as pdr
import pandas as pd
from datetime import datetime
start = datetime(1990, 1, 1) #started 1990 as latest that any of these FRED IDEALLY
end = datetime.now()


#GATHERING ECONOMIC FACTOR DATA
usa_gdp = pdr.get_data_fred("FYGDP", start, end) # Usa Gross Domestic Product in billions
usa_cpi = pdr.get_data_fred("CPIAUCSL", start, end) #Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
usa_unemployment = pdr.get_data_fred('UNRATE', start, end) #Usa Unemployment rate
usa_pop = pdr.get_data_fred('CNP16OV', start, end) #Usa Population Level in 1000s
usa_interest = pdr.get_data_fred('REAINTRATREARAT1YE', start, end) #1-Year Real Interest Rate (REAINTRATREARAT1YE)
usa_oil = pdr.get_data_fred("DCOILWTICO", start, end) #West Texas Intermediate (WTI) - Cushing, Oklahoma 
asia_lng = pdr.get_data_fred("PNGASJPUSDM", start, end) #Global price of LNG, Asia
china_gdp = pdr.get_data_fred("MKTGDPCNA646NWDB", start, end) #Gross Domestic Product for China 
china_cpi = pdr.get_data_fred("CPALTT01CNM659N", start, end) #Consumer Price Index: All Items: Total for China
china_unemployment = pdr.get_data_fred("LMUNRRTTCNQ156S", start, end) #Other Labor Market Measures: Registered Unemployment: Rate: Total for China
china_pop = pdr.get_data_fred("POPTOTCNA647NWDB", start, end) #Population, Total for China
china_interest = pdr.get_data_fred("IR3TIB01CNM156N", start, end) #Interest Rates: 3-Month or 90-Day Rates and Yields: Interbank Rates: Total for China 
japan_gdp = pdr.get_data_fred("JPNNGDP", start, end) #Gross Domestic Product for Japan
japan_cpi = pdr.get_data_fred("JPNCPIALLMINMEI", start, end) #Consumer Price Index: All Items: Total for Japan
japan_unemployment = pdr.get_data_fred("LRUN64TTJPM156S", start, end) #Unemployment Rate: Aged 15-64: All Persons for Japan 
japan_pop = pdr.get_data_fred("POPTOTJPA647NWDB", start, end) #Population, Total for Japan
japan_interest = pdr.get_data_fred("IRLTLT01JPM156N", start, end) # Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) for Japan
india_gdp = pdr.get_data_fred("MKTGDPINA646NWDB", start, end) #Gross Domestic Product for India
india_cpi = pdr.get_data_fred("INDCPIALLMINMEI", start, end) #Consumer Price Index: All Items: Total for India
india_pop = pdr.get_data_fred("POPTOTINA647NWDB", start, end) #Population, Total for India
india_interest = pdr.get_data_fred("INDIRLTLT01STM", start, end) #Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) for India
uk_oil = pdr.get_data_fred("OLPRUKA", start, end) #Oil Prices in the United Kingdom
uk_gdp = pdr.get_data_fred("UKNGDP", start, end) #Gross Domestic Product for United Kingdom
uk_cpi = pdr.get_data_fred("GBRCPIALLMINMEI", start, end) #Consumer Price Index: All Items: Total for United Kingdom
uk_unemployment = pdr.get_data_fred("UNRTUKA", start, end) #Unemployment Rate in the United Kingdom
uk_pop = pdr.get_data_fred("POPUKA", start, end) #Population in the United Kingdom
uk_interest = pdr.get_data_fred("IRLTLT01GBM156N", start, end) #Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) for United Kingdom (IRLTLT01GBM156N)	
saudi_gdp = pdr.get_data_fred("MKTGDPSAA646NWDB", start, end) #Gross Domestic Product for Saudi Arabia
saudi_cpi = pdr.get_data_fred("SAUCPALTT01IXOBM", start, end) #Consumer Price Index: All Items: Total for Saudi Arabia
saudi_pop = pdr.get_data_fred("POPTOTSAA647NWDB", start, end) #Population, Total for Saudi Arabia
#saudi_interest = pdr.get_data_fred("IRLTLT01GBM156N", start, end) #Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) fo
eu_oil = pdr.get_data_fred("DCOILBRENTEU", start, end) #Crude Oil Prices: Brent - Europe
eu_cpi = pdr.get_data_fred("CP0000EZ19M086NEST", start, end) #Harmonized Index of Consumer Prices: All Items for Euro area (19 countries)
eu_interest = pdr.get_data_fred("BAMLHE00EHYIEY", start, end) #ICE BofA Euro High Yield Index Effective Yield 
eu_unemployment = pdr.get_data_fred("LRHUTTTTEZM156S", start, end) #Harmonised Unemployment - Monthly Rates: Total: All Persons for the Euro Area (19 Countries)
eu_population = pdr.get_data_fred("SPPOPTOTLEUU", start, end) #Population, Total for the European Union
eu_gdp = pdr.get_data_fred("CPMNACSCAB1GQEU272020", start, end) #Gross Domestic Product for European Union (27 Countries from 2020
saudi_oil = pd.read_csv(r"C:\Users\Lewis\Downloads\wti-crude-oil-prices-10-year-daily-chart.csv")
india_unemployment_rate_df = pd.read_csv(r"C:\Users\Lewis\Downloads\india-unemployment-rate.csv")
saudi_unemployment = pd.read_csv(r"C:\Users\Lewis\Downloads\statistic_id262524_unemployment-rate-in-saudi-arabia-in-2022.xlsx")
saudi_interest = pd.read_csv(r"C:\Users\Lewis\Downloads\estimated_interest_rates.csv")
