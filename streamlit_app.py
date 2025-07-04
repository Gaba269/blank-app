import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import json
import time
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Portfolio Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
custom_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background-color: #f0f4ff;
}
.metric-card {
    background-color: #f8f9ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #4f46e5;
    margin: 0.5rem 0;
}
.warning-card {
    background-color: #fef3f2;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #ef4444;
    margin: 0.5rem 0;
}
.success-card {
    background-color: #f0fdf4;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #22c55e;
    margin: 0.5rem 0;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

class TickerService:
    """Service pour la recherche et validation des tickers"""

    @staticmethod
    def search_tickers(query: str, limit: int = 10) -> List[Dict]:
        """Recherche de tickers avec Yahoo Finance"""
        results = []

        # Source 1: Yahoo Finance Search API
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount={limit}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                quotes = data.get("quotes", [])

                for quote in quotes:
                    if quote.get('symbol') and quote.get('shortname'):
                        results.append({
                            'symbol': quote['symbol'],
                            'name': quote['shortname'],
                            'type': quote.get('typeDisp', 'Stock'),
                            'exchange': quote.get('exchange', 'Unknown'),
                            'source': 'Yahoo'
                        })
        except Exception as e:
            st.warning(f"Erreur lors de la recherche Yahoo: {e}")

        # Source 2: Recherche par pattern (pour les tickers connus)
        pattern_results = TickerService._pattern_search(query)
        results.extend(pattern_results)

        # Déduplication et tri
        seen = set()
        unique_results = []
        for item in results:
            if item['symbol'] not in seen:
                seen.add(item['symbol'])
                unique_results.append(item)

        return unique_results[:limit]

    @staticmethod
    def _pattern_search(query: str) -> List[Dict]:
        """Recherche par patterns pour les tickers populaires"""
        common_tickers = {
            "Microsoft": "MSFT",
            "Nvidia": "NVDA",
            "Apple Inc.": "AAPL",
            "Amazon": "AMZN",
            "Alphabet Inc. (Class C)": "GOOG",
            "Alphabet Inc. (Class A)": "GOOGL",
            "Meta Platforms": "META",
            "Broadcom": "AVGO",
            "Tesla, Inc.": "TSLA",
            "Berkshire Hathaway": "BRK.B",
            "Walmart": "WMT",
            "JPMorgan Chase": "JPM",
            "Visa Inc.": "V",
            "Lilly (Eli)": "LLY",
            "Mastercard": "MA",
            "Netflix": "NFLX",
            "Oracle Corporation": "ORCL",
            "Costco": "COST",
            "ExxonMobil": "XOM",
            "Procter & Gamble": "PG",
            "Johnson & Johnson": "JNJ",
            "Home Depot (The)": "HD",
            "Bank of America": "BAC",
            "AbbVie": "ABBV",
            "Palantir Technologies": "PLTR",
            "Coca-Cola Company (The)": "KO",
            "Philip Morris International": "PM",
            "T-Mobile US": "TMUS",
            "UnitedHealth Group": "UNH",
            "GE Aerospace": "GE",
            "Salesforce": "CRM",
            "Cisco": "CSCO",
            "Wells Fargo": "WFC",
            "IBM": "IBM",
            "Chevron Corporation": "CVX",
            "Abbott Laboratories": "ABT",
            "McDonald's": "MCD",
            "Linde plc": "LIN",
            "Intuit": "INTU",
            "ServiceNow": "NOW",
            "American Express": "AXP",
            "Morgan Stanley": "MS",
            "Walt Disney Company (The)": "DIS",
            "AT&T": "T",
            "Accenture": "ACN",
            "Intuitive Surgical": "ISRG",
            "Merck & Co.": "MRK",
            "Verizon": "VZ",
            "Goldman Sachs": "GS",
            "RTX Corporation": "RTX",
            "PepsiCo": "PEP",
            "Booking Holdings": "BKNG",
            "Advanced Micro Devices": "AMD",
            "Adobe Inc.": "ADBE",
            "Uber": "UBER",
            "Progressive Corporation": "PGR",
            "Texas Instruments": "TXN",
            "Caterpillar Inc.": "CAT",
            "Charles Schwab Corporation": "SCHW",
            "Qualcomm": "QCOM",
            "S&P Global": "SPGI",
            "Boeing": "BA",
            "Boston Scientific": "BSX",
            "Amgen": "AMGN",
            "Thermo Fisher Scientific": "TMO",
            "BlackRock": "BLK",
            "Stryker Corporation": "SYK",
            "Honeywell": "HON",
            "NextEra Energy": "NEE",
            "TJX Companies": "TJX",
            "Citigroup": "C",
            "Deere & Company": "DE",
            "Gilead Sciences": "GILD",
            "Danaher Corporation": "DHR",
            "Pfizer": "PFE",
            "Union Pacific Corporation": "UNP",
            "Automatic Data Processing": "ADP",
            "GE Vernova": "GEV",
            "Comcast": "CMCSA",
            "Palo Alto Networks": "PANW",
            "Best Buy": "BBY",
            "Avery Dennison": "AVY",
            "Hologic": "HOLX",
            "J.B. Hunt": "JBHT",
            "UDR, Inc.": "UDR",
            "IDEX Corporation": "IEX",
            "Cooper Companies (The)": "COO",
            "Textron": "TXT",
            "Jack Henry & Associates": "JKHY",
            "Masco": "MAS",
            "Align Technology": "ALGN",
            "Regency Centers": "REG",
            "TKO Group Holdings": "TKO",
            "Solventum": "SOLV",
            "Teradyne": "TER",
            "Incyte": "INCY",
            "Camden Property Trust": "CPT",
            "Allegion": "ALLE",
            "Universal Health Services": "UHS",
            "Alexandria Real Estate Equities": "ARE",
            "Healthpeak Properties": "DOC",
            "Nordson Corporation": "NDSN",
            "Juniper Networks": "JNPR",
            "J.M. Smucker Company (The)": "SJM",
            "Builders FirstSource": "BLDR",
            "Mosaic Company (The)": "MOS",
            "C.H. Robinson": "CHRW",
            "Franklin Resources": "BEN",
            "Pool Corporation": "POOL",
            "Conagra Brands": "CAG",
            "Pinnacle West": "PNW",
            "Molson Coors Beverage Company": "TAP",
            "Akamai Technologies": "AKAM",
            "Host Hotels & Resorts": "HST",
            "BXP, Inc.": "BXP",
            "Revvity": "RVTY",
            "Bunge Global": "BG",
            "LKQ Corporation": "LKQ",
            "Skyworks Solutions": "SWKS",
            "Viatris": "VTRS",
            "DaVita": "DVA",
            "Assurant": "AIZ",
            "Moderna": "MRNA",
            "Campbell Soup Company": "CPB",
            "Stanley Black & Decker": "SWK",
            "Globe Life": "GL",
            "EPAM Systems": "EPAM",
            "CarMax": "KMX",
            "Walgreens Boots Alliance": "WBA",
            "Wynn Resorts": "WYNN",
            "Dayforce": "DAY",
            "Hasbro": "HAS",
            "A. O. Smith": "AOS",
            "Eastman Chemical Company": "EMN",
            "Interpublic Group of Companies (The)": "IPG",
            "Huntington Ingalls Industries": "HII",
            "Henry Schein": "HSIC",
            "MGM Resorts": "MGM",
            "Federal Realty Investment Trust": "FRT",
            "Paramount Global": "PARA",
            "MarketAxess": "MKTX",
            "Norwegian Cruise Line Holdings": "NCLH",
            "Lamb Weston": "LW",
            "Bio-Techne": "TECH",
            "Match Group": "MTCH",
            "Generac": "GNRC",
            "AES Corporation": "AES",
            "Charles River Laboratories": "CRL",
            "Albemarle Corporation": "ALB",
            "Invesco": "IVZ",
            "Mohawk Industries": "MHK",
            "APA Corporation": "APA",
            "Caesars Entertainment": "CZR",
            "Enphase Energy": "ENPH",
            "Adidas AG": "ADS.DE",
            "Allianz SE": "ALV.DE",
            "BASF SE": "BAS.DE",
            "Bayer AG": "BAYN.DE",
            "Beiersdorf AG": "BEI.DE",
            "BMW AG": "BMW.DE",
            "Continental AG": "CON.DE",
            "Covestro AG": "1COV.DE",
            "Daimler AG": "DAI.DE",
            "Deutsche Bank AG": "DBK.DE",
            "Deutsche Boerse AG": "DB1.DE",
            "Deutsche Post AG": "DPW.DE",
            "Deutsche Telekom AG": "DTE.DE",
            "Deutsche Wohnen SE": "DWNI.DE",
            "E.ON SE": "EOAN.DE",
            "Fresenius Medical Care AG & Co. KGaA": "FME.DE",
            "Fresenius SE & Co. KGaA": "FRE.DE",
            "HeidelbergCement AG": "HEI.DE",
            "Henkel AG & Co. KGaA": "HEN3.DE",
            "Infineon Technologies AG": "IFX.DE",
            "Linde plc": "LIN.DE",
            "Merck KGaA": "MRK.DE",
            "MTU Aero Engines AG": "MTX.DE",
            "Muenchener Rueckversicherungs-Gesellschaft AG": "MUV2.DE",
            "Puma SE": "PUM.DE",
            "Qiagen N.V.": "QIA.DE",
            "RWE AG": "RWE.DE",
            "SAP SE": "SAP.DE",
            "Siemens AG": "SIE.DE",
            "Siemens Healthineers AG": "SHL.DE",
            "Symrise AG": "SY1.DE",
            "Vonovia SE": "VNA.DE",
            "Volkswagen AG": "VOW3.DE",
            "Wirecard AG": "WDI.DE",
            "Zalando SE": "ZAL.DE",
            "3i Group PLC": "III.L",
            "Admiral Group PLC": "ADM.L",
            "Anglo American PLC": "AAL.L",
            "Antofagasta PLC": "ANTO.L",
            "Ashtead Group PLC": "AHT.L",
            "Associated British Foods PLC": "ABF.L",
            "AstraZeneca PLC": "AZN.L",
            "Auto Trader Group PLC": "AUTO.L",
            "Aviva PLC": "AV/.L",
            "B&M European Value Retail S.A.": "BME.L",
            "BAE Systems PLC": "BA/.L",
            "Barratt Developments PLC": "BDEV.L",
            "Berkeley Group Holdings PLC": "BKG.L",
            "BHP Group PLC": "BHP.L",
            "BP PLC": "BP/.L",
            "British American Tobacco PLC": "BATS.L",
            "British Land Company PLC": "BLND.L",
            "Bunzl PLC": "BNZL.L",
            "Burberry Group PLC": "BRBY.L",
            "Coca-Cola Europacific Partners PLC": "CCEP.L",
            "Croda International PLC": "CRDA.L",
            "CRH PLC": "CRH.L",
            "DCC PLC": "DCC.L",
            "Diageo PLC": "DGE.L",
            "Entain PLC": "ENT.L",
            "Experian PLC": "EXPN.L",
            "Ferguson PLC": "FERG.L",
            "Fresnillo PLC": "FRES.L",
            "GlaxoSmithKline PLC": "GSK.L",
            "Glencore PLC": "GLEN.L",
            "Halma PLC": "HLMA.L",
            "Haleon PLC": "HLN.L",
            "HSBC Holdings PLC": "HSBA.L",
            "Hiscox Ltd": "HSX.L",
            "Howden Joinery Group PLC": "HWDN.L",
            "International Consolidated Airlines Group SA": "IAG.L",
            "Intermediate Capital Group PLC": "ICG.L",
            "InterContinental Hotels Group PLC": "IHG.L",
            "3i Group PLC": "III.L",
            "Imperial Brands PLC": "IMB.L",
            "IMI PLC": "IMI.L",
            "Informa PLC": "INF.L",
            "Intertek Group PLC": "ITRK.L",
            "JD Sports Fashion PLC": "JD/.L",
            "Kingfisher PLC": "KGF.L",
            "Land Securities Group PLC": "LAND.L",
            "Legal & General Group PLC": "LGEN.L",
            "Lloyds Banking Group PLC": "LLOY.L",
            "LondonMetric Property PLC": "LMP.L",
            "London Stock Exchange Group PLC": "LSEG.L",
            "Marks & Spencer Group PLC": "MKS.L",
            "Mondi PLC": "MNDI.L",
            "M&G PLC": "MNG.L",
            "Melrose Industries PLC": "MRO.L",
            "National Grid PLC": "NG/.L",
            "NatWest Group PLC": "NWG.L",
            "Next PLC": "NXT.L",
            "Polar Capital Technology Trust PLC": "PCT.L",
            "Phoenix Group Holdings PLC": "PHNX.L",
            "Prudential PLC": "PRU.L",
            "Pershing Square Holdings Ltd/Fund": "PSH.L",
            "Persimmon PLC": "PSN.L",
            "Pearson PLC": "PSON.L",
            "RELX PLC": "REL.L",
            "Rio Tinto PLC": "RIO.L",
            "Reckitt Benckiser Group PLC": "RKT.L",
            "Rightmove PLC": "RMV.L",
            "Rolls-Royce Holdings PLC": "RR/.L",
            "Rentokil Initial PLC": "RTO.L",
            "J Sainsbury PLC": "SBRY.L",
            "Schroders PLC": "SDR.L",
            "Sage Group PLC/The": "SGE.L",
            "Segro PLC": "SGRO.L",
            "Shell PLC": "SHEL.L",
            "Smiths Group PLC": "SMIN.L",
            "Scottish Mortgage Investment Trust PLC": "SMT.L",
            "Smith & Nephew PLC": "SN/.L",
            "Spirax Group PLC": "SPX.L",
            "SSE PLC": "SSE.L",
            "Standard Chartered PLC": "STAN.L",
            "St James's Place PLC": "STJ.L",
            "Severn Trent PLC": "SVT.L",
            "Tesco PLC": "TSCO.L",
            "Taylor Wimpey PLC": "TW/.L",
            "Unilever PLC": "ULVR.L",
            "UNITE Group PLC/The": "UTG.L",
            "United Utilities Group PLC": "UU/.L",
            "Vodafone Group PLC": "VOD.L",
            "Weir Group PLC/The": "WEIR.L",
            "WPP PLC": "WPP.L",
            "Whitbread PLC": "WTB.L",
            "Microsoft": "MSFT",
            "Nvidia": "NVDA",
            "Apple Inc.": "AAPL",
            "Amazon": "AMZN",
            "Alphabet Inc. (Class C)": "GOOG",
            "Alphabet Inc. (Class A)": "GOOGL",
            "Meta Platforms": "META",
            "Broadcom Inc.": "AVGO",
            "Tesla, Inc.": "TSLA",
            "Netflix": "NFLX",
            "Costco": "COST",
            "Palantir Technologies": "PLTR",
            "ASML Holding": "ASML",
            "T-Mobile US": "TMUS",
            "Cisco": "CSCO",
            "AstraZeneca": "AZN",
            "Linde plc": "LIN",
            "Intuit": "INTU",
            "Intuitive Surgical": "ISRG",
            "PepsiCo": "PEP",
            "Booking Holdings": "BKNG",
            "Advanced Micro Devices Inc.": "AMD",
            "Adobe Inc.": "ADBE",
            "Texas Instruments": "TXN",
            "Qualcomm": "QCOM",
            "Amgen": "AMGN",
            "Honeywell": "HON",
            "PDD Holdings": "PDD",
            "Gilead Sciences": "GILD",
            "Applovin Corp": "APP",
            "ADP": "ADP",
            "Arm Holdings": "ARM",
            "MercadoLibre": "MELI",
            "Comcast": "CMCSA",
            "Palo Alto Networks": "PANW",
            "Applied Materials": "AMAT",
            "CrowdStrike": "CRWD",
            "Vertex Pharmaceuticals": "VRTX",
            "Analog Devices": "ADI",
            "Micron Technology": "MU",
            "Lam Research": "LRCX",
            "MicroStrategy Inc.": "MSTR",
            "KLA Corporation": "KLAC",
            "Constellation Energy": "CEG",
            "Starbucks": "SBUX",
            "Cintas": "CTAS",
            "DoorDash": "DASH",
            "Mondelez International": "MDLZ",
            "Intel": "INTC",
            "Airbnb": "ABNB",
            "Cadence Design Systems": "CDNS",
            "O'Reilly Automotive": "ORLY",
            "Fortinet": "FTNT",
            "Marriott International": "MAR",
            "Synopsys": "SNPS",
            "PayPal": "PYPL",
            "Workday, Inc.": "WDAY",
            "Autodesk": "ADSK",
            "Monster Beverage": "MNST",
            "Roper Technologies": "ROP",
            "CSX Corporation": "CSX",
            "Axon Enterprise Inc.": "AXON",
            "Paychex": "PAYX",
            "American Electric Power": "AEP",
            "Charter Communications": "CHTR",
            "Atlassian": "TEAM",
            "Regeneron Pharmaceuticals": "REGN",
            "Marvell Technology": "MRVL",
            "Copart": "CPRT",
            "Paccar": "PCAR",
            "NXP Semiconductors": "NXPI",
            "Fastenal": "FAST",
            "Ross Stores": "ROST",
            "Keurig Dr Pepper": "KDP",
            "Exelon": "EXC",
            "Verisk": "VRSK",
            "Zscaler": "ZS",
            "Coca-Cola Europacific Partners": "CCEP",
            "Cerner": "CERN",
            "J.B. Hunt": "JBHT",
            "UDR, Inc.": "UDR",
            "IDEX Corporation": "IEX",
            "Cooper Companies (The)": "COO",
            "Textron": "TXT",
            "Jack Henry & Associates": "JKHY",
            "Masco": "MAS",
            "Align Technology": "ALGN",
            "Regency Centers": "REG",
            "TKO Group Holdings": "TKO",
            "Solventum": "SOLV",
            "Teradyne": "TER",
            "Incyte": "INCY",
            "Camden Property Trust": "CPT",
            "Allegion": "ALLE",
            "Universal Health Services": "UHS",
            "Alexandria Real Estate Equities": "ARE",
            "Healthpeak Properties": "DOC",
            "Nordson Corporation": "NDSN",
            "Juniper Networks": "JNPR",
            "J.M. Smucker Company (The)": "SJM",
            "Builders FirstSource": "BLDR",
            "Mosaic Company (The)": "MOS",
            "C.H. Robinson": "CHRW",
            "Franklin Resources": "BEN",
            "Pool Corporation": "POOL",
            "Conagra Brands": "CAG",
            "Pinnacle West": "PNW",
            "Molson Coors Beverage Company": "TAP",
            "Akamai Technologies": "AKAM",
            "Host Hotels & Resorts": "HST",
            "BXP, Inc.": "BXP",
            "Revvity": "RVTY",
            "Bunge Global": "BG",
            "LKQ Corporation": "LKQ",
            "Skyworks Solutions": "SWKS",
            "Viatris": "VTRS",
            "DaVita": "DVA",
            "Assurant": "AIZ",
            "Moderna": "MRNA",
            "Campbell Soup Company": "CPB",
            "Stanley Black & Decker": "SWK",
            "Globe Life": "GL",
            "EPAM Systems": "EPAM",
            "CarMax": "KMX",
            "Walgreens Boots Alliance": "WBA",
            "Wynn Resorts": "WYNN",
            "Dayforce": "DAY",
            "Hasbro": "HAS",
            "A. O. Smith": "AOS",
            "Eastman Chemical Company": "EMN",
            "Interpublic Group of Companies (The)": "IPG",
            "Huntington Ingalls Industries": "HII",
            "Henry Schein": "HSIC",
            "MGM Resorts": "MGM",
            "Federal Realty Investment Trust": "FRT",
            "Paramount Global": "PARA",
            "MarketAxess": "MKTX",
            "Norwegian Cruise Line Holdings": "NCLH",
            "Lamb Weston": "LW",
            "Bio-Techne": "TECH",
            "Match Group": "MTCH",
            "Generac": "GNRC",
            "AES Corporation": "AES",
            "Charles River Laboratories": "CRL",
            "Albemarle Corporation": "ALB",
            "Invesco": "IVZ",
            "Mohawk Industries": "MHK",
            "APA Corporation": "APA",
            "Caesars Entertainment": "CZR",
            "Enphase Energy": "ENPH",
            "Air France KLM": "AF.PA",
            "Accor": "AC.PA",
            "Air Liquide": "AI.PA",
            "Capgemini": "CAP.PA",
            "AXA": "CS.PA",
            "Danone": "BN.PA",
            "Engie": "ENGI.PA",
            "BNP Paribas": "BNP.PA",
            "Bouygues": "EN.PA",
            "LVMH": "MC.PA",
            "L'Oréal": "OR.PA",
            "Schneider Electric": "SU.PA",
            "Saint-Gobain": "SGO.PA",
            "Carrefour": "CA.PA",
            "EssilorLuxottica": "EL.PA",
            "Pernod Ricard": "RI.PA",
            "Crédit Agricole": "ACA.PA",
            "Vallourec": "VK.PA",
            "Orange": "ORA.PA",
            "Société Générale": "GLE.PA",
            "Michelin": "ML.PA",
            "Airbus Group": "AIR.PA",
            "Alstom": "ALO.PA",
            "Sanofi": "SAN.PA",
            "Vivendi": "VIV.PA",
            "TotalEnergies SE": "TTE.PA",
            "STMicroelectronics": "STM",
            "Vinci": "DG.PA",
            "Renault": "RNO.PA",
            "Veolia Environnement": "VIE.PA",
            "Solvay": "SOLB.PA",
            "Kering": "KER.PA",
            "Unibail-Rodamco-Westfield": "URW.PA",
            "ArcelorMittal": "MT.PA",
            "Métropole Télévision": "MMT.PA",
            "Eramet": "ERA.PA",
            "Thales": "HO.PA",
            "Gecina": "GFC.PA",
            "Dassault Systèmes": "DSY.PA",
            "Forvia": "FVIA.PA",
            "Aéroports de Paris": "ADP.PA",
            "Eurofins Scientific": "ERF.PA",
            "Imerys": "NK.PA",
            "Rexel": "RXL.PA",
            "Rémy Cointreau": "RCO.PA",
            "SES": "SESL.PA",
            "Virbac": "VIRP.PA",
            "Publicis Groupe": "PUB.PA",
            "Arkema": "AKE.PA",
            "Sodexo": "SW.PA",
            "Ubisoft Entertainment": "UBI.PA",
            "Hermès International": "RMS.PA",
            "Soitec": "SOI.PA",
            "Safran": "SAF.PA",
            "Wendel": "MF.PA",
            "Mercialys": "MERY.PA",
            "Eiffage": "FGR.PA",
            "Groupe SEB": "SK.PA",
            "Trigano": "TRI.PA",
            "SCOR": "SCR.PA",
            "Sartorius Stedim Biotech": "DIM.PA",
            "Covivio": "COV.PA",
            "Eutelsat Communications": "ETL.PA",
            "Atos": "ATO.PA",
            "Valeo": "FR.PA",
            "Plastic Omnium": "POXY.PA",
            "Nexans": "NEX.PA",
            "BIC": "BB.PA",
            "Ipsos": "IPS.PA",
            "Alten": "ATE.PA",
            "CGG": "CGG.PA",
            "Nexity": "NXI.PA",
            "Beneteau": "BEN.PA",
            "Legrand": "LR.PA",
            "Klepierre": "LI.PA",
            "Mersen": "MRN.PA",
            "Ipsen": "IPN.PA",
            "Orpea": "ORP.PA",
            "JCDecaux": "DEC.PA",
            "BioMérieux": "BIM.PA",
            "TF1 Group": "TFI.PA",
            "Teleperformance": "TEF.PA",
            "Sopra Steria Group": "SOP.PA",
            "Rubis": "RUI.PA",
            "Inter Parfums": "IP.PA",
            "Lectra": "LSS.PA",
            "Bureau Veritas": "BVI.PA",
            "Derichebourg": "DBG.PA",
            "Icade": "ICAD.PA",
            "Getlink": "GET.PA",
            "Edenred": "EDEN.PA",
            "Bolloré": "BOL.PA",
            "Aperam": "APAM.PA",
            "Argan": "ARGS.PA",
            "Carmila": "CARM.PA",
            "Dassault Aviation": "AM.PA",
            "VusionGroup": "VUSN.PA",
            "Valneva": "VLA.PA",
            "Clariane": "CLAR.PA",
            "Eurazeo": "RF.PA",
            "ID Logistics": "IDL.PA",
            "Fnac Darty": "FNAC.PA",
            "Spie": "SPIE.PA",
            "Solutions 30": "S30.PA",
            "Coface": "COFA.PA",
            "Gaztransport & Technigaz": "GTT.PA",
            "Elior Group": "ELOR.PA",
            "Euronext": "ENX.PA",
            "Elis": "ELIS.PA",
            "Voltalia": "VLTSA.PA",
            "Worldline": "WLN.PA",
            "Amundi": "AMUN.PA",
            "X-Fab Silicon Foundries": "XFAB.PA",
            "Ayvens": "ALD.PA",
            "Verallia": "VRLA.PA",
            "FDJ": "FDJ.PA",
            "Stellantis": "STLA",
            "Technip Energies": "TE.PA",
            "Euroapi": "EAPI.PA",
            "Bitcoin": "BTC",
            "Ethereum": "ETH",
            "Tether": "USDT",
            "XRP": "XRP",
            "BNB": "BNB",
            "Solana": "SOL",
            "USD Coin": "USDC",
            "TRON": "TRX",
            "Dogecoin": "DOGE",
            "Cardano": "ADA",
            "Wrapped Bitcoin": "WBTC",
            "Bitcoin Cash": "BCH",
            "Sui": "SUI",
            "Chainlink": "LINK",
            "UNUS SED LEO": "LEO",
            "Avalanche": "AVAX",
            "Stellar": "XLM",
            "Toncoin": "TON",
            "Shiba Inu": "SHIB",
            "Litecoin": "LTC",
            "WhiteBIT Token": "WBT",
            "Hedera": "HBAR",
            "Monero": "XMR",
            "Dai": "DAI",
            "Bitget Token": "BGB",
            "Polkadot": "DOT",
            "Uniswap": "UNI",
            "Aave": "AAVE",
            "Pepe": "PEPE",
            "OKB": "OKB",
            "Aptos": "APT",
            "Bittensor": "TAO",
            "NEAR Protocol": "NEAR",
            "Internet Computer": "ICP",
            "Cronos": "CRO",
            "Ethereum Classic": "ETC",
            "Ondo": "ONDO",
            "Kaspa": "KAS",
            "Mantle": "MNT",
            "Fasttoken": "FTN",
            "GateToken": "GT",
            "Polygon Ecosystem Token": "POL",
            "VeChain": "VET",
            "OFFICIAL TRUMP": "TRUMP",
            "Tokenize Xchange": "TKX",
            "Arbitrum": "ARB",
            "Artificial Superintelligence Alliance": "FET",
            "Render Token": "RENDER",
            "Ethena": "ENA",
            "Cosmos": "ATOM",
            'Zalando': 'ZAL'
        }
        query_upper = query.upper()
        results = []

        for name, symbol in common_tickers.items():
            if query_upper in name or query_upper in symbol:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    results.append({
                        'symbol': symbol,
                        'name': info.get('shortName', name),
                        'type': 'Stock',
                        'exchange': info.get('exchange', 'Unknown'),
                        'source': 'Pattern'
                    })
                except:
                    continue

        return results

    @staticmethod
    def validate_ticker(symbol: str) -> Dict:
        """Validation d'un ticker avec données financières"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                hist = ticker.history(period="5d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            if not current_price:
                return {'valid': False, 'error': 'Prix indisponible'}
            return {
                'valid': True,
                'symbol': symbol,
                'name': info.get('shortName', symbol),
                'price': float(current_price),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'isin': info.get('isin', 'Unknown'),
                'type': TickerService._classify_asset_type(info)
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    @staticmethod
    def _classify_asset_type(info: Dict) -> str:
        """Classification automatique du type d'actif"""
        symbol = info.get('symbol', '')
        sector = info.get('sector', '').lower()
        industry = info.get('industry', '').lower()
        name = info.get('shortName', '').lower()

        if any(x in symbol for x in ['-USD', '-EUR']) or 'crypto' in name:
            return 'Cryptocurrency'
        if any(x in name for x in ['etf', 'fund', 'index']):
            return 'ETF'
        if 'real estate' in sector or 'reit' in name:
            return 'REIT'
        sector_mapping = {
            'technology': 'Tech Stock',
            'healthcare': 'Healthcare Stock',
            'financial': 'Financial Stock',
            'energy': 'Energy Stock',
            'consumer': 'Consumer Stock',
            'industrial': 'Industrial Stock',
            'utilities': 'Utility Stock',
            'materials': 'Materials Stock',
            'telecommunication': 'Telecom Stock'
        }
        for key, value in sector_mapping.items():
            if key in sector:
                return value
        return 'Stock'

class DiversificationAnalyzer:
    """Analyseur de diversification"""

    @staticmethod
    def calculate_concentration_metrics(df: pd.DataFrame) -> Dict:
        """Calcule les métriques de concentration"""
        if 'weight' not in df.columns:
            return {
                'hhi': 0,
                'effective_stocks': 0,
                'top3_concentration': 0,
                'entropy_ratio': 0,
                'concentration_level': 'Non calculé'
            }
        weights = df['weight'].values
        hhi = np.sum(weights ** 2)
        effective_stocks = 1 / hhi if hhi > 0 else 0
        top3_weight = weights[np.argsort(weights)[-3:]].sum()
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = -np.log(1/len(weights))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return {
            'hhi': hhi,
            'effective_stocks': effective_stocks,
            'top3_concentration': top3_weight,
            'entropy_ratio': normalized_entropy,
            'concentration_level': DiversificationAnalyzer._get_concentration_level(hhi)
        }

    @staticmethod
    def _get_concentration_level(hhi: float) -> str:
        """Détermine le niveau de concentration"""
        if hhi > 0.25:
            return "Très Concentré"
        elif hhi > 0.15:
            return "Concentré"
        elif hhi > 0.10:
            return "Modérément Concentré"
        else:
            return "Bien Diversifié"

    @staticmethod
    def analyze_sector_diversification(df: pd.DataFrame) -> pd.DataFrame:
        """Analyse la diversification sectorielle"""
        if 'sector' not in df.columns or 'weight' not in df.columns:
            return pd.DataFrame()
        sector_analysis = df.groupby('sector').agg({
            'weight': 'sum',
            'amount': 'sum',
            'perf': 'mean',
            'name': 'count'
        }).round(4)
        sector_analysis.columns = ['Weight', 'Amount', 'Avg_Performance', 'Count']
        sector_analysis['Weight_Pct'] = sector_analysis['Weight'] * 100
        return sector_analysis.sort_values('Weight', ascending=False)

    @staticmethod
    def analyze_geographic_diversification(df: pd.DataFrame) -> pd.DataFrame:
        """Analyse la diversification géographique"""
        if 'symbol' not in df.columns or 'weight' not in df.columns:
            return pd.DataFrame()
        symbol_to_region = {
            '': 'USA',
            '.US': 'USA',
            '.PA': 'France',
            '.L': 'UK',
            '.DE': 'Germany',
            '.MI': 'Italy',
            '.AS': 'Netherlands',
            '.SW': 'Switzerland',
            '.MC': 'Spain',
            '.BR': 'Belgium',
            '.VI': 'Austria',
            '.HE': 'Finland',
            '.ST': 'Sweden',
            '.OL': 'Norway',
            '.CO': 'Denmark',
            '.T': 'Japan',
            '.HK': 'Hong Kong',
            '.SS': 'China',
            '.SZ': 'China',
            '.KS': 'South Korea',
            '.SI': 'Singapore',
            '.AX': 'Australia',
            '.NZ': 'New Zealand',
            '.TO': 'Canada',
            '.V': 'Canada',
            '.SA': 'Brazil',
            '.MX': 'Mexico',
            '.JO': 'South Africa',
            '.TA': 'Israel',
        }
        def get_region_from_symbol(symbol):
            if pd.isna(symbol) or symbol == '':
                return 'Unknown'
            symbol = str(symbol).upper()
            for suffix, region in symbol_to_region.items():
                if suffix == '':
                    continue
                elif symbol.endswith(suffix):
                    return region
            if any(pattern in symbol for pattern in ['BTC', 'ETH', 'ADA', 'DOT']):
                return 'Cryptocurrency'
            elif len(symbol) <= 5 and '.' not in symbol:
                return 'USA'
            else:
                return 'Other'
        df_copy = df.copy()
        df_copy['region'] = df_copy['symbol'].apply(get_region_from_symbol)
        geo_analysis = df_copy.groupby('region').agg({
            'weight': 'sum',
            'amount': 'sum',
            'perf': 'mean',
            'name': 'count'
        }).round(4)
        geo_analysis.columns = ['Weight', 'Amount', 'Avg_Performance', 'Count']
        geo_analysis['Weight_Pct'] = geo_analysis['Weight'] * 100
        return geo_analysis.sort_values('Weight', ascending=False)

class PortfolioManager:
    """Gestionnaire de portefeuille"""

    def __init__(self):
        if 'portfolio_df' not in st.session_state:
            st.session_state.portfolio_df = pd.DataFrame()

    @staticmethod
    def calculate_annualized_return(initial_value: float, final_value: float, days_held: int) -> float:
        """Calcule le rendement annualisé"""
        if initial_value <= 0 or days_held <= 0:
            return 0.0
        total_return = (final_value / initial_value) - 1
        years_held = days_held / 365.25
        if years_held > 0:
            try:
                annualized_return = ((1 + total_return) ** (1 / years_held)) - 1
            except (OverflowError, ZeroDivisionError):
                annualized_return = total_return
        else:
            annualized_return = total_return
        return annualized_return * 100

    def add_stock_to_portfolio(self, ticker_data: Dict, quantity: int, buying_price: float = None, purchase_date=None):
        """Ajoute une action au portefeuille"""
        purchase_price = buying_price if buying_price is not None else ticker_data['price']
        if purchase_date is None:
            purchase_date = datetime.now().date()
        elif isinstance(purchase_date, str):
            try:
                purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d').date()
            except ValueError:
                purchase_date = datetime.now().date()
        new_row = {
            'name': ticker_data['name'],
            'symbol': ticker_data['symbol'],
            'isin': ticker_data.get('isin', 'Unknown'),
            "purchase_date": purchase_date,
            'quantity': quantity,
            'buyingPrice': purchase_price,
            'lastPrice': ticker_data['price'],
            'currency': ticker_data.get('currency', 'USD'),
            'exchange': ticker_data.get('exchange', 'Unknown'),
            'sector': ticker_data.get('sector', 'Unknown'),
            'industry': ticker_data.get('industry', 'Unknown'),
            'asset_type': ticker_data.get('type', 'Stock'),
            'intradayVariation': 0.0,
            'amount': quantity * ticker_data['price'],
            'amountVariation': quantity * (ticker_data['price'] - purchase_price),
            'variation': ((ticker_data['price'] - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0.0,
            'Tickers': ticker_data['symbol']
        }
        if st.session_state.portfolio_df.empty:
            st.session_state.portfolio_df = pd.DataFrame([new_row])
        else:
            st.session_state.portfolio_df = pd.concat([
                st.session_state.portfolio_df,
                pd.DataFrame([new_row])
            ], ignore_index=True)
        return True

    def update_portfolio_metrics(self):
        """Met à jour toutes les métriques du portefeuille"""
        if st.session_state.portfolio_df.empty:
            return {
                'total_value': 0,
                'portfolio_performance': 0,
                'annualized_return': 0,
                'weighted_annualized_return': 0
            }
        df = st.session_state.portfolio_df.copy()
        current_date = datetime.now().date()
        total_value = df['amount'].sum()
        if total_value > 0:
            df['weight'] = df['amount'] / total_value
            df['weight_pct'] = df['weight'] * 100
        else:
            df['weight'] = 0
            df['weight_pct'] = 0
        df['perf'] = ((df['lastPrice'] - df['buyingPrice']) / df['buyingPrice'] * 100).fillna(0)
        df['days_held'] = df['purchase_date'].apply(
            lambda x: max(1, (current_date - x).days) if pd.notna(x) else 1
        )
        annualized_returns = []
        for _, row in df.iterrows():
            initial_val = row['buyingPrice'] * row['quantity']
            final_val = row['lastPrice'] * row['quantity']
            days = row['days_held']
            ann_return = self.calculate_annualized_return(initial_val, final_val, days)
            annualized_returns.append(ann_return)
        df['annualized_return'] = annualized_returns
        portfolio_perf = (df['weight'] * df['perf']).sum()
        weighted_annualized_return = (df['weight'] * df['annualized_return']).sum()
        total_initial_value = (df['buyingPrice'] * df['quantity']).sum()
        total_current_value = (df['lastPrice'] * df['quantity']).sum()
        weighted_days_held = (df['weight'] * df['days_held']).sum()
        portfolio_annualized_return = self.calculate_annualized_return(
            total_initial_value,
            total_current_value,
            max(1, int(weighted_days_held))
        )
        st.session_state.portfolio_df = df
        return {
            'total_value': total_value,
            'portfolio_performance': portfolio_perf,
            'annualized_return': portfolio_annualized_return,
            'weighted_annualized_return': weighted_annualized_return,
            'total_initial_value': total_initial_value,
            'total_current_value': total_current_value,
            'weighted_days_held': weighted_days_held
        }

    def get_portfolio_annualized_metrics(self) -> Dict:
        """Retourne les métriques annualisées détaillées du portefeuille"""
        metrics = self.update_portfolio_metrics()
        df = st.session_state.portfolio_df
        if df.empty:
            return metrics
        current_date = datetime.now().date()
        min_purchase_date = df['purchase_date'].min()
        max_purchase_date = df['purchase_date'].max()
        portfolio_age_days = (current_date - min_purchase_date).days if pd.notna(min_purchase_date) else 0
        portfolio_age_years = portfolio_age_days / 365.25
        if len(df) > 1:
            individual_returns = df['annualized_return'].values
            portfolio_volatility = np.std(individual_returns)
        else:
            portfolio_volatility = 0
        risk_free_rate = 2.0
        excess_return = metrics['annualized_return'] - risk_free_rate
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        metrics.update({
            'portfolio_age_days': portfolio_age_days,
            'portfolio_age_years': portfolio_age_years,
            'min_purchase_date': min_purchase_date,
            'max_purchase_date': max_purchase_date,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'risk_free_rate': risk_free_rate,
            'excess_return': excess_return
        })
        return metrics

    def display_annualized_performance(self):
        """Affiche les performances annualisées dans Streamlit"""
        metrics = self.get_portfolio_annualized_metrics()
        if metrics['total_value'] == 0:
            st.warning("Aucune position dans le portefeuille")
            return
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Valeur totale",
                f"${metrics['total_value']:,.2f}",
                f"${metrics.get('total_current_value', 0) - metrics.get('total_initial_value', 0):,.2f}"
            )
        with col2:
            st.metric(
                "Performance totale",
                f"{metrics['portfolio_performance']:.2f}%"
            )
        with col3:
            st.metric(
                "Rendement annualisé",
                f"{metrics['annualized_return']:.2f}%"
            )
        with col4:
            st.metric(
                "Ratio de Sharpe",
                f"{metrics['sharpe_ratio']:.2f}"
            )
        st.subheader("Détails des performances")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Âge du portefeuille:** {metrics['portfolio_age_years']:.1f} ans")
            st.write(f"**Volatilité estimée:** {metrics['portfolio_volatility']:.2f}%")
            st.write(f"**Rendement excédentaire:** {metrics['excess_return']:.2f}%")
        with col2:
            st.write(f"**Première acquisition:** {metrics['min_purchase_date']}")
            st.write(f"**Dernière acquisition:** {metrics['max_purchase_date']}")
            st.write(f"**Taux sans risque:** {metrics['risk_free_rate']:.1f}%")
        if not st.session_state.portfolio_df.empty:
            st.subheader("Détail par position")
            display_df = st.session_state.portfolio_df[[
                'symbol', 'quantity', 'buyingPrice', 'lastPrice',
                'perf', 'annualized_return', 'days_held', 'weight_pct'
            ]].copy()
            display_df.columns = [
                'Symbole', 'Quantité', 'Prix d\'achat', 'Prix actuel',
                'Performance (%)', 'Rendement annualisé (%)', 'Jours détention', 'Poids (%)'
            ]
            display_df['Prix d\'achat'] = display_df['Prix d\'achat'].apply(lambda x: f"${x:.2f}")
            display_df['Prix actuel'] = display_df['Prix actuel'].apply(lambda x: f"${x:.2f}")
            display_df['Performance (%)'] = display_df['Performance (%)'].apply(lambda x: f"{x:.2f}%")
            display_df['Rendement annualisé (%)'] = display_df['Rendement annualisé (%)'].apply(lambda x: f"{x:.2f}%")
            display_df['Poids (%)'] = display_df['Poids (%)'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df, use_container_width=True)

def generate_recommendations(df: pd.DataFrame, concentration: Dict,
                           sector_analysis: pd.DataFrame, geo_analysis: pd.DataFrame):
    """Génère des recommandations personnalisées"""
    recommendations = []
    if concentration['hhi'] > 0.25:
        recommendations.append({
            'type': 'warning',
            'title': '⚠️ Concentration excessive',
            'message': f"Votre portefeuille est très concentré (HHI: {concentration['hhi']:.3f}). "
                      f"Considérez réduire vos 3 plus grosses positions qui représentent "
                      f"{concentration['top3_concentration']:.1%} du total."
        })
    if not sector_analysis.empty:
        max_sector = sector_analysis.iloc[0]
        if max_sector['Weight_Pct'] > 40:
            recommendations.append({
                'type': 'warning',
                'title': '🏭 Concentration sectorielle',
                'message': f"Le secteur '{max_sector.name}' représente {max_sector['Weight_Pct']:.1f}% "
                          f"de votre portefeuille. Diversifiez vers d'autres secteurs."
            })
    if not geo_analysis.empty:
        max_region = geo_analysis.iloc[0]
        if max_region['Weight_Pct'] > 70:
            recommendations.append({
                'type': 'info',
                'title': '🌍 Diversification géographique',
                'message': f"Votre exposition à la région '{max_region.name}' est de {max_region['Weight_Pct']:.1f}%. "
                          f"Considérez une exposition internationale plus large."
            })
    if len(df) < 10:
        recommendations.append({
            'type': 'info',
            'title': '📊 Nombre de positions',
            'message': f"Avec {len(df)} positions, votre portefeuille pourrait bénéficier de plus de diversification. "
                      f"Considérez ajouter 5-10 positions supplémentaires pour réduire le risque spécifique."
        })
    elif len(df) > 50:
        recommendations.append({
            'type': 'warning',
            'title': '📊 Trop de positions',
            'message': f"Avec {len(df)} positions, votre portefeuille pourrait être trop complexe à gérer. "
                      f"Considérez consolider vers 20-30 positions principales."
        })
    if 'perf' in df.columns and len(df) > 0:
        avg_perf = df['perf'].mean()
        perf_std = df['perf'].std()
        if perf_std > 50:
            recommendations.append({
                'type': 'warning',
                'title': '📈 Volatilité élevée',
                'message': f"La volatilité de vos positions est élevée (écart-type: {perf_std:.1f}%). "
                          f"Considérez ajouter des actifs plus stables (obligations, dividendes)."
            })
        losing_positions = df[df['perf'] < -20]
        if len(losing_positions) > len(df) * 0.3:
            recommendations.append({
                'type': 'warning',
                'title': '📉 Positions perdantes',
                'message': f"{len(losing_positions)} positions affichent des pertes > 20%. "
                          f"Évaluez si certaines doivent être soldées pour limiter les pertes."
            })
    if concentration['hhi'] < 0.10 and len(df) >= 15:
        recommendations.append({
            'type': 'success',
            'title': '✅ Bonne diversification',
            'message': "Votre portefeuille présente une bonne diversification avec un risque de concentration faible."
        })
    if not sector_analysis.empty and len(sector_analysis) >= 5:
        recommendations.append({
            'type': 'success',
            'title': '✅ Diversification sectorielle',
            'message': f"Excellente diversification avec {len(sector_analysis)} secteurs représentés."
        })
    if recommendations:
        for rec in recommendations:
            if rec['type'] == 'warning':
                st.markdown(f"""
                <div class="warning-card">
                    <h4>{rec['title']}</h4>
                    <p>{rec['message']}</p>
                </div>
                """, unsafe_allow_html=True)
            elif rec['type'] == 'success':
                st.markdown(f"""
                <div class="success-card">
                    <h4>{rec['title']}</h4>
                    <p>{rec['message']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{rec['title']}</h4>
                    <p>{rec['message']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Aucune recommandation spécifique pour le moment.")

def enhance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Améliore automatiquement un DataFrame importé"""
    column_mapping = {
        'name': ['name', 'nom', 'title', 'security', 'instrument'],
        'quantity': ['quantity', 'qty', 'quantite', 'shares', 'units'],
        'date': ['purchase_date', 'date'],
        'buyingPrice': ['buyingPrice', 'prix_achat', 'purchase_price', 'cost'],
        'lastPrice': ['lastPrice', 'prix_actuel', 'current_price', 'market_price'],
        'isin': ['isin', 'ISIN'],
        'symbol': ['symbol', 'ticker', 'symbole']
    }
    df_enhanced = df.copy()
    for standard_col, possible_names in column_mapping.items():
        for col in df_enhanced.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                df_enhanced = df_enhanced.rename(columns={col: standard_col})
                break
    required_columns = {
        'isin': 'Unknown',
        'symbol': '',
        'currency': 'EUR',
        'exchange': 'Unknown',
        'sector': 'Unknown',
        'industry': 'Unknown',
        'asset_type': 'Stock',
        'intradayVariation': 0.0,
        'amountVariation': 0.0,
        'variation': 0.0
    }
    for col, default_value in required_columns.items():
        if col not in df_enhanced.columns:
            df_enhanced[col] = default_value
    if 'amount' not in df_enhanced.columns and 'quantity' in df_enhanced.columns and 'lastPrice' in df_enhanced.columns:
        df_enhanced['amount'] = df_enhanced['quantity'] * df_enhanced['lastPrice']
    if 'symbol' in df_enhanced.columns and 'name' in df_enhanced.columns:
        for idx, row in df_enhanced.iterrows():
            if not row['symbol'] or row['symbol'] == '':
                try:
                    search_results = TickerService.search_tickers(row['name'], limit=1)
                    if search_results:
                        df_enhanced.at[idx, 'symbol'] = search_results[0]['symbol']
                        ticker_data = TickerService.validate_ticker(search_results[0]['symbol'])
                        if ticker_data['valid']:
                            df_enhanced.at[idx, 'sector'] = ticker_data.get('sector', 'Unknown')
                            df_enhanced.at[idx, 'industry'] = ticker_data.get('industry', 'Unknown')
                            df_enhanced.at[idx, 'asset_type'] = ticker_data.get('type', 'Stock')
                            df_enhanced.at[idx, 'exchange'] = ticker_data.get('exchange', 'Unknown')
                except:
                    continue
    if 'Tickers' not in df_enhanced.columns and 'symbol' in df_enhanced.columns:
        df_enhanced['Tickers'] = df_enhanced['symbol']
    return df_enhanced

def display_portfolio_summary(df: pd.DataFrame):
    """Affiche un résumé avancé du portefeuille"""
    st.header("📋 Résumé du portefeuille")
    required_cols = ['name', 'weight_pct', 'perf']
    if not all(col in df.columns for col in required_cols):
        st.warning("Certaines données nécessaires manquent pour afficher le résumé complet.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔝 Top 5 positions")
        top5 = df.nlargest(5, 'weight_pct')[['name', 'weight_pct', 'perf']]
        top5.columns = ['Action', 'Poids (%)', 'Performance (%)']
        st.dataframe(top5.style.format({
            'Poids (%)': '{:.1f}',
            'Performance (%)': '{:.2f}'
        }), use_container_width=True)
    with col2:
        st.subheader("📉 Plus fortes baisses")
        worst5 = df.nsmallest(5, 'perf')[['name', 'weight_pct', 'perf']]
        worst5.columns = ['Action', 'Poids (%)', 'Performance (%)']
        st.dataframe(worst5.style.format({
            'Poids (%)': '{:.1f}',
            'Performance (%)': '{:.2f}'
        }), use_container_width=True)

class EfficientFrontier:
    """Classe pour le calcul de la frontière efficiente"""

    @staticmethod
    def get_historical_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Récupère les données historiques pour les symboles donnés"""
        try:
            data = yf.download(symbols, start=start_date, end=end_date)['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame()
                data.columns = symbols
            data = data.dropna(thresh=len(data) * 0.7, axis=1)
            return data.dropna()
        except Exception as e:
            print(f"Erreur lors du téléchargement des données: {e}")
            return pd.DataFrame()

    @staticmethod
    def calculate_portfolio_performance(weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, float]:
        """Calcule le rendement et la volatilité du portefeuille"""
        portfolio_return = np.sum(weights * returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        return portfolio_return, portfolio_volatility

    @staticmethod
    def negative_sharpe_ratio(weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Fonction objectif pour maximiser le ratio de Sharpe (on minimise le négatif)"""
        portfolio_return, portfolio_volatility = EfficientFrontier.calculate_portfolio_performance(weights, returns, cov_matrix)
        if portfolio_volatility == 0:
            return -np.inf
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    @staticmethod
    def get_efficient_frontier(symbols: List[str], start_date: str, end_date: str, risk_free_rate: float = 0.02) -> Tuple[pd.DataFrame, Dict]:
        """Calcule le portefeuille optimal sur la frontière efficiente"""
        try:
            price_data = EfficientFrontier.get_historical_data(symbols, start_date, end_date)
            if price_data.empty or len(price_data.columns) < 2:
                return pd.DataFrame(), {'error': 'Données insuffisantes'}
            returns = price_data.pct_change().dropna()
            if len(returns) < 30:
                return pd.DataFrame(), {'error': 'Historique trop court (moins de 30 jours)'}
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            if np.any(np.isnan(cov_matrix.values)) or np.any(np.isinf(cov_matrix.values)):
                return pd.DataFrame(), {'error': 'Matrice de covariance invalide'}
            num_assets = len(symbols)
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_weights = np.array([1/num_assets] * num_assets)
            result = minimize(
                EfficientFrontier.negative_sharpe_ratio,
                initial_weights,
                args=(mean_returns.values, cov_matrix.values, risk_free_rate),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            if not result.success:
                return pd.DataFrame(), {'error': f'Optimisation échouée: {result.message}'}
            optimal_weights = result.x
            portfolio_return, portfolio_volatility = EfficientFrontier.calculate_portfolio_performance(
                optimal_weights, mean_returns.values, cov_matrix.values
            )
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            results_df = pd.DataFrame({
                'weight': optimal_weights
            }, index=symbols)
            results_df = results_df[results_df['weight'] > 0.005].sort_values('weight', ascending=False)
            metrics = {
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
            return results_df, metrics
        except Exception as e:
            return pd.DataFrame(), {'error': f'Erreur lors du calcul: {str(e)}'}

    @staticmethod
    def generate_efficient_frontier_curve(symbols: List[str], start_date: str, end_date: str, num_portfolios: int = 50) -> Tuple[List[float], List[float]]:
        """Génère la courbe de la frontière efficiente"""
        try:
            price_data = EfficientFrontier.get_historical_data(symbols, start_date, end_date)
            if price_data.empty:
                return [], []
            returns = price_data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            num_assets = len(symbols)
            results = []
            for _ in range(num_portfolios * 10):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                portfolio_return, portfolio_volatility = EfficientFrontier.calculate_portfolio_performance(
                    weights, mean_returns.values, cov_matrix.values
                )
                results.append({
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe': (portfolio_return - 0.02) / portfolio_volatility if portfolio_volatility > 0 else 0
                })
            results.sort(key=lambda x: x['sharpe'], reverse=True)
            results = results[:num_portfolios]
            returns_list = [r['return'] for r in results]
            volatility_list = [r['volatility'] for r in results]
            return returns_list, volatility_list
        except Exception as e:
            print(f"Erreur lors de la génération de la courbe: {e}")
            return [], []

class RiskPerformanceAnalyzer:
    """Analyseur avancé de risque et performance"""

    @staticmethod
    def get_beta(ticker: str, period: str = "2y") -> float:
        """Récupère le bêta d'une action calculé par rapport au marché (S&P 500)"""
        try:
            stock = yf.Ticker(ticker)
            market = yf.Ticker("^GSPC")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            stock_data = stock.history(start=start_date, end=end_date)
            market_data = market.history(start=start_date, end=end_date)
            if len(stock_data) < 50 or len(market_data) < 50:
                beta = stock.info.get('beta', 1.0)
                return beta if beta is not None else 1.0
            stock_returns = stock_data['Close'].pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna()
            common_dates = stock_returns.index.intersection(market_returns.index)
            stock_returns = stock_returns.loc[common_dates]
            market_returns = market_returns.loc[common_dates]
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            return beta
        except Exception as e:
            print(f"Erreur lors du calcul du bêta pour {ticker}: {e}")
            return 1.0

    @staticmethod
    def calculate_advanced_metrics(df: pd.DataFrame, period_days: int = 252) -> Dict:
        """Calcule les métriques avancées de risque et performance"""
        if 'perf' not in df.columns or 'weight' not in df.columns or len(df) == 0:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_drawdown': 0,
                'var_95': 0,
                'cvar_95': 0,
                'beta': 1.0,
                'alpha': 0,
                'information_ratio': 0,
                'treynor_ratio': 0,
                'portfolio_return': 0,
                'portfolio_volatility': 0
            }
        returns = df['perf'].values / 100
        weights = df['weight'].values / 100
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        portfolio_return = np.sum(weights * returns)
        individual_volatilities = np.abs(returns - np.mean(returns))
        portfolio_volatility = np.sqrt(np.sum((weights**2) * (individual_volatilities**2)))
        annualized_return = portfolio_return * period_days
        annualized_volatility = portfolio_volatility * np.sqrt(period_days)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns) * np.sqrt(period_days)
        else:
            downside_deviation = annualized_volatility
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        sorted_returns = np.sort(returns)
        max_drawdown = abs(sorted_returns[0]) if len(sorted_returns) > 0 else 0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        var_95 = np.percentile(returns, 5)
        returns_below_var = returns[returns <= var_95]
        cvar_95 = np.mean(returns_below_var) if len(returns_below_var) > 0 else var_95
        portfolio_beta = 1.0
        if 'symbol' in df.columns:
            try:
                betas = []
                for idx, row in df.iterrows():
                    if pd.notna(row['symbol']) and row['symbol'].strip():
                        beta = RiskPerformanceAnalyzer.get_beta(row['symbol'])
                        betas.append(beta * weights[idx])
                if betas:
                    portfolio_beta = np.sum(betas)
            except:
                portfolio_beta = 1.0
        market_return = 0.08
        alpha = annualized_return - (risk_free_rate + portfolio_beta * (market_return - risk_free_rate))
        tracking_error = annualized_volatility * 0.8
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        treynor_ratio = (annualized_return - risk_free_rate) / portfolio_beta if portfolio_beta > 0 else 0
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'beta': portfolio_beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'portfolio_return': annualized_return,
            'portfolio_volatility': annualized_volatility
        }

    @staticmethod
    def get_performance_grade(sharpe_ratio: float, sortino_ratio: float) -> str:
        """Détermine une note de performance basée sur les ratios"""
        if sharpe_ratio >= 2.0 and sortino_ratio >= 2.5:
            return "A+ (Excellent)"
        elif sharpe_ratio >= 1.5 and sortino_ratio >= 2.0:
            return "A (Très Bon)"
        elif sharpe_ratio >= 1.0 and sortino_ratio >= 1.5:
            return "B+ (Bon)"
        elif sharpe_ratio >= 0.5 and sortino_ratio >= 1.0:
            return "B (Acceptable)"
        elif sharpe_ratio >= 0.0 and sortino_ratio >= 0.5:
            return "C (Médiocre)"
        else:
            return "D (Insuffisant)"

def create_advanced_risk_analysis(df: pd.DataFrame, ticker_data: Optional[List[Dict]] = None):
    """Analyse de risque avancée avec frontière efficiente corrigée"""
    if not isinstance(df, pd.DataFrame):
        st.error("L'argument df doit être un DataFrame pandas.")
        return
    if ticker_data and len(ticker_data) > 0:
        if 'symbol' not in df.columns:
            df['symbol'] = ''
        for i, ticker in enumerate(ticker_data):
            if i < len(df):
                df.loc[i, 'symbol'] = ticker.get('symbol', '')
    st.subheader("⚠️ Analyse de Risque Avancée")
    required_columns = ['perf', 'weight']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Les colonnes suivantes sont manquantes : {missing_columns}")
        return
    if len(df) == 0:
        st.info("Aucune donnée disponible pour l'analyse")
        return
    try:
        metrics = RiskPerformanceAnalyzer.calculate_advanced_metrics(df)
        st.markdown("#### 📊 Métriques de Performance")
        grade = RiskPerformanceAnalyzer.get_performance_grade(
            metrics['sharpe_ratio'],
            metrics['sortino_ratio']
        )
        st.info(f"**Note de Performance:** {grade}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rendement Annuel", f"{metrics['portfolio_return']:.2%}")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
            st.metric("Alpha", f"{metrics['alpha']:.2%}")
        with col2:
            st.metric("Volatilité Annuelle", f"{metrics['portfolio_volatility']:.2%}")
            st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
            st.metric("Beta", f"{metrics['beta']:.3f}")
        with col3:
            st.metric("VaR 95%", f"{metrics['var_95']:.2%}")
            st.metric("CVaR 95%", f"{metrics['cvar_95']:.2%}")
            st.metric("Treynor Ratio", f"{metrics['treynor_ratio']:.3f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}")
            st.metric("Information Ratio", f"{metrics['information_ratio']:.3f}")
        st.markdown("#### 📈 Profil de Risque")
        metrics_normalized = {
            'Sharpe Ratio': max(0, min(metrics['sharpe_ratio'] / 3, 1)),
            'Sortino Ratio': max(0, min(metrics['sortino_ratio'] / 3, 1)),
            'Calmar Ratio': max(0, min(metrics['calmar_ratio'] / 2, 1)),
            'Information Ratio': max(0, min((metrics['information_ratio'] + 1) / 2, 1)),
            'Treynor Ratio': max(0, min(metrics['treynor_ratio'] / 0.1, 1)),
            'Alpha': max(0, min((metrics['alpha'] + 0.05) / 0.1, 1))
        }
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(metrics_normalized.values()),
            theta=list(metrics_normalized.keys()),
            fill='toself',
            name='Profil de Risque'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Profil de Risque du Portefeuille"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("#### 📈 Optimisation de Portefeuille")
        if 'symbol' in df.columns and len(df) >= 2:
            valid_symbols = []
            for symbol in df['symbol'].dropna():
                if symbol and isinstance(symbol, str) and symbol.strip():
                    valid_symbols.append(symbol.strip())
            valid_symbols = list(set(valid_symbols))
            if len(valid_symbols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    periods = {
                        "6 mois": 180,
                        "1 an": 365,
                        "2 ans": 730,
                        "3 ans": 1095
                    }
                    selected_period = st.selectbox("Période d'analyse", list(periods.keys()), index=1)
                    days_back = periods[selected_period]
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_back)
                with col2:
                    max_tickers = min(10, len(valid_symbols))
                    selected_tickers = st.multiselect(
                        "Sélectionner les actifs (max 10)",
                        valid_symbols,
                        default=valid_symbols[:max_tickers],
                        max_selections=10
                    )
                if st.button("🔄 Optimiser le portefeuille", key="optimize_portfolio"):
                    if len(selected_tickers) >= 2:
                        with st.spinner("Calcul de l'optimisation..."):
                            optimal_weights_df, metrics_ef = EfficientFrontier.get_efficient_frontier(
                                selected_tickers,
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d')
                            )
                            if not optimal_weights_df.empty and 'error' not in metrics_ef:
                                st.success("✅ Optimisation réussie!")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Rendement Optimal", f"{metrics_ef['expected_return']:.2%}")
                                with col2:
                                    st.metric("Volatilité Optimale", f"{metrics_ef['volatility']:.2%}")
                                with col3:
                                    st.metric("Sharpe Ratio Optimal", f"{metrics_ef['sharpe_ratio']:.3f}")
                                st.subheader("🎯 Allocation Optimale")
                                comparison_data = []
                                for symbol in optimal_weights_df.index:
                                    current_weight = 0
                                    if symbol in df['symbol'].values:
                                        mask = df['symbol'] == symbol
                                        if mask.any():
                                            current_weight = df.loc[mask, 'weight'].iloc[0] / 100
                                    optimal_weight = optimal_weights_df.loc[symbol, 'weight']
                                    comparison_data.append({
                                        'Actif': symbol,
                                        'Poids Actuel (%)': current_weight * 100,
                                        'Poids Optimal (%)': optimal_weight * 100,
                                        'Différence (%)': (optimal_weight - current_weight) * 100
                                    })
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(
                                    comparison_df.style.format({
                                        'Poids Actuel (%)': '{:.2f}%',
                                        'Poids Optimal (%)': '{:.2f}%',
                                        'Différence (%)': '{:+.2f}%'
                                    }).background_gradient(subset=['Différence (%)'], cmap='RdYlGn'),
                                    use_container_width=True
                                )
                                fig_comparison = go.Figure()
                                fig_comparison.add_trace(go.Bar(
                                    name='Poids Actuel',
                                    x=comparison_df['Actif'],
                                    y=comparison_df['Poids Actuel (%)'],
                                    marker_color='lightcoral'
                                ))
                                fig_comparison.add_trace(go.Bar(
                                    name='Poids Optimal',
                                    x=comparison_df['Actif'],
                                    y=comparison_df['Poids Optimal (%)'],
                                    marker_color='lightblue'
                                ))
                                fig_comparison.update_layout(
                                    title='Comparaison Allocation Actuelle vs Optimale',
                                    xaxis_title='Actifs',
                                    yaxis_title='Poids (%)',
                                    barmode='group',
                                    height=400
                                )
                                st.plotly_chart(fig_comparison, use_container_width=True)
                            else:
                                error_msg = metrics_ef.get('error', 'Erreur inconnue')
                                st.error(f"❌ Erreur lors de l'optimisation: {error_msg}")
                    else:
                        st.warning("⚠️ Sélectionnez au moins 2 actifs")
            else:
                st.warning(f"⚠️ Au moins 2 symboles valides requis. Trouvés: {len(valid_symbols)}")
        else:
            st.error("❌ Colonne 'symbol' manquante ou moins de 2 actifs")
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        st.write("Débogage:", str(e))

def export_portfolio_report(df: pd.DataFrame):
    """Permet d'exporter un rapport du portefeuille"""
    st.subheader("📤 Export du rapport")
    if st.button("📊 Générer rapport CSV"):
        export_columns = ['name', 'symbol', 'quantity', 'buyingPrice', 'lastPrice',
                          'amount', 'weight_pct', 'perf', 'sector', 'asset_type']
        available_columns = [col for col in export_columns if col in df.columns]
        export_df = df[available_columns].copy()
        column_rename = {
            'name': 'Nom',
            'symbol': 'Symbole',
            'quantity': 'Quantité',
            'buyingPrice': 'Prix_Achat',
            'lastPrice': 'Prix_Actuel',
            'amount': 'Montant',
            'weight_pct': 'Poids_Pct',
            'perf': 'Performance_Pct',
            'sector': 'Secteur',
            'asset_type': 'Type_Actif'
        }
        export_df = export_df.rename(columns={k: v for k, v in column_rename.items() if k in export_df.columns})
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="💾 Télécharger CSV",
            data=csv,
            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
        st.success("✅ Rapport CSV généré avec succès!")
    if st.button("📋 Générer rapport JSON"):
        report_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_positions': len(df),
                'total_value': df['amount'].sum() if 'amount' in df.columns else 0,
                'portfolio_performance': (df['weight'] * df['perf']).sum() if all(col in df.columns for col in ['weight', 'perf']) else 0
            },
            'positions': df.to_dict('records')
        }
        json_str = json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="💾 Télécharger JSON",
            data=json_str,
            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime='application/json'
        )
        st.success("✅ Rapport JSON généré avec succès!")
    with st.expander("👀 Aperçu des données d'export"):
        if not df.empty:
            st.dataframe(df.head(10))
        else:
            st.info("Aucune donnée de portefeuille disponible")

def main():
    """Fonction principale de l'application Streamlit"""
    st.title("📊 Portfolio Analyzer Pro")
    st.markdown("### Analysez et optimisez votre portefeuille d'investissement")
    portfolio_manager = PortfolioManager()
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("📁 Import/Export")
        uploaded_file = st.file_uploader(
            "Importer un portefeuille",
            type=['csv', 'xlsx', 'json'],
            help="Formats supportés: CSV, Excel, JSON"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_imported = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df_imported = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    json_data = json.load(uploaded_file)
                    if 'positions' in json_data:
                        df_imported = pd.DataFrame(json_data['positions'])
                    else:
                        df_imported = pd.DataFrame(json_data)
                df_enhanced = enhance_dataframe(df_imported)
                st.session_state.portfolio_df = df_enhanced
                st.session_state.original_df = df_imported.copy()
                st.success(f"✅ Fichier importé: {len(df_enhanced)} positions")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'import: {str(e)}")
        st.subheader("➕ Ajouter une action")
        search_query = st.text_input("Rechercher un ticker ou nom d'entreprise")
        if search_query:
            with st.spinner("Recherche en cours..."):
                search_results = TickerService.search_tickers(search_query, limit=5)
            if search_results:
                ticker_options = [f"{result['symbol']} - {result['name']}" for result in search_results]
                selected_ticker_idx = st.selectbox(
                    "Sélectionner un ticker",
                    range(len(ticker_options)),
                    format_func=lambda x: ticker_options[x]
                )
                selected_ticker = search_results[selected_ticker_idx]
                with st.spinner("Validation du ticker..."):
                    ticker_data = TickerService.validate_ticker(selected_ticker['symbol'])
                if ticker_data['valid']:
                    st.info(f"**{ticker_data['name']}**\nPrix actuel: {ticker_data['price']:.2f} {ticker_data['currency']}")
                    quantity = st.number_input("Quantité", min_value=1, value=1)
                    purchase_date = st.date_input("Date d'achat", datetime.now().strftime("%Y-%m-%d"))
                    st.write(f"**Date d'achat:** {purchase_date}")
                    st.markdown("**Prix d'achat:**")
                    price_option = st.radio(
                        "Choisir le prix d'achat",
                        ["Prix actuel", "Prix personnalisé"],
                        key="price_option"
                    )
                    buying_price = None
                    if price_option == "Prix actuel":
                        buying_price = ticker_data['price']
                        st.success(f"✅ Prix d'achat: {buying_price:.2f} {ticker_data['currency']} (prix actuel)")
                    else:
                        buying_price = st.number_input(
                            f"Prix d'achat personnalisé ({ticker_data['currency']})",
                            min_value=0.01,
                            value=ticker_data['price'],
                            step=0.01,
                            format="%.2f"
                        )
                        if buying_price != ticker_data['price']:
                            pnl_per_share = ticker_data['price'] - buying_price
                            pnl_total = pnl_per_share * quantity
                            pnl_percent = (pnl_per_share / buying_price * 100) if buying_price > 0 else 0
                            if pnl_per_share > 0:
                                st.success(f"📈 Plus-value: +{pnl_total:.2f} {ticker_data['currency']} ({pnl_percent:+.2f}%)")
                            elif pnl_per_share < 0:
                                st.error(f"📉 Moins-value: {pnl_total:.2f} {ticker_data['currency']} ({pnl_percent:+.2f}%)")
                            else:
                                st.info("➡️ Aucune plus/moins-value")
                    with st.expander("📋 Résumé de l'ajout"):
                        total_cost = buying_price * quantity
                        current_value = ticker_data['price'] * quantity
                        st.write(f"**Quantité:** {quantity}")
                        st.write(f"**Prix d'achat unitaire:** {buying_price:.2f} {ticker_data['currency']}")
                        st.write(f"**Prix actuel unitaire:** {ticker_data['price']:.2f} {ticker_data['currency']}")
                        st.write(f"**Coût total d'achat:** {total_cost:.2f} {ticker_data['currency']}")
                        st.write(f"**Valeur actuelle:** {current_value:.2f} {ticker_data['currency']}")
                        pnl = current_value - total_cost
                        if pnl != 0:
                            pnl_color = "green" if pnl > 0 else "red"
                            st.markdown(f"**Plus/Moins-value:** <span style='color: {pnl_color}'>{pnl:+.2f} {ticker_data['currency']}</span>", unsafe_allow_html=True)
                    if st.button("Ajouter au portefeuille"):
                        success = portfolio_manager.add_stock_to_portfolio(ticker_data, quantity, buying_price, purchase_date)
                        if success:
                            st.success("✅ Action ajoutée au portefeuille!")
                            st.rerun()
                        else:
                            st.error("❌ Erreur lors de l'ajout")
                else:
                    st.error(f"❌ Ticker invalide: {ticker_data.get('error', 'Erreur inconnue')}")
            else:
                st.info("Aucun résultat trouvé")
    if not st.session_state.portfolio_df.empty:
        df = st.session_state.portfolio_df
        metrics = portfolio_manager.update_portfolio_metrics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Valeur totale", f"{metrics['total_value']:,.2f} €")
        with col2:
            st.metric("Nombre de positions", len(df))
        with col3:
            st.metric("Performance globale", f"{metrics['portfolio_performance']:.2f}%")
        with col4:
            avg_weight = df['weight_pct'].mean() if 'weight_pct' in df.columns else 0
            st.metric("Poids moyen", f"{avg_weight:.1f}%")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Vue d'ensemble",
            "📈 Diversification",
            "⚠️ Analyse de risque",
            "🎯 Recommandations",
            "📤 Export"
        ])
        with tab1:
            display_portfolio_summary(df)
            col1, col2 = st.columns(2)
            with col1:
                if 'weight_pct' in df.columns:
                    fig_pie = px.pie(df.head(10), values='weight_pct', names='name',
                                   title="Répartition par position (Top 10)")
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                if 'asset_type' in df.columns and 'weight_pct' in df.columns:
                    asset_dist = df.groupby('asset_type')['weight_pct'].sum().reset_index()
                    fig_asset = px.bar(asset_dist, x='asset_type', y='weight_pct',
                                     title="Répartition par type d'actif")
                    fig_asset.update_layout(height=400)
                    st.plotly_chart(fig_asset, use_container_width=True)
        with tab2:
            st.subheader("🎯 Analyse de diversification")
            concentration_metrics = DiversificationAnalyzer.calculate_concentration_metrics(df)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Indice HHI", f"{concentration_metrics['hhi']:.3f}")
            with col2:
                st.metric("Actions effectives", f"{concentration_metrics['effective_stocks']:.1f}")
            with col3:
                st.metric("Top 3 concentration", f"{concentration_metrics['top3_concentration']:.1%}")
            with col4:
                st.metric("Niveau", concentration_metrics['concentration_level'])
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏭 Diversification sectorielle")
                sector_analysis = DiversificationAnalyzer.analyze_sector_diversification(df)
                if not sector_analysis.empty:
                    st.dataframe(sector_analysis.style.format({
                        'Weight_Pct': '{:.1f}%',
                        'Avg_Performance': '{:.2f}%'
                    }))
                    fig_sector = px.bar(sector_analysis.head(8), x=sector_analysis.head(8).index,
                                      y='Weight_Pct', title="Exposition sectorielle (%)")
                    fig_sector.update_layout(height=300, xaxis_title="Secteur", yaxis_title="Poids (%)")
                    st.plotly_chart(fig_sector, use_container_width=True)
                else:
                    st.info("Données sectorielles non disponibles")
            with col2:
                st.subheader("🌍 Diversification géographique")
                geo_analysis = DiversificationAnalyzer.analyze_geographic_diversification(df)
                if not geo_analysis.empty:
                    st.dataframe(geo_analysis.style.format({
                        'Weight_Pct': '{:.1f}%',
                        'Avg_Performance': '{:.2f}%'
                    }))
                    fig_geo = px.pie(geo_analysis, values='Weight_Pct', names=geo_analysis.index,
                                   title="Répartition géographique")
                    fig_geo.update_layout(height=300)
                    st.plotly_chart(fig_geo, use_container_width=True)
                else:
                    st.info("Données géographiques non disponibles")
        with tab3:
            create_advanced_risk_analysis(df)
        with tab4:
            st.subheader("🎯 Recommandations personnalisées")
            concentration_metrics = DiversificationAnalyzer.calculate_concentration_metrics(df)
            sector_analysis = DiversificationAnalyzer.analyze_sector_diversification(df)
            geo_analysis = DiversificationAnalyzer.analyze_geographic_diversification(df)
            generate_recommendations(df, concentration_metrics, sector_analysis, geo_analysis)
        with tab5:
            export_portfolio_report(df)
        st.subheader("📋 Détail du portefeuille")
        display_columns = ['name', 'symbol', 'quantity', "purchase_date", 'buyingPrice', 'lastPrice',
                          'amount', 'weight_pct', 'perf', 'sector']
        available_display_columns = [col for col in display_columns if col in df.columns]
        if available_display_columns:
            df_display = df[available_display_columns].copy()
            column_names = {
                'name': 'Nom',
                'symbol': 'Symbole',
                'quantity': 'Quantité',
                "purchase_date": 'Date',
                'buyingPrice': "Prix d'achat",
                'lastPrice': 'Prix actuel',
                'amount': 'Montant (€)',
                'weight_pct': 'Poids (%)',
                'perf': 'Performance (%)',
                'sector': 'Secteur'
            }
            df_display = df_display.rename(columns={k: v for k, v in column_names.items() if k in df_display.columns})
            def color_performance(val):
                try:
                    if val > 0:
                        return 'color: green'
                    elif val < 0:
                        return 'color: red'
                    else:
                        return 'color: black'
                except:
                    return 'color: black'
            format_dict = {}
            if 'Prix d\'achat' in df_display.columns:
                format_dict['Prix d\'achat'] = '{:.2f}'
            if 'Prix actuel' in df_display.columns:
                format_dict['Prix actuel'] = '{:.2f}'
            if 'Montant (€)' in df_display.columns:
                format_dict['Montant (€)'] = '{:,.2f}'
            if 'Poids (%)' in df_display.columns:
                format_dict['Poids (%)'] = '{:.1f}'
            if 'Performance (%)' in df_display.columns:
                format_dict['Performance (%)'] = '{:.2f}'
            styled_df = df_display.style.format(format_dict)
            if 'Performance (%)' in df_display.columns:
                styled_df = styled_df.applymap(color_performance, subset=['Performance (%)'])
            st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            st.dataframe(df, use_container_width=True, height=400)
        st.subheader("🗑️ Gestion des positions")
        if len(df) > 0:
            position_to_delete = st.selectbox(
                "Sélectionner une position à supprimer",
                range(len(df)),
                format_func=lambda x: f"{df.iloc[x]['name']} ({df.iloc[x].get('symbol', 'N/A')})"
            )
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Supprimer", type="secondary"):
                    st.session_state.portfolio_df = st.session_state.portfolio_df.drop(
                        st.session_state.portfolio_df.index[position_to_delete]
                    ).reset_index(drop=True)
                    st.success("Position supprimée!")
                    st.rerun()
            with col2:
                if st.button("🔄 Actualiser les prix", type="primary"):
                    with st.spinner("Actualisation des prix en cours..."):
                        updated_count = 0
                        for idx, row in st.session_state.portfolio_df.iterrows():
                            if 'symbol' in row and row['symbol']:
                                try:
                                    ticker_data = TickerService.validate_ticker(row['symbol'])
                                    if ticker_data['valid']:
                                        st.session_state.portfolio_df.at[idx, 'lastPrice'] = ticker_data['price']
                                        updated_count += 1
                                except:
                                    continue
                        if updated_count > 0:
                            portfolio_manager.update_portfolio_metrics()
                            st.success(f"✅ {updated_count} prix mis à jour!")
                            st.rerun()
                        else:
                            st.warning("Aucun prix n'a pu être mis à jour")
    else:
        st.info("🚀 Commencez par importer un portefeuille ou ajouter des actions via la barre latérale.")
        with st.expander("📄 Format de fichier d'import"):
            st.markdown("""
            **Colonnes requises/recommandées pour l'import CSV/Excel:**
            - `name` ou `nom`: Nom de l'entreprise/action
            - `symbol` ou `ticker`: Symbole boursier (ex: AAPL, MC.PA)
            - `quantity` ou `quantite`: Nombre d'actions détenues
            - `buyingPrice` ou `prix_achat`: Prix d'achat unitaire
            - `lastPrice` ou `prix_actuel`: Prix actuel (optionnel, sera actualisé automatiquement)
            - `isin`: Code ISIN (optionnel)
            **Exemple:**
            ```
            name,symbol,quantity,buyingPrice
            Apple Inc,AAPL,10,150.00
            LVMH,MC.PA,5,650.00
            Microsoft,MSFT,8,280.00
            ```
            """)

if __name__ == "__main__":
    main()

