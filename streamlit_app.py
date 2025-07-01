import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib as plt
[theme]
base="dark"
primaryColor="deepblue"

st.title("Upload your CSV File")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(df)

#On calcule la val tot
total_value = df['amount'].sum()
st.write(f"Valeur totale portefeuille : {total_value:.2f} EUR")

#On calcule la perf
df['weight'] = df['amount'] / total_value
df['perf'] = (df['lastPrice'] - df['buyingPrice']) / df['buyingPrice'] * 100
portfolio_perf = (df['weight'] * df['perf']).sum()
print(f"Performance globale portefeuille : {portfolio_perf:.2f} %")

# On calcule la répartition
df['weight_pct'] = df['weight'] * 100

# on crée un camembert
plt.figure(figsize=(7,7))
plt.pie(df['weight_pct'], labels=df['name'], autopct='%1.1f%%')
plt.title("Répartition du portefeuille")
plt.show()

# barre avec les perfs de chaque actif

colors = np.where(df['perf'] >= 0, 'green', 'red')

plt.figure(figsize=(10,5))
plt.bar(df['name'], df['perf'], color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Performance en %")
plt.title("Performance par actif")
plt.show()

import pandas as pd
import requests
import json
import time
from typing import List, Dict, Optional

def get_ticker_from_isin(isin: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Récupère le ticker d'un ISIN via l'API OpenFIGI

    Args:
        isin: Code ISIN
        api_key: Clé API OpenFIGI (optionnelle pour les requêtes limitées)

    Returns:
        Ticker ou None si non trouvé
    """
    url = "https://api.openfigi.com/v3/mapping"

    headers = {
        'Content-Type': 'application/json'
    }

    # Ajouter la clé API si fournie
    if api_key:
        headers['X-OPENFIGI-APIKEY'] = api_key

    # Payload pour la recherche
    payload = [{
        "idType": "ID_ISIN",
        "idValue": isin,
        "exchCode": "US"  # Vous pouvez ajuster selon vos besoins
    }]

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        # Vérifier si des résultats sont retournés
        if data and len(data) > 0 and 'data' in data[0]:
            results = data[0]['data']
            if results:
                # Prendre le premier résultat avec un ticker
                for result in results:
                    if 'ticker' in result and result['ticker']:
                        return result['ticker']

        return None

    except requests.exceptions.RequestException as e:
        print(f"Erreur API pour ISIN {isin}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON pour ISIN {isin}: {e}")
        return None

def get_tickers_batch(isins: List[str], api_key: Optional[str] = None, batch_size: int = 10) -> Dict[str, str]:
    """
    Récupère les tickers pour une liste d'ISIN par lots

    Args:
        isins: Liste des codes ISIN
        api_key: Clé API OpenFIGI
        batch_size: Taille des lots pour les requêtes

    Returns:
        Dictionnaire {ISIN: ticker}
    """
    url = "https://api.openfigi.com/v3/mapping"

    headers = {
        'Content-Type': 'application/json'
    }

    if api_key:
        headers['X-OPENFIGI-APIKEY'] = api_key

    results = {}

    # Traitement par lots
    for i in range(0, len(isins), batch_size):
        batch = isins[i:i + batch_size]

        # Construire le payload pour le lot
        payload = []
        for isin in batch:
            payload.append({
                "idType": "ID_ISIN",
                "idValue": isin
            })

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()

            # Traiter les résultats
            for j, isin in enumerate(batch):
                ticker = None
                if j < len(data) and 'data' in data[j] and data[j]['data']:
                    # Chercher le premier ticker disponible
                    for result in data[j]['data']:
                        if 'ticker' in result and result['ticker']:
                            ticker = result['ticker']
                            break

                results[isin] = ticker

            # Respecter les limites de taux (pour API gratuite)
            if not api_key:
                time.sleep(1)  # Pause d'1 seconde entre les lots

        except requests.exceptions.RequestException as e:
            print(f"Erreur API pour le lot {i//batch_size + 1}: {e}")
            # Marquer tous les ISIN du lot comme non trouvés
            for isin in batch:
                results[isin] = None
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON pour le lot {i//batch_size + 1}: {e}")
            for isin in batch:
                results[isin] = None

    return results

def extract_tickers_from_dataframe(df: pd.DataFrame, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Extrait les tickers pour tous les ISIN d'un DataFrame

    Args:
        df: DataFrame contenant une colonne 'isin'
        api_key: Clé API OpenFIGI (optionnelle)

    Returns:
        DataFrame avec une nouvelle colonne 'Tickers'
    """
    # Vérifier que la colonne 'isin' existe
    if 'isin' not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'isin'")

    # Récupérer la liste unique des ISIN
    unique_isins = df['isin'].dropna().unique().tolist()

    print(f"Extraction des tickers pour {len(unique_isins)} ISIN uniques...")

    # Récupérer les tickers par lots
    isin_to_ticker = get_tickers_batch(unique_isins, api_key)

    # Mapper les résultats au DataFrame
    df_copy = df.copy()
    df_copy['Tickers'] = df_copy['isin'].map(isin_to_ticker)

    # Statistiques
    found_count = df_copy['Tickers'].notna().sum()
    total_count = len(df_copy)

    print(f"Tickers trouvés: {found_count}/{total_count} ({found_count/total_count*100:.1f}%)")

    return df_copy

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple avec un DataFrame de test

    print("DataFrame original:")
    print(df)
    print()

    # Extraire les tickers (sans clé API pour cet exemple)
    # Pour utiliser avec une clé API: df_with_tickers = extract_tickers_from_dataframe(df, api_key="VOTRE_CLE_API")
    df = extract_tickers_from_dataframe(df)

    print("DataFrame avec tickers:")
    print(df)

import pandas as pd
import numpy as np

# Valeur totale du portefeuille
total_value = df['amount'].sum()
print(f"Valeur totale portefeuille : {total_value:.2f} EUR")

# Performance pondérée
df['weight'] = df['amount'] / total_value
df['perf'] = (df['lastPrice'] - df['buyingPrice']) / df['buyingPrice'] * 100
portfolio_perf = (df['weight'] * df['perf']).sum()
print(f"Performance globale portefeuille : {portfolio_perf:.2f} %")

# Répartition en %
df['weight_pct'] = df['weight'] * 100

# Graphique camembert
plt.figure(figsize=(7,7))
plt.pie(df['weight_pct'], labels=df['name'], autopct='%1.1f%%')
plt.title("Répartition du portefeuille")
plt.show()

# Graphique barres performance


colors = np.where(df['perf'] >= 0, 'green', 'red')

plt.figure(figsize=(10,5))
plt.bar(df['name'], df['perf'], color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Performance en %")
plt.title("Performance par actif")
plt.show()

# ici, idéalement ticker boursier, sinon ISIN ou mapping
tickers=df['Tickers'].tolist()
# Exemple avec tickers boursiers (à adapter)
prices = yf.download(tickers, period="1y")['Close']
print(prices)
# Rendements journaliers
returns = prices.pct_change().dropna()

# Création du mapping name -> ticker (ex : 'APPLE INC' -> 'AAPL')
name_to_ticker = dict(zip(df['name'], df['Tickers']))

# Création de la Series des poids avec index = name
weights_named = df.set_index('name')['weight']

# Renommer les index avec les tickers pour qu’ils correspondent aux colonnes de `returns`
weights = weights_named.rename(index=name_to_ticker)

# Réordonner les poids dans le même ordre que returns.columns
weights_vector = weights.loc[returns.columns].values

# Matrice de covariance annualisée
cov_matrix = returns.cov() * 252

# Calcul de la volatilité annualisée du portefeuille
portfolio_vol = np.sqrt(weights_vector.T @ cov_matrix @ weights_vector)

# Affichage
print(f"Volatilité annualisée portefeuille : {portfolio_vol:.4%}")

#MDD
weighted_returns = (returns * weights).sum(axis=1)
cumulative = (1 + weighted_returns).cumprod()

rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max
max_drawdown = drawdown.min()
print(f"Max Drawdown : {max_drawdown:.2%}")
plt.figure(figsize=(10, 4))
drawdown.plot(title="Drawdown du portefeuille", color="red")
plt.axhline(max_drawdown, linestyle='--', color='black', label=f'Max DD: {max_drawdown:.2%}')
plt.legend()
plt.grid(True)
plt.show()

# Calcul des métriques (après votre code existant)
weighted_returns = returns @ weights_vector
risk_free_rate = 0.02
daily_risk_free = risk_free_rate / 252

# Métriques de base
annual_return = weighted_returns.mean() * 252
annual_volatility = weighted_returns.std() * np.sqrt(252)
excess_returns = weighted_returns - daily_risk_free

# Ratios
sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

downside_returns = weighted_returns[weighted_returns < daily_risk_free]
downside_deviation = downside_returns.std() * np.sqrt(252)
sortino_ratio = (annual_return - risk_free_rate) / downside_deviation

cumulative_returns = (1 + weighted_returns).cumprod()
rolling_max = cumulative_returns.expanding().max()
drawdowns = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = abs(drawdowns.min())
calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else np.inf

# Ratio d'information (vs benchmark - supposons un indice de référence)
# Si vous avez un benchmark, remplacez par les vraies données
benchmark_return = 0.08  # Exemple : 8% annuel pour un indice
information_ratio = (annual_return - benchmark_return) / annual_volatility

print("="*60)
print("ANALYSE COMPLÈTE DU PORTEFEUILLE")
print("="*60)

print(f"\n📊 MÉTRIQUES DE BASE:")
print(f"Rendement annualisé     : {annual_return:>8.2%}")
print(f"Volatilité annualisée   : {annual_volatility:>8.2%}")
print(f"Drawdown maximum        : {max_drawdown:>8.2%}")

print(f"\n📈 RATIOS DE PERFORMANCE:")
print(f"Ratio de Sharpe         : {sharpe_ratio:>8.3f}")
print(f"Ratio de Sortino        : {sortino_ratio:>8.3f}")
print(f"Ratio de Calmar         : {calmar_ratio:>8.3f}")
print(f"Ratio d'Information     : {information_ratio:>8.3f}")

print(f"\n" + "="*60)
print("INTERPRÉTATION DES RÉSULTATS")
print("="*60)

# Analyse du ratio de Sharpe
print(f"\n🎯 RATIO DE SHARPE ({sharpe_ratio:.3f}):")
if sharpe_ratio > 2:
    sharpe_eval = "EXCELLENT - Performance exceptionnelle ajustée du risque"
elif sharpe_ratio > 1:
    sharpe_eval = "BON - Bonne compensation du risque pris"
elif sharpe_ratio > 0.5:
    sharpe_eval = "ACCEPTABLE - Compensation modérée du risque"
elif sharpe_ratio > 0:
    sharpe_eval = "FAIBLE - Peu de compensation pour le risque pris"
else:
    sharpe_eval = "NÉGATIF - Performance inférieure au taux sans risque"

print(f"   → {sharpe_eval}")
print(f"   → Pour chaque unité de risque, vous gagnez {sharpe_ratio:.3f} unités de rendement excédentaire")

# Analyse du ratio de Sortino
print(f"\n📉 RATIO DE SORTINO ({sortino_ratio:.3f}):")
if sortino_ratio > sharpe_ratio:
    sortino_eval = "POSITIF - Vos pertes sont moins fréquentes que la volatilité globale"
else:
    sortino_eval = "ATTENTION - Volatilité importante à la baisse"

print(f"   → {sortino_eval}")
print(f"   → Ratio {sortino_ratio/sharpe_ratio:.1f}x supérieur au Sharpe = {'faible asymétrie négative' if sortino_ratio/sharpe_ratio < 1.5 else 'forte asymétrie positive'}")

# Analyse du ratio de Calmar
print(f"\n⬇️ RATIO DE CALMAR ({calmar_ratio:.3f}):")
if calmar_ratio > 1:
    calmar_eval = "EXCELLENT - Rendement supérieur au pire drawdown"
elif calmar_ratio > 0.5:
    calmar_eval = "BON - Rendement décent par rapport aux pertes maximales"
elif calmar_ratio > 0.2:
    calmar_eval = "ACCEPTABLE - Attention aux périodes de pertes"
else:
    calmar_eval = "FAIBLE - Drawdowns importants par rapport au rendement"

print(f"   → {calmar_eval}")
print(f"   → Votre pire période a généré {max_drawdown:.1%} de perte")
print(f"   → Il faudrait {max_drawdown/annual_return:.1f} années au rendement actuel pour compenser")

# Analyse du ratio d'information
print(f"\n📊 RATIO D'INFORMATION ({information_ratio:.3f}):")
if information_ratio > 0.5:
    info_eval = "EXCELLENT - Surperformance significative vs benchmark"
elif information_ratio > 0:
    info_eval = "POSITIF - Légère surperformance du marché"
elif information_ratio > -0.5:
    info_eval = "NEUTRE - Performance proche du marché"
else:
    info_eval = "NÉGATIF - Sous-performance du marché"

print(f"   → {info_eval}")
print(f"   → Alpha généré : {(annual_return - benchmark_return)*100:.1f} points de base")

# Recommandations
print(f"\n" + "="*60)
print("🎯 RECOMMANDATIONS STRATÉGIQUES")
print("="*60)

if sharpe_ratio < 0.5:
    print("⚠️  RISQUE ÉLEVÉ:")
    print("   • Considérez réduire l'exposition aux actifs les plus volatiles")
    print("   • Augmentez la diversification sectorielle/géographique")

if max_drawdown > 0.2:
    print("⚠️  DRAWDOWN IMPORTANT:")
    print("   • Implémentez une stratégie de stop-loss")
    print("   • Considérez un rebalancement plus fréquent")

if sortino_ratio / sharpe_ratio < 1.2:
    print("⚠️  ASYMÉTRIE NÉGATIVE:")
    print("   • Vos pertes sont proportionnellement importantes")
    print("   • Envisagez des stratégies de protection (puts, VIX)")

if sharpe_ratio > 1 and calmar_ratio > 0.5:
    print("✅ PORTEFEUILLE ÉQUILIBRÉ:")
    print("   • Bonne gestion risque/rendement")
    print("   • Maintenez votre stratégie actuelle")

# Benchmark de l'industrie
print(f"\n📈 COMPARAISON MARCHÉ:")
print(f"   • Fonds indiciels      : Sharpe ~0.3-0.6")
print(f"   • Gestion active       : Sharpe ~0.4-0.8")
print(f"   • Hedge funds          : Sharpe ~0.6-1.2")
print(f"   • Votre portefeuille   : Sharpe {sharpe_ratio:.3f}")

benchmark_category = "Sous-performant" if sharpe_ratio < 0.3 else \
                    "Marché passif" if sharpe_ratio < 0.6 else \
                    "Gestion active" if sharpe_ratio < 1.2 else "Elite"
print(f"   → Classification : {benchmark_category}")

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re

# 1. MAPPING COMPLET DES SUFFIXES YAHOO FINANCE → INDICES
def get_comprehensive_index_mapping(tickers):
    """
    Mapping complet basé sur les suffixes officiels Yahoo Finance
    Source: https://help.yahoo.com/kb/SLN2310.html
    """

    # Dictionnaire complet des suffixes Yahoo Finance → Indice de référence
    YAHOO_SUFFIX_MAPPING = {
        # EUROPE
        '.PA': 'CAC',           # Euronext Paris (France)
        '.NX': 'CAC',           # Euronext (général)
        '.AS': 'AEX',           # Euronext Amsterdam (Pays-Bas)
        '.BR': 'BEL20',         # Euronext Brussels (Belgique)
        '.LS': 'PSI20',         # Euronext Lisbon (Portugal)

        '.L': 'FTSE',           # London Stock Exchange (UK)
        '.IL': 'FTSE',          # London IOB

        '.DE': 'DAX',           # XETRA (Allemagne)
        '.F': 'DAX',            # Frankfurt Stock Exchange
        '.DU': 'DAX',           # Düsseldorf
        '.HM': 'DAX',           # Hamburg
        '.HA': 'DAX',           # Hanovre
        '.MU': 'DAX',           # Munich
        '.SG': 'DAX',           # Stuttgart
        '.BE': 'DAX',           # Berlin

        '.MI': 'FTSE_MIB',      # Borsa Italiana (Italie)

        '.MC': 'IBEX',          # Madrid Stock Exchange (Espagne)
        '.BC': 'IBEX',          # Barcelona

        '.SW': 'SMI',           # SIX Swiss Exchange (Suisse)

        '.VI': 'ATX',           # Vienna Stock Exchange (Autriche)

        '.OL': 'OSE',           # Oslo Stock Exchange (Norvège)
        '.ST': 'OMX',           # Stockholm Stock Exchange (Suède)
        '.CO': 'OMX',           # Copenhagen Stock Exchange (Danemark)
        '.HE': 'OMX',           # Helsinki Stock Exchange (Finlande)
        '.IC': 'OMX',           # Iceland Stock Exchange

        '.PR': 'WIG',           # Warsaw Stock Exchange (Pologne)
        '.BD': 'SOFIX',         # Bulgarian Stock Exchange
        '.RG': 'BET',           # Bucharest Stock Exchange (Roumanie)

        # AMÉRIQUE DU NORD
        # USA - Pas de suffixe = NASDAQ par défaut pour les actions tech, NYSE pour les autres
        '.OQ': 'NASDAQ',        # NASDAQ
        '.N': 'NYSE',           # New York Stock Exchange
        '.A': 'NYSE_ARCA',      # NYSE Arca
        '.P': 'NYSE_ARCA',      # NYSE Arca (options)

        '.TO': 'TSX',           # Toronto Stock Exchange (Canada)
        '.V': 'TSX_V',          # TSX Venture Exchange

        '.MX': 'IPC',           # Mexican Stock Exchange

        # ASIE-PACIFIQUE
        '.T': 'NIKKEI',         # Tokyo Stock Exchange (Japon)
        '.SS': 'SSE',           # Shanghai Stock Exchange (Chine)
        '.SZ': 'SZSE',          # Shenzhen Stock Exchange (Chine)
        '.HK': 'HSI',           # Hong Kong Stock Exchange
        '.KS': 'KOSPI',         # Korea Stock Exchange
        '.KQ': 'KOSDAQ',        # KOSDAQ (Corée du Sud)
        '.TW': 'TAIEX',         # Taiwan Stock Exchange
        '.SI': 'STI',           # Singapore Stock Exchange
        '.AX': 'ASX',           # Australian Securities Exchange
        '.NZ': 'NZX',           # New Zealand Stock Exchange

        '.BO': 'BSE',           # Bombay Stock Exchange (Inde)
        '.NS': 'NSE',           # National Stock Exchange (Inde)

        '.BK': 'SET',           # Stock Exchange of Thailand
        '.JK': 'IDX',           # Indonesia Stock Exchange
        '.KL': 'KLSE',          # Kuala Lumpur Stock Exchange (Malaisie)

        # MOYEN-ORIENT & AFRIQUE
        '.TA': 'TA125',         # Tel Aviv Stock Exchange (Israël)
        '.CA': 'EGX',           # Egyptian Exchange
        '.JO': 'ASE',           # Amman Stock Exchange (Jordanie)

        # AMÉRIQUE DU SUD
        '.SA': 'BOVESPA',       # São Paulo Stock Exchange (Brésil)
        '.BA': 'MERVAL',        # Buenos Aires Stock Exchange (Argentine)
        '.SN': 'IPSA',          # Santiago Stock Exchange (Chili)
    }

    # Patterns pour identifier les actions US sans suffixe
    US_PATTERNS = {
        'NASDAQ': [
            # Tech giants et biotechs
            r'^(AAPL|MSFT|GOOGL|GOOG|AMZN|TSLA|META|NVDA|AMD|INTC|ADBE|CRM|NFLX|PYPL|CSCO|ORCL|AVGO|TXN|QCOM|INTU|ISRG|MRNA|GILD|AMGN|BIIB|REGN|VRTX|CELG)$',
            # Biotech/Pharma patterns
            r'^[A-Z]{3,4}(X|B|N)$',
            # 4-letter tickers (souvent NASDAQ)
            r'^[A-Z]{4}$',
        ],
        'NYSE': [
            # Industrielles traditionnelles
            r'^(JNJ|PG|KO|PEP|WMT|JPM|BAC|WFC|C|GS|MS|XOM|CVX|T|VZ|GE|CAT|BA|MMM|IBM|MCD|NKE|DIS|HD|UNH|V|MA)$',
            # 1-3 lettres (souvent NYSE)
            r'^[A-Z]{1,3}$',
        ]
    }

    index_mapping = {}

    for ticker in tickers:
        ticker_clean = ticker.upper().strip()

        # 1. Vérification des suffixes explicites
        suffix_found = False
        for suffix, index in YAHOO_SUFFIX_MAPPING.items():
            if ticker_clean.endswith(suffix):
                index_mapping[ticker] = index
                suffix_found = True
                break

        # 2. Si pas de suffixe, logique US
        if not suffix_found:
            # Vérification patterns NASDAQ
            for pattern in US_PATTERNS['NASDAQ']:
                if re.match(pattern, ticker_clean):
                    index_mapping[ticker] = 'NASDAQ'
                    suffix_found = True
                    break

            # Si toujours pas trouvé, vérification NYSE
            if not suffix_found:
                for pattern in US_PATTERNS['NYSE']:
                    if re.match(pattern, ticker_clean):
                        index_mapping[ticker] = 'NYSE'
                        suffix_found = True
                        break

            # Défaut : NASDAQ pour actions US sans suffixe
            if not suffix_found:
                index_mapping[ticker] = 'NASDAQ'

    return index_mapping

# 2. TÉLÉCHARGEMENT COMPLET DES INDICES
def download_comprehensive_indices(period="1y"):
    """
    Télécharge tous les indices majeurs mondiaux
    """
    indices_tickers = {
        # EUROPE
        'CAC': '^FCHI',         # CAC 40 (France)
        'AEX': '^AEX',          # AEX (Pays-Bas)
        'BEL20': '^BFX',        # BEL 20 (Belgique)
        'PSI20': 'PSI20.LS',    # PSI 20 (Portugal)
        'FTSE': '^FTSE',        # FTSE 100 (UK)
        'DAX': '^GDAXI',        # DAX (Allemagne)
        'FTSE_MIB': 'FTSEMIB.MI', # FTSE MIB (Italie)
        'IBEX': '^IBEX',        # IBEX 35 (Espagne)
        'SMI': '^SSMI',         # SMI (Suisse)
        'ATX': '^ATX',          # ATX (Autriche)
        'OSE': '^OSEAX',        # OBX (Norvège)
        'OMX': '^OMX',          # OMX Nordic (Nordiques)
        'WIG': '^WIG',          # WIG (Pologne)

        # AMÉRIQUE DU NORD
        'NASDAQ': '^IXIC',      # NASDAQ Composite
        'NYSE': '^NYA',         # NYSE Composite
        'NYSE_ARCA': '^XAX',    # NYSE Arca
        'SP500': '^GSPC',       # S&P 500
        'TSX': '^GSPTSE',       # S&P/TSX (Canada)
        'IPC': '^MXX',          # IPC (Mexique)

        # ASIE-PACIFIQUE
        'NIKKEI': '^N225',      # Nikkei 225 (Japon)
        'SSE': '000001.SS',     # Shanghai Composite
        'SZSE': '399001.SZ',    # Shenzhen Component
        'HSI': '^HSI',          # Hang Seng (Hong Kong)
        'KOSPI': '^KS11',       # KOSPI (Corée du Sud)
        'KOSDAQ': '^KQ11',      # KOSDAQ
        'TAIEX': '^TWII',       # Taiwan Weighted
        'STI': '^STI',          # Straits Times (Singapour)
        'ASX': '^AXJO',         # All Ordinaries (Australie)
        'NZX': '^NZ50',         # NZX 50 (Nouvelle-Zélande)
        'BSE': '^BSESN',        # BSE Sensex (Inde)
        'NSE': '^NSEI',         # Nifty 50 (Inde)

        # AMÉRIQUE DU SUD
        'BOVESPA': '^BVSP',     # Bovespa (Brésil)
        'MERVAL': '^MERV',      # Merval (Argentine)
        'IPSA': '^IPSA',        # IPSA (Chili)
    }

    indices_data = {}
    failed_downloads = []

    print(f"📈 TÉLÉCHARGEMENT DE {len(indices_tickers)} INDICES MONDIAUX:")
    print("-" * 60)

    for name, ticker in indices_tickers.items():
        try:
            data = yf.download(ticker, period=period, progress=False)
            if not data.empty and 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 50:  # Minimum de données
                    indices_data[name] = returns
                    print(f"✅ {name:<12} ({ticker:<12}) - {len(returns):>4} points")
                else:
                    failed_downloads.append((name, "Données insuffisantes"))
                    print(f"⚠️  {name:<12} ({ticker:<12}) - Données insuffisantes")
            else:
                failed_downloads.append((name, "Données vides"))
                print(f"❌ {name:<12} ({ticker:<12}) - Échec téléchargement")
        except Exception as e:
            failed_downloads.append((name, str(e)[:30]))
            print(f"❌ {name:<12} ({ticker:<12}) - Erreur: {str(e)[:30]}")

    print(f"\n✅ {len(indices_data)} indices téléchargés avec succès")
    if failed_downloads:
        print(f"❌ {len(failed_downloads)} échecs: {[x[0] for x in failed_downloads]}")

    return indices_data

# 3. CALCUL AVANCÉ DES BÊTAS
def calculate_comprehensive_betas(returns, indices_data, index_mapping, weights_vector):
    """
    Calcule les bêtas avec gestion d'erreurs avancée
    """
    betas_results = {}
    fallback_used = {}

    # Indices de fallback (plus liquides)
    FALLBACK_INDICES = {
        'CAC': ['SP500', 'NASDAQ'],
        'FTSE': ['SP500', 'NASDAQ'],
        'DAX': ['SP500', 'NASDAQ'],
        'NASDAQ': ['SP500'],
        'NYSE': ['SP500', 'NASDAQ'],
    }

    for i, ticker in enumerate(returns.columns):
        if ticker not in index_mapping:
            continue

        primary_index = index_mapping[ticker]
        beta_calculated = False

        # Tentative avec l'indice primaire
        if primary_index in indices_data:
            beta_result = _calculate_single_beta(
                returns[ticker], indices_data[primary_index], ticker, primary_index
            )
            if beta_result:
                beta_result['weight'] = weights_vector[i]
                betas_results[ticker] = beta_result
                beta_calculated = True

        # Fallback si échec
        if not beta_calculated and primary_index in FALLBACK_INDICES:
            for fallback_index in FALLBACK_INDICES[primary_index]:
                if fallback_index in indices_data:
                    beta_result = _calculate_single_beta(
                        returns[ticker], indices_data[fallback_index], ticker, fallback_index
                    )
                    if beta_result:
                        beta_result['weight'] = weights_vector[i]
                        beta_result['fallback'] = True
                        betas_results[ticker] = beta_result
                        fallback_used[ticker] = f"{primary_index} → {fallback_index}"
                        beta_calculated = True
                        break

        if not beta_calculated:
            print(f"⚠️  Impossible de calculer le beta pour {ticker} ({primary_index})")

    return betas_results, fallback_used

def _calculate_single_beta(stock_returns, index_returns, ticker, index_name):
    """
    Calcule le beta entre une action et un indice
    """
    try:
        # Alignement des dates
        stock_clean = stock_returns.dropna()
        common_dates = stock_clean.index.intersection(index_returns.index)

        if len(common_dates) < 50:  # Minimum 50 observations
            return None

        stock_aligned = stock_clean.loc[common_dates]
        index_aligned = index_returns.loc[common_dates]

        # Régression linéaire
        X = index_aligned.values.reshape(-1, 1)
        y = stock_aligned.values

        reg = LinearRegression().fit(X, y)

        return {
            'beta': reg.coef_[0],
            'alpha': reg.intercept_,
            'r_squared': reg.score(X, y),
            'index': index_name,
            'observations': len(common_dates),
            'fallback': False
        }

    except Exception as e:
        print(f"❌ Erreur calcul beta {ticker}: {str(e)}")
        return None

# 4. EXECUTION PRINCIPALE
print("🌍 ANALYSE MONDIALE DES BÊTAS")
print("=" * 60)

# Mapping complet
tickers = df['Tickers'].tolist()
index_mapping = get_comprehensive_index_mapping(tickers)

print(f"\n📊 MAPPING AUTOMATIQUE COMPLET:")
print("-" * 40)
mapping_stats = {}
for ticker, index in index_mapping.items():
    region = 'Europe' if index in ['CAC', 'FTSE', 'DAX', 'FTSE_MIB', 'IBEX', 'SMI'] else \
             'USA' if index in ['NASDAQ', 'NYSE', 'SP500'] else \
             'Asie' if index in ['NIKKEI', 'HSI', 'KOSPI'] else 'Autres'

    print(f"   {ticker:>12} → {index:<12} ({region})")
    mapping_stats[region] = mapping_stats.get(region, 0) + 1

print(f"\n📈 RÉPARTITION GÉOGRAPHIQUE:")
for region, count in mapping_stats.items():
    print(f"   {region:<10}: {count:>2} actions ({count/len(tickers)*100:.1f}%)")

# Téléchargement des indices
indices_data = download_comprehensive_indices("1y")

# Calcul des bêtas
betas_results, fallback_used = calculate_comprehensive_betas(
    returns, indices_data, index_mapping, weights_vector
)

# 5. AFFICHAGE DÉTAILLÉ
print(f"\n" + "=" * 80)
print("📊 BÊTAS INDIVIDUELS DÉTAILLÉS")
print("=" * 80)
print(f"{'Ticker':<10} {'Indice':<12} {'Beta':<8} {'Alpha':<9} {'R²':<8} {'Obs':<5} {'Poids':<8} {'Note'}")
print("-" * 80)

total_beta_weighted = 0
coverage = len(betas_results) / len(tickers)

# Statistiques par région
region_stats = {}

for ticker, data in betas_results.items():
    beta_contribution = data['beta'] * data['weight']
    total_beta_weighted += beta_contribution

    # Note qualitative
    note = "⭐⭐⭐" if data['r_squared'] > 0.7 else \
           "⭐⭐" if data['r_squared'] > 0.4 else "⭐"

    fallback_marker = " (FB)" if data.get('fallback', False) else ""

    # Stats par région
    region = 'Europe' if data['index'] in ['CAC', 'FTSE', 'DAX', 'FTSE_MIB', 'IBEX'] else \
             'USA' if data['index'] in ['NASDAQ', 'NYSE', 'SP500'] else 'Autres'

    if region not in region_stats:
        region_stats[region] = {'weight': 0, 'beta_weighted': 0, 'count': 0}
    region_stats[region]['weight'] += data['weight']
    region_stats[region]['beta_weighted'] += beta_contribution
    region_stats[region]['count'] += 1

    print(f"{ticker:<10} {data['index']:<12} {data['beta']:<8.3f} "
          f"{data['alpha']:<9.4f} {data['r_squared']:<8.3f} "
          f"{data['observations']:<5} {data['weight']:<8.2%} {note}{fallback_marker}")

print("-" * 80)
print(f"BETA PORTEFEUILLE GLOBAL : {total_beta_weighted:.3f}")
print(f"COUVERTURE DE L'ANALYSE  : {coverage:.1%} ({len(betas_results)}/{len(tickers)} actions)")

# Fallbacks utilisés
if fallback_used:
    print(f"\n⚠️  INDICES DE SUBSTITUTION UTILISÉS:")
    for ticker, substitution in fallback_used.items():
        print(f"   {ticker}: {substitution}")

# 6. ANALYSE PAR RÉGION
print(f"\n" + "=" * 60)
print("🌍 ANALYSE PAR RÉGION")
print("=" * 60)

for region, stats in region_stats.items():
    avg_beta = stats['beta_weighted'] / stats['weight'] if stats['weight'] > 0 else 0
    print(f"{region:<10}: {stats['weight']:<8.1%} | Beta {avg_beta:<6.3f} | {stats['count']:>2} actions")


print(f"\n🎯 RECOMMANDATIONS FINALES:")
if total_beta_weighted > 1.3:
    print("⚠️  Portefeuille très risqué - Considérez des actifs défensifs")
elif total_beta_weighted < 0.7:
    print("💤 Portefeuille peu risqué - Potentiel de rendement limité")
else:
    print("✅ Niveau de risque équilibré pour un portefeuille diversifié")

if coverage < 0.8:
    print(f"⚠️  Couverture incomplète ({coverage:.1%}) - Vérifiez les tickers manquants")

#rajouter la diversification
#type d'actif et concentration



