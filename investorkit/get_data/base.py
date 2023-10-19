from typing import Union, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger  # Using Loguru for logging

try:
    from tqdm import tqdm

    ENABLE_TQDM = True
except ImportError:
    ENABLE_TQDM = False


### Useful for debug
# tickers=list(df['symbol'])
# statement="cashflow"
# api_key=FMP_API_KEY
# quarter = True
# start_date="2000-01-01"
# end_date=None
# rounding = 4
# progress_bar = True


def get_financial_statements(
    tickers: Union[str, List[str]],
    statement: str = "",
    api_key: str = "",
    quarter: bool = True,
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
    rounding: Union[int, None] = 4,
    progress_bar: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    if not isinstance(tickers, (list, str)):
        raise ValueError(f"Invalid type for tickers: {type(tickers)}")

    ticker_list = tickers if isinstance(tickers, list) else [tickers]

    statement_to_location = {
        "balance": "balance-sheet-statement",
        "income": "income-statement",
        "cashflow": "cash-flow-statement",
    }

    location = statement_to_location.get(statement)
    if location is None:
        raise ValueError(
            "Invalid statement type. Choose 'balance', 'income', or 'cashflow'."
        )

    period = "quarter" if quarter else "annual"
    financial_statement_dict = {}
    invalid_tickers = []

    ticker_iterator = (
        tqdm(ticker_list, desc=f"Obtaining {statement} data")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_iterator:
        url = f"https://financialmodelingprep.com/api/v3/{location}/{ticker}?period={period}&apikey={api_key}"

        try:
            financial_statement = pd.read_json(url)
            if financial_statement.empty:
                invalid_tickers.append(ticker)
                continue
        except Exception as error:
            invalid_tickers.append(ticker)
            continue

        date_col = "date" if quarter else "calendarYear"
        freq = "Q" if quarter else "Y"
        financial_statement[date_col] = pd.to_datetime(
            financial_statement[date_col].astype(str)
        ).dt.to_period(freq)

        financial_statement_dict[ticker] = financial_statement

    if not financial_statement_dict:
        return pd.DataFrame(), invalid_tickers

    financial_statement_total = pd.concat(financial_statement_dict)
    financial_statement_total.reset_index(drop=True, inplace=True)
    financial_statement_total = financial_statement_total.drop_duplicates().reset_index(
        drop=True
    )

    if start_date or end_date:
        mask = True
        if start_date:
            mask &= financial_statement_total["date"] >= start_date
        if end_date:
            mask &= financial_statement_total["date"] <= end_date
        financial_statement_total = financial_statement_total[mask]

    financial_statement_total["date"] = financial_statement_total["date"].astype(str)
    return financial_statement_total, invalid_tickers


def get_profile(tickers, api_key):
    if not isinstance(tickers, (list, str)):
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    tickers = tickers if isinstance(tickers, list) else [tickers]
    profiles = {}

    for ticker in tqdm(tickers):
        try:
            data = pd.read_json(
                f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
            )
            profiles[ticker] = data
        except Exception as error:
            logger.warning(f"Could not fetch data for {ticker}. Error: {error}")

    profile_dataframe = pd.concat(profiles)
    profile_dataframe = profile_dataframe.reset_index(drop=True)

    return profile_dataframe


def get_historical_market_cap(
    tickers: Union[str, List[str]],
    api_key: str = "",
    start_date: Union[str, None] = None,
) -> pd.DataFrame:
    if not isinstance(tickers, (list, str)):
        raise ValueError(f"Invalid type for tickers: {type(tickers)}")

    ticker_list = tickers if isinstance(tickers, list) else [tickers]
    df_marketcap = pd.DataFrame()

    for ticker in ticker_list:
        url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}?&apikey={api_key}"
        try:
            data_mod = pd.read_json(url)
            if data_mod.empty:
                continue

            data_mod["date"] = pd.to_datetime(data_mod["date"])

            if start_date:
                start_date_dt = pd.to_datetime(start_date)
                data_mod = data_mod[data_mod["date"] > start_date_dt]

            df_marketcap = pd.concat([df_marketcap, data_mod], ignore_index=True)
        except Exception as error:
            logger.warning(f"Could not fetch data for {ticker}. Error: {error}")

    return df_marketcap


def get_historical_prices(
    tickers: List[str],
    new_tickers: List[str],
    api_key: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    df_final_price = pd.DataFrame()

    ticker_iterator = tqdm(tickers)
    for ticker in ticker_iterator:
        base_url = "https://financialmodelingprep.com/api/v3/historical-price-full/"
        if ticker not in new_tickers:
            url = f"{base_url}{ticker}?from={start_date}&to={end_date}&apikey={api_key}"
        else:
            url = f"{base_url}{ticker}?from=2000-01-01&to={end_date}&apikey={api_key}"

        try:
            df_price = pd.read_json(url)
            if "historical" in df_price.columns:
                exploded_df = df_price["historical"].apply(pd.Series)
                final_df = pd.concat([df_price["symbol"], exploded_df], axis=1)
                df_final_price = pd.concat(
                    [df_final_price, final_df], ignore_index=True
                )

        except Exception as e:
            logger.warning(f"Could not fetch data for {ticker}. Error: {e}")

    logger.info(f"Shape of historical prices DataFrame: {df_final_price.shape}")

    df_final_price["date"] = pd.to_datetime(df_final_price["date"])

    return df_final_price
