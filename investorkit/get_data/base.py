from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.WARNING)

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
    """
    Retrieves financial statements for one or multiple companies and returns a DataFrame.

    Args:
        tickers: List of company tickers.
        statement: Type of financial statement ("balance", "income", or "cashflow").
        api_key: API key for financial data provider.
        quarter: Whether to retrieve quarterly data.
        start_date: Start date to filter data.
        end_date: End date to filter data.
        rounding: Rounding precision.
        progress_bar: Show progress bar for more than 10 tickers.

    Returns:
        Tuple containing the DataFrame of financial statements and a list of invalid tickers.

    Example:

        df, invalid_tickers = get_financial_statements(
            tickers=["AAPL", "META"],
            statement="cashflow",
            api_key="your_api_key_here",
            start_date="2000-01-01"
        )
    """

    # Ensure tickers is a list
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
                logging.warning(f"Received empty data for {ticker}")
                invalid_tickers.append(ticker)
                continue

        except Exception as error:
            logging.warning(f"Could not fetch data for {ticker}. Error: {error}")
            invalid_tickers.append(ticker)
            continue

        # Convert date to appropriate format
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
    """
    Description
    ----
    Gives information about the profile of a company which includes
    i.a. beta, company description, industry, and sector.

    Parameters
    ----
    tickers : list or str
        The company tickers (e.g., "AAPL" or ["AAPL", "GOOGL"]).
    api_key : str
        The API Key obtained from Financial Modeling Prep.

    Returns
    ----
    pd.DataFrame
        Data with variables in rows and tickers in columns.
    """
    # Ensure tickers is a list
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
            print(f"Warning: Could not fetch data for {ticker}. Error: {error}")

    profile_dataframe = pd.concat(profiles)
    profile_dataframe = profile_dataframe.reset_index(drop=True)

    return profile_dataframe
