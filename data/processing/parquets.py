import os
from functools import reduce

import pandas as pd

BASE_DIR = "/home/diego/Documents/Python Projects/TCC/data"
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
DAILY_RAW_DIR = os.path.join(BASE_DIR, "_raw_data", "daily")
QUARTERLY_RAW_DIR = os.path.join(BASE_DIR, "_raw_data", "quarterly")
PARQUET_DIR = os.path.join(BASE_DIR, "parquets")


def _read_cleaned_csv(file_path: str) -> pd.DataFrame:
	df = pd.read_csv(file_path, sep=";", decimal=",")
	df = df.rename(columns={df.columns[0]: "date"})
	df["date"] = pd.to_datetime(df["date"], dayfirst=True)
	return df


def _get_file_list(directory: str) -> list[str]:
	if not os.path.isdir(directory):
		return []
	return sorted([f for f in os.listdir(directory) if f.endswith(".csv")])


def _infer_tickers(reference_files: list[str]) -> list[str]:
	max_cols = 0
	tickers: list[str] = []
	for file_name in reference_files:
		file_path = os.path.join(CLEANED_DIR, file_name)
		if not os.path.exists(file_path):
			continue
		df = _read_cleaned_csv(file_path)
		cols = [c for c in df.columns if c != "date"]
		if len(cols) > max_cols:
			max_cols = len(cols)
			tickers = cols
	return tickers


def _build_time_series_features(daily_files: list[str], tickers: list[str]) -> pd.DataFrame:
	per_ticker_frames: list[pd.DataFrame] = []
	macro_frames: list[pd.DataFrame] = []

	for file_name in daily_files:
		file_path = os.path.join(CLEANED_DIR, file_name)
		if not os.path.exists(file_path):
			continue
		df = _read_cleaned_csv(file_path)
		cols = [c for c in df.columns if c != "date"]
		feature_name = os.path.splitext(file_name)[0]

		if tickers and set(cols).issubset(set(tickers)):
			melted = df.melt(id_vars="date", var_name="ticker", value_name=feature_name)
			per_ticker_frames.append(melted)
		else:
			macro_frames.append(df)

	if per_ticker_frames:
		x_ts = reduce(
			lambda left, right: left.merge(right, on=["date", "ticker"], how="outer"),
			per_ticker_frames,
		)
	else:
		x_ts = pd.DataFrame(columns=["date", "ticker"])

	if macro_frames:
		macro_df = reduce(lambda left, right: left.merge(right, on="date", how="outer"), macro_frames)
		x_ts = x_ts.merge(macro_df, on="date", how="left")

	return x_ts


def _build_static_features(quarterly_files: list[str], tickers: list[str]) -> pd.DataFrame:
	frames: list[pd.DataFrame] = []
	for file_name in quarterly_files:
		file_path = os.path.join(CLEANED_DIR, file_name)
		if not os.path.exists(file_path):
			continue
		df = _read_cleaned_csv(file_path)
		cols = [c for c in df.columns if c != "date"]
		feature_name = os.path.splitext(file_name)[0]

		if tickers and not set(cols).issubset(set(tickers)):
			continue

		melted = df.melt(id_vars="date", var_name="ticker", value_name=feature_name)
		frames.append(melted)

	if not frames:
		return pd.DataFrame(columns=["date", "ticker"])

	return reduce(
		lambda left, right: left.merge(right, on=["date", "ticker"], how="outer"),
		frames,
	)


def build_parquets() -> None:
	os.makedirs(PARQUET_DIR, exist_ok=True)

	daily_files = _get_file_list(DAILY_RAW_DIR)
	quarterly_files = _get_file_list(QUARTERLY_RAW_DIR)
	tickers = _infer_tickers(daily_files)

	x_ts = _build_time_series_features(daily_files, tickers)
	x_static = _build_static_features(quarterly_files, tickers)

	x_ts = x_ts.sort_values(["date", "ticker"])
	x_static = x_static.sort_values(["date", "ticker"])

	x_ts.to_parquet(os.path.join(PARQUET_DIR, "x_ts.parquet"), index=False)
	x_static.to_parquet(os.path.join(PARQUET_DIR, "x_static.parquet"), index=False)

	_preview_parquet(os.path.join(PARQUET_DIR, "x_ts.parquet"), label="X_ts")
	_preview_parquet(os.path.join(PARQUET_DIR, "x_static.parquet"), label="X_static")


def _preview_parquet(file_path: str, label: str) -> None:
	if not os.path.exists(file_path):
		print(f"{label} not found: {file_path}")
		return

	df = pd.read_parquet(file_path)
	print("=" * 80)
	print(f"Preview {label}")
	print(f"Shape: {df.shape}")
	print("Dtypes:")
	print(df.dtypes)
	print("Head:")
	print(df.head())


if __name__ == "__main__":
	build_parquets()