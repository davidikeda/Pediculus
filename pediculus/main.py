import pandas as pd
import os, subprocess

from experiment import run
from config import DATASET_SIZES, TRAILS_PER_SIZE

def checkIfWifiEnabled():
    try:
        output = subprocess.check_output(
            "netsh wlan show interfaces",
            shell=True
        ).decode("utf-8", errors="ignore")

        if "State" in output and "connected" in output.lower():
            for line in output.split("\n"):
                if "SSID" in line and "BSSID" not in line:
                    ssid = line.split(":")[1].strip()
                    print(f"Connected to Wi-Fi SSID: {ssid}")
                    return True

        print("Not connected to Wi-Fi.")
        return False

    except subprocess.CalledProcessError:
        print(
            "Could not check Wi-Fi status (Command failed). "
            "If you are on a mac/linux you can ignore this message as the check is for windows only computers."
        )
        return False

def summarize(df):
    summary = df.groupby("fraction").agg(
        avg_accuracy=("accuracy", "mean"),
        avg_fpr=("false_positive_rate", "mean"),
        std_accuracy=("accuracy", "std"),
        std_fpr=("false_positive_rate", "std")
    ).reset_index()

    print("\nsummary by training set size")
    print(summary.to_string(index=False))
    return summary

def main():
    checkIfWifiEnabled()
    results = run()

    print("\nresults:")
    print(f"Result rows: {len(results)}")
    print("Tickers:", sorted(results["ticker"].unique()))
    print("Dataset fractions:", sorted(results["fraction"].unique()))

    summary = summarize(results)

    oPath = "results.csv"
    results.to_csv(oPath, index=False)
    print(f"\nsaved full results to {oPath}")


# main entrypoint

main()