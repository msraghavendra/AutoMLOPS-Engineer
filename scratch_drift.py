import json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ref_data = {
    "feature_1": [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3]*5,
    "feature_2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]*5
}
ref_df = pd.DataFrame(ref_data)

current_data = {
    "feature_1": [10.0, 20.0, 15.0, 25.0, 18.0, 22.0, 19.0, 21.0, 17.0, 23.0]*5,
    "feature_2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]*5
}
current_df = pd.DataFrame(current_data)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=current_df)
result_json = report.json()
data = json.loads(result_json)

# Print all metrics
print(f"Total metrics in report: {len(data['metrics'])}")
for idx, metric_item in enumerate(data["metrics"]):
    print(f"\n--- Metric {idx}: {metric_item['metric']} ---")
    print("Result keys:", list(metric_item["result"].keys()))
    if "drift_by_columns" in metric_item["result"]:
        print("Found drift_by_columns! Column details:")
        for col, col_data in metric_item["result"]["drift_by_columns"].items():
            print(f"  Col '{col}' keys: {list(col_data.keys())}")
            # print first feature's dict values to see what keys exist
            print(f"  Col '{col}' sample: {col_data}")
            break

