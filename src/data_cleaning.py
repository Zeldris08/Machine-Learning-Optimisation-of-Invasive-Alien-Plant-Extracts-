import pandas as pd


# ----------------------------
# File names
# ----------------------------

MEASUREMENTS_FILE = "C:/Users/rudsi/Desktop/Results(25).csv"
MAPPING_FILE = "mapping(25).csv"
OUTPUT_FILE = "seedling_measurements(25).csv"


# ----------------------------
# Load data
# ----------------------------

measurements = pd.read_csv(MEASUREMENTS_FILE)
mapping = pd.read_csv(MAPPING_FILE)

# Make lookup table using measurement ID

length_lookup = measurements.set_index("ID")["Length"]

rows = []

for _, row in mapping.iterrows():

    root_ids = range(
        int(row["Root_Start"]),
        int(row["Root_End"]) + 1
    )

    root_length = length_lookup.loc[list(root_ids)].sum()

    shoot_length = length_lookup.loc[
        int(row["Shoot"])
    ]

    rows.append({

        "Replicate": int(row["Replicate"]),
        "Seedling": int(row["Seed"]),
        "Root (mm)": round(root_length, 3),
        "Shoot (mm)": round(shoot_length, 3)

    })

output = pd.DataFrame(rows)

output.to_csv(OUTPUT_FILE, index=False)

print(output)
print()
print(f"Saved to {OUTPUT_FILE}")