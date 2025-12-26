import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

# ----------------------------
# Generate demographic features
# ----------------------------
age = np.random.randint(1, 95, N)
gender = np.random.choice([0, 1], N)  # 0 = Female, 1 = Male
chronic_condition = np.random.choice([0, 1], N, p=[0.75, 0.25])

# ----------------------------
# Generate vital signs
# ----------------------------
heart_rate = np.random.normal(85, 18, N).clip(40, 180)
systolic_bp = np.random.normal(120, 20, N).clip(70, 200)
diastolic_bp = np.random.normal(80, 12, N).clip(40, 120)
spo2 = np.random.normal(96, 3, N).clip(75, 100)
resp_rate = np.random.normal(18, 5, N).clip(8, 45)
temperature = np.random.normal(98.4, 1.2, N).clip(95, 104)

pain_score = np.random.randint(0, 11, N)
consciousness = np.random.choice([0, 1], N, p=[0.08, 0.92])
arrival_mode = np.random.choice([0, 1], N, p=[0.7, 0.3])  # 1 = Ambulance

# ----------------------------
# Clinical risk scoring (NON-LINEAR)
# ----------------------------
risk_score = (
    (heart_rate > 120).astype(int) +
    (systolic_bp < 90).astype(int) * 2 +
    (spo2 < 92).astype(int) * 2 +
    (resp_rate > 30).astype(int) +
    (temperature > 101).astype(int) +
    (pain_score > 7).astype(int) +
    (consciousness == 0).astype(int) * 3 +
    chronic_condition +
    arrival_mode
).astype(float)

# Inject uncertainty (very important)
risk_score += np.random.normal(0, 1.3, N)

# ----------------------------
# Triage labeling with overlap
# ----------------------------
triage_level = np.where(
    risk_score >= 6, 3,
    np.where(risk_score >= 4, 2,
    np.where(risk_score >= 2, 1, 0))
)

# ----------------------------
# Create DataFrame
# ----------------------------
df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "heart_rate": heart_rate,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "spo2": spo2,
    "respiratory_rate": resp_rate,
    "temperature": temperature,
    "pain_score": pain_score,
    "consciousness": consciousness,
    "arrival_mode": arrival_mode,
    "chronic_condition": chronic_condition,
    "triage_level": triage_level
})

# ----------------------------
# Add missing values (real hospital data)
# ----------------------------
for col in ["heart_rate", "spo2", "systolic_bp"]:
    df.loc[df.sample(frac=0.07).index, col] = np.nan

# ----------------------------
# Save dataset
# ----------------------------
df.to_csv("ai_emergency_triage_dataset.csv", index=False)

print("Dataset generated successfully!")
print("Shape:", df.shape)
print("\nClass distribution:")
print(df["triage_level"].value_counts(normalize=True))
