import json
import pandas as pd

from sklearn.preprocessing import StandardScaler


# Function for preparing dataframe for model
def prepare_df(data: pd.DataFrame) -> pd.DataFrame:
    # Delete index, email and phone numbers
    data = data.drop(['id', 'x5', 'x6'], axis=1)

    # Get month and day of week
    data['month'] = data['dt'].dt.month
    data['day_of_week'] = data['dt'].dt.day_of_week
    data = data.drop('dt', axis=1)

    # Fill missing values in x2 with median
    data['x2'] = data['x2'].fillna(data.x2.median())

    # Fill the missing values in binary columns with mode and convert them to int type
    cols = ['x7', 'x8', 'x9', 'x10']

    for col in cols:
        mode_value = data[col].mode()[0]
        data[col] = data[col].fillna(mode_value).astype(int)

    # Let's fill the missing values in categorical columns with mode
    cat_cols = ['x3', 'x4']

    for col in cat_cols:
        mode_value = data[col].mode()[0]
        data[col] = data[col].fillna(mode_value)

    # Parse JSON
    data['x11'] = data['x11'].apply(safe_json_loads)
    data[['total_score', 'max_score', 'min_score', 'mean_score']] = data['x11'].apply(
        lambda x: pd.Series(calculate_score_stats(x)))

    data = data.drop('x11', axis=1)
    data.drop(['total_score', 'mean_score'], axis=1, inplace=True)

    # OHE
    ohe_cols = ['x3', 'x4']
    data = pd.get_dummies(data, columns=ohe_cols, dtype=int)

    # Scaler
    scaler = StandardScaler()
    numerical_features_df = [i for i in data.describe(include='number').columns]
    numerical_features_df.remove('y')
    data[numerical_features_df] = scaler.fit_transform(data[numerical_features_df])

    return data


# Load info from json
def safe_json_loads(s):
    try:
        return json.loads(s)
    except ValueError:
        return None


# Create new features from score
def calculate_score_stats(dicts):
    scores = [d.get('score') for d in dicts if 'score' in d]
    if not scores:
        return {'total_score': None, 'max_score': None, 'min_score': None}
    total_score = sum(scores)
    max_score = max(scores)
    min_score = min(scores)
    mean_score = sum(scores) / len(scores)
    return {'total_score': total_score, 'max_score': max_score, 'min_score': min_score, 'mean_score': mean_score}


# Load data
path = 'data/text_data_model.xlsx'
df = pd.read_excel(path)

print(prepare_df(df))
