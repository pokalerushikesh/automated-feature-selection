import seaborn as sns
from feature_selector import FeatureSelector
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the Titanic dataset
df = sns.load_dataset('titanic')
df.drop(['deck', 'embark_town', 'class', 'who'], axis=1, inplace=True)
df.dropna(inplace=True)
df[['adult_male', 'alone']] = df[['adult_male', 'alone']].astype(int)

label_cols = ['sex', 'embarked', 'alive']
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Prepare features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Initialize and run FeatureSelector
fs = FeatureSelector(X, y)

corr_features = fs.correlation_filter()
mi_scores = fs.mutual_info()
rfe_features = fs.rfe(n_features_to_select=5)
tree_importances = fs.tree_importance()

# Print results
print("RFE selected features:", rfe_features)
print("Top 5 by Mutual Information:\n", mi_scores.head())
print("Top 5 by Tree-Based Importance:\n", tree_importances.head())