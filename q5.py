import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_prior(df, Y):
    # Calculate the prior probabilities P(Y)
    classes = sorted(list(df[Y].unique()))
    prior = [np.mean(df[Y] == c) for c in classes]
    return prior

def calculate_likelihood_gaussian(df, feature, x, Y, c, epsilon=1e-9):
    # Calculate the Gaussian likelihood P(X|Y)
    mean = df[feature][df[Y] == c].mean()
    std = df[feature][df[Y] == c].std() + epsilon  # Add epsilon to avoid division by zero
    exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def naive_bayes_gaussian(df, X, Y):
    # Get feature names
    features = list(df.columns)[:-1]

    # Calculate prior probabilities
    prior = calculate_prior(df, Y)

    Y_pred = []
    # Loop over every data sample
    for x in X:
        # Calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        # Calculate posterior probability (numerator only)
        post_prob = [1] * len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)

# Example usage
# Load your dataset (adjust the path to your actual CSV file)
df = pd.read_csv('Titanic.csv')

# Select relevant columns
df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[['Age', 'Fare']] = imputer.fit_transform(df[['Age', 'Fare']])
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Split the data into train and test sets
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine the training set into a single dataframe for passing to the function
train_df = pd.concat([X_train, y_train], axis=1)

# Convert to numpy array for iteration
X_train_np = X_train.values
X_test_np = X_test.values

# Train the model
predictions_train = naive_bayes_gaussian(train_df, X_train_np, 'Survived')

# Predict the classes for the test set
predictions_test = naive_bayes_gaussian(train_df, X_test_np, 'Survived')

# Evaluate the model
accuracy = accuracy_score(y_test, predictions_test)
cm = confusion_matrix(y_test, predictions_test)

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
