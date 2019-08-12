
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'notebooks'))
    print(os.getcwd())
except:
    pass

# %%
import os
import tarfile
import typing

from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
parent_dir = os.path.abspath('..')
dataset_dir = os.path.join(parent_dir, "dataset")
HOUSING_PATH = os.path.join(dataset_dir, "apartments")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()


fetch_housing_data()


# %%


def load_housing_data(housing_path: str = HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing_data: pd.DataFrame = load_housing_data()

housing_data.head()


# %%
housing_data.info()


# %%
housing_data["ocean_proximity"].value_counts()


# %%
housing_data.describe()


# %%
get_ipython().run_line_magic('matplotlib', 'inline')

housing_data.hist(bins=50, figsize=(20, 15))
plt.show()


# %%


def split_train_test(data: pd.DataFrame, test_ratio: float) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing_data, 0.2)


# %%
housing_data["income_cat"] = np.ceil(housing_data["median_income"] / 1.5)
housing_data["income_cat"].where(
    housing_data["income_cat"] < 5, 5.0, inplace=True)
housing_data.hist(bins=50, figsize=(20, 15))
plt.show()
housing_data.describe()


# %%

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]

housing_data["income_cat"].value_counts()/len(housing_data)


# %%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing_data.describe()


# %%
housing_data: pd.DataFrame = strat_train_set.copy()
housing_data.plot(kind="scatter", x="longitude", y="latitude")

housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                  s=housing_data["population"]/100, label="Population", figsize=(10, 7),
                  c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                  )
plt.legend()


# %%

corr_matrix: pd.DataFrame = housing_data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
attributes = ["median_house_value", "median_income",
              "total_rooms", "housing_median_age"]
scatter_matrix(housing_data[attributes], figsize=(12, 8))

housing_data.plot(kind="scatter", x="median_income",
                  y="median_house_value", alpha=0.1)


# %%
housing_data["rooms_per_household"] = housing_data["total_rooms"] / \
    housing_data["households"]
housing_data["bedrooms_per_rooms"] = housing_data["total_bedrooms"] / \
    housing_data["total_rooms"]
housing_data["population_per_household"] = housing_data["population"] / \
    housing_data["households"]

corr_matrix: pd.DataFrame = housing_data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# %%
housing_data = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# clean data
# housing_data.dropna(subset=["total_bedrooms"]) # first option -> drop rows with missing value
# housing_data.drop("total_bedroooms", axis = 1) # second option -> drop whole attribute
# median = housing_data["total_bedrooms"].median()
# housing_data["total_bedrooms"].fillna(median, inplace = True )


# %%

imputer = SimpleImputer(strategy="median")
housing_data_num = housing_data.drop("ocean_proximity", axis=1)
imputer.fit(housing_data_num)
imputer.statistics_


# %%
housing_data_num.median().values


# %%
transform_table = imputer.transform(housing_data_num)
housing_tr = pd.DataFrame(transform_table, columns=housing_data_num.columns)


# %%
encoder = LabelEncoder()
housing_cat = housing_data["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


# %%
print(encoder.classes_)


# %%

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
housing_cat_1hot

# %%
housing_cat_1hot.toarray()

# %%
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot


# %%

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_households = X[:, rooms_ix] / X[:, household_ix]
        population_per_households = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_households, population_per_households, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_households, population_per_households]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing_data.values)


# %% pipes

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_data_num)
housing_num_tr

# %%


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output

    def fit(self, X, y=None):
        self.enc = LabelBinarizer(sparse_output=self.sparse_output)
        self.enc.fit(X)
        return self

    def transform(self, X, y=None):
        return self.enc.transform(X)


num_attribs = list(housing_data_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CustomLabelBinarizer()),
])


# %%


full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_data_prepared = full_pipeline.fit_transform(housing_data)
housing_data_prepared


# %%
housing_data_prepared.shape

# %% training linear model
lin_reg = LinearRegression()
lin_reg.fit(housing_data_prepared, housing_labels)

some_data = housing_data.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predicted: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))


# %%
