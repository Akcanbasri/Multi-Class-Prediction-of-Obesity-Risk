import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder,
    StandardScaler,
    RobustScaler,
)
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    validation_curve,
    cross_validate,
)

pd.set_option("display.max_colwidth", None)  # Increase the width of the output
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

""""
Features:
    FAVC: Frequent consumption of high caloric food,(Yüksek kalorili gıdaların sık tüketilmesi)
    FCVC: Frequency of consumption of vegetables, (Sebze tüketim sıklığı)
    NCP: Number of main meals, (Ana öğün sayısı)
    CAEC: Consumption of food between meals, (Öğün arası yemek tüketimi)
    CH20: Consumption of water daily, (Günlük su tüketimi)
    SCC: Calories consumption monitoring, (Kalori tüketimi izleme)
    FAF: Physical activity frequency, (Fiziksel aktivite sıklığı)
    TUE: Time using technology devices, (Teknoloji cihazları kullanma süresi)
    CALC: Consumption of alcohol. (Alkol tüketimi)
    MTRANS: Transportation used. (Kullanılan ulaşım aracı)
    Gender, (Cinsiyet)
    Age, (Yaş)
    Height and (Boy)
    Weight. (Kilo)
    NObeyesdad: Insufficient Weight, (Yetersiz kilo)F


Targets:
    Insufficient_Weight: label 0
    Normal_Weight: label 1
    Obesity_Type_I: label 2
    Obesity_Type_II: label 3
    Obesity_Type_III: label 4
    Overweight_Level_I: label 5
    Overweight_Level_II: label 6
"""

# importin data from csv files
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
sample_submission = pd.read_csv("./data/sample_submission.csv")

#############################################
# Exploratory Data Analysis
#############################################


# check_train function for checking the data
def check_train(data_frame, head=5):
    print("##################### Shape #####################")
    print(data_frame.shape)
    print("##################### Types #####################")
    print(data_frame.dtypes)
    print("##################### Head #####################")
    print(data_frame.head(head))
    print("##################### Tail #####################")
    print(data_frame.tail(head))
    print("##################### NA #####################")
    print(data_frame.isnull().sum())
    print("##################### Describe #####################")
    print(data_frame.describe().T)


check_train(train)


# grab column names for categorical, numerical, categorical but cardinal variables
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """we can use this function for getting categorical, numerical, categorical but cardinal variables names

    Args:
    -------
        dataframe (pandas.DataFrame): all data
        cat_th ([int, floot], optional):

        numeric fakat kategorik olan değişkenler için sınıf eşiği. Defaults to 10.

        car_th ([int, floot], optional):
        katagorik fakat kardinal değişkenler için sınıf eşik değeri. Defaults to 20.

    Returns:
    -------
    cat_cols: List
        kategorik değişken isimleri

    num_cols: List
        numerik değişken isimleri

    cat_but_car: List
        kategorik görünüp aslında kardinal olan değişken isimleri

    Notes:
    -------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat  cat_cols un içinde
        return olan üç liste toplamı toplam değişken sayısına eşittir.

    """

    # cat_cols, cat_but_car
    cat_cols = [
        col
        for col in dataframe.columns
        if str(dataframe[col].dtypes) in ["object", "category", "bool"]
    ]

    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th
        and dataframe[col].dtypes in ["int64", "float64"]
    ]

    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th
        and str(dataframe[col].dtypes) in ["object", "category"]
    ]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [
        col
        for col in dataframe.columns
        if dataframe[col].dtypes in ["int64", "float64"] and col not in cat_cols
    ]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(train)

# drop the id column for outlier detection
num_cols = [col for col in num_cols if "id" not in col]


# Summary of categorical variabes
def cat_summary(data_frame, colm_name, plot=False):
    print(
        pd.DataFrame(
            {
                colm_name: data_frame[colm_name].value_counts(),
                "Ratio": 100 * data_frame[colm_name].value_counts() / len(data_frame),
            }
        )
    )

    if plot:
        sns.countplot(x=data_frame[colm_name], data=data_frame)
        plt.show(block=True)


for col in cat_cols:
    print(f"##################### {col} #####################")
    # bool type columns are converted to int type
    if train[col].dtypes == "bool":
        train[col] = train[col].astype(int)
        cat_summary(train, col, plot=True)
    else:
        cat_summary(train, col, plot=True)


# Summary of numerical variables
def num_summary(dataframe, num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)

    if plot:
        dataframe[num_cols].hist(bins=20)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show(block=True)


for col in num_cols:
    # wrrite the variable name
    print(f"########################### {col} ###########################")
    num_summary(train, col, plot=True)

train.head()

# label encoding for NObeyesdad column for visualization
label_encoder = LabelEncoder()
train["NObeyesdad_encoded"] = label_encoder.fit_transform(train["NObeyesdad"])


# target variable analysis for categorical variables
def target_summary_with_cat(dataframe, target, categorical_col, graph=False):
    # show percentage(?/100) of target variable for categorical variables
    print(
        dataframe.groupby(categorical_col)
        .agg({target: ["count", "mean", "median", "std"]})
        .sort_values(by=(target, "mean"), ascending=False)
    )

    # barplot for categorical variables
    if graph:
        sns.barplot(data=dataframe, x=categorical_col, y=target)
        plt.show(block=True)


for col in cat_cols:
    print(f"##################### {col} #####################")
    target_summary_with_cat(train, "NObeyesdad_encoded", col, graph=True)


# target variable analysis for numerical variables
def target_summary_with_num(dataframe, target, numerical_col, graph=False):
    # show percentage(?/100) of target variable for numerical variables
    print(
        dataframe.groupby(target)
        .agg({numerical_col: ["count", "mean", "median", "std"]})
        .sort_values(by=(numerical_col, "mean"), ascending=False)
    )

    # barplot for numerical variables
    if graph:
        sns.barplot(data=dataframe, x=target, y=numerical_col)
        plt.show(block=True)


for col in num_cols:
    print(f"##################### {col} #####################")
    target_summary_with_num(train, "NObeyesdad_encoded", col, graph=True)


# Correlation matrix analysis for numerical variables
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    This function takes a pandas dataframe as input and returns a list of column names that have a correlation
    coefficient greater than the specified threshold. If plot is set to True, it also displays a heatmap of the
    correlation matrix using seaborn and matplotlib.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    plot (bool): Whether or not to display a heatmap of the correlation matrix. Default is False.
    corr_th (float): The correlation threshold. Columns with a correlation coefficient greater than this value
                     will be included in the returned list. Default is 0.90.

    Returns:
    list: A list of column names that have a correlation coefficient greater than the specified threshold.
    """
    # Calculate the correlation matrix
    corr_matrix = dataframe.corr().abs()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create a series with the correlation coefficients
    high_corr = corr_matrix.mask(mask).stack().sort_values(ascending=False)

    # Filter the series to include only the correlations above the threshold
    high_corr = high_corr[high_corr > corr_th]

    # Convert the series index to a list of tuples
    high_corr.index = high_corr.index.to_list()

    # Create a list of the column names
    high_corr_list = [x[0] for x in high_corr.index]

    # Display the heatmap
    if plot:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    return high_corr_list


high_corr_list = high_correlated_cols(train, plot=True, corr_th=0.90)

# drop highly correlated columns
train = train.drop(high_corr_list, axis=1)

train.shape

#############################################
# Outliers
#############################################


# outlier_thresholds function for finding the limits of outliers
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# check_outlier function for checking the outliers
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ].any(axis=None):
        return True
    else:
        return False


cat_cols, num_cols, cat_but_car = grab_col_names(train)

# drop id from num cols
num_cols = [col for col in num_cols if "id" not in col]


train.head()

# check the outliers before replacing
for col in num_cols:
    print(f"##################### {col} #####################")
    print(check_outlier(train, col, q1=0.1, q3=0.9))


# grab_outliers function for grabbing the outliers
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if (
        dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0]
        > 10
    ):
        print(
            dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head()
        )
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[
            ((dataframe[col_name] < low) | (dataframe[col_name] > up))
        ].index
        return outlier_index


# remove_outlier function for removing the outliers
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[
        ~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))
    ]
    return df_without_outliers


# replace_with_thresholds function for replacing the outliers with the limits
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# replace the outliers with the limits
for col in num_cols:
    if col != "NObeyesdad_encoded":
        replace_with_thresholds(train, col)

# check the outliers again after replacing
for col in num_cols:
    print(f"##################### {col} #####################")
    print(check_outlier(train, col, q1=0.1, q3=0.9))


#############################################
# Missing Values
#############################################

# missing values
train.isnull().sum()

# test missing values
test.isnull().sum()

# we don't have any missing values in the train and test data
train.head()


cat_cols, num_cols, cat_but_car = grab_col_names(train)


#############################################
# Feature Extraction
#############################################
def feature_extraction(df):

    #############################################
    # Age to Gender Group
    # 0-17: young & male = youngmale
    df.loc[(df["Age"] < 18) & (df["Gender"] == "Male"), "NEW_GENDER_AGE_GROUP"] = (
        "youngmale"
    )

    # 18-50: adult & male = maturemale
    df.loc[
        (df["Gender"] == "Male") & (df["Age"] >= 18) & (df["Age"] <= 50),
        "NEW_GENDER_AGE_GROUP",
    ] = "maturemale"

    # 50+: senior & male = seniormale
    df.loc[(df["Gender"] == "Male") & (df["Age"] > 50), "NEW_GENDER_AGE_GROUP"] = (
        "seniormale"
    )

    # 0-17: young & femail = youngfemale
    df.loc[(df["Age"] < 18) & (df["Gender"] == "Female"), "NEW_GENDER_AGE_GROUP"] = (
        "youngfemale"
    )

    # 18-50: adult & femail = maturefemale
    df.loc[
        (df["Gender"] == "Female") & (df["Age"] >= 18) & (df["Age"] <= 50),
        "NEW_GENDER_AGE_GROUP",
    ] = "maturefemale"

    # 50+: senior & femail = seniorfemale
    df.loc[(df["Gender"] == "Female") & (df["Age"] > 50), "NEW_GENDER_AGE_GROUP"] = (
        "seniorfemale"
    )

    #############################################
    # Weight to Gender Group
    # 0-50: light & Male = lightmale
    df.loc[(df["Weight"] < 50) & (df["Gender"] == "Male"), "NEW_GEBDER_WEIGHT"] = (
        "lightmale"
    )

    # 50-90 : normal & male = normalmale
    df.loc[
        (df["Gender"] == "Male") & (df["Weight"] >= 50) & (df["Weight"] <= 90),
        "NEW_GEBDER_WEIGHT",
    ] = "normalmale"

    # 90+: heavy & male = heavymale
    df.loc[(df["Gender"] == "Male") & (df["Weight"] > 90), "NEW_GEBDER_WEIGHT"] = (
        "heavymale"
    )

    # 0-50: light & Female = lightfemale
    df.loc[(df["Weight"] < 50) & (df["Gender"] == "Female"), "NEW_GEBDER_WEIGHT"] = (
        "lightfemale"
    )

    # 50-90 : normal & female = normalfemale
    df.loc[
        (df["Gender"] == "Female") & (df["Weight"] >= 50) & (df["Weight"] <= 90),
        "NEW_GEBDER_WEIGHT",
    ] = "normalfemale"

    # 90+: heavy & female = heavyfemale
    df.loc[(df["Gender"] == "Female") & (df["Weight"] > 90), "NEW_GEBDER_WEIGHT"] = (
        "heavyfemale"
    )

    df.head()

    #############################################
    # family_history_with_overweight and Gender Group
    # family_history_with_overweight = yes & Gender == Male = yesmale
    df.loc[
        (df["family_history_with_overweight"] == "yes") & (df["Gender"] == "Male"),
        "NEW_FAMILY_HISTORY_WITH_Gender",
    ] = "yesfammilymale"

    # family_history_with_overweight = yes & Gender == Female = yesfemale
    df.loc[
        (df["family_history_with_overweight"] == "yes") & (df["Gender"] == "Female"),
        "NEW_FAMILY_HISTORY_WITH_Gender",
    ] = "yesfammilyfemale"

    # family_history_with_overweight = no & Gender == Male = nomale
    df.loc[
        (df["family_history_with_overweight"] == "no") & (df["Gender"] == "Male"),
        "NEW_FAMILY_HISTORY_WITH_Gender",
    ] = "nofammilymale"

    # family_history_with_overweight = no & Gender == Female = nofemale
    df.loc[
        (df["family_history_with_overweight"] == "no") & (df["Gender"] == "Female"),
        "NEW_FAMILY_HISTORY_WITH_Gender",
    ] = "nofammilyfemale"

    #############################################
    # FAVC and Weight interaction
    # FAVC = yes & Weight < 50 = yeslight
    df.loc[(df["FAVC"] == "yes") & (df["Weight"] < 50), "NEW_FAVC_WEIGHT"] = (
        "yescalorilight"
    )

    # FAVC = yes & Weight >= 50 & Weight <= 90 = yesnormal
    df.loc[
        (df["FAVC"] == "yes") & (df["Weight"] >= 50) & (df["Weight"] <= 90),
        "NEW_FAVC_WEIGHT",
    ] = "yescalorinormal"

    # FAVC = yes & Weight > 90 = yesheavy
    df.loc[(df["FAVC"] == "yes") & (df["Weight"] > 90), "NEW_FAVC_WEIGHT"] = (
        "yescaloriheavy"
    )

    # FAVC = no & Weight < 50 = nolight
    df.loc[(df["FAVC"] == "no") & (df["Weight"] < 50), "NEW_FAVC_WEIGHT"] = (
        "nocalorilight"
    )

    # FAVC = no & Weight >= 50 & Weight <= 90 = nonormal
    df.loc[
        (df["FAVC"] == "no") & (df["Weight"] >= 50) & (df["Weight"] <= 90),
        "NEW_FAVC_WEIGHT",
    ] = "nocalorinormal"

    # FAVC = no & Weight > 90 = noheavy
    df.loc[(df["FAVC"] == "no") & (df["Weight"] > 90), "NEW_FAVC_WEIGHT"] = (
        "nocaloriheavy"
    )

    #############################################
    # FAVC and Age interaction
    # FAVC = yes & Age < 18 = yesyoung
    df.loc[(df["FAVC"] == "yes") & (df["Age"] < 18), "NEW_FAVC_AGE"] = "yescaloriyoung"

    # FAVC = yes & Age >= 18 & Age <= 50 = yesmature
    df.loc[
        (df["FAVC"] == "yes") & (df["Age"] >= 18) & (df["Age"] <= 50),
        "NEW_FAVC_AGE",
    ] = "yescalorimature"

    # FAVC = yes & Age > 50 = yessenior
    df.loc[(df["FAVC"] == "yes") & (df["Age"] > 50), "NEW_FAVC_AGE"] = "yescalorisenior"

    # FAVC = no & Age < 18 = noyoung
    df.loc[(df["FAVC"] == "no") & (df["Age"] < 18), "NEW_FAVC_AGE"] = "nocaloriyoung"

    # FAVC = no & Age >= 18 & Age <= 50 = nomature
    df.loc[
        (df["FAVC"] == "no") & (df["Age"] >= 18) & (df["Age"] <= 50),
        "NEW_FAVC_AGE",
    ] = "nocalorimature"

    # FAVC = no & Age > 50 = nosenior
    df.loc[(df["FAVC"] == "no") & (df["Age"] > 50), "NEW_FAVC_AGE"] = "nocalorisenior"

    #############################################
    # Smoking and Age interaction
    # SC = yes & Age < 18 = scyoung
    df.loc[(df["SCC"] == "yes") & (df["Age"] < 18), "NEW_SCC_AGE"] = "smokeyoung"

    # SC = yes & Age >= 18 & Age <= 50 = scmature
    df.loc[
        (df["SCC"] == "yes") & (df["Age"] >= 18) & (df["Age"] <= 50),
        "NEW_SCC_AGE",
    ] = "smokemature"

    # SC = yes & Age > 50 = scsenior
    df.loc[(df["SCC"] == "yes") & (df["Age"] > 50), "NEW_SCC_AGE"] = "smokesenior"

    # SC = no & Age < 18 = nocyoung
    df.loc[(df["SCC"] == "no") & (df["Age"] < 18), "NEW_SCC_AGE"] = "nosokeyoung"

    # SC = no & Age >= 18 & Age <= 50 = nocmature
    df.loc[
        (df["SCC"] == "no") & (df["Age"] >= 18) & (df["Age"] <= 50),
        "NEW_SCC_AGE",
    ] = "nosokemature"

    # SC = no & Age > 50 = nocsenior
    df.loc[(df["SCC"] == "no") & (df["Age"] > 50), "NEW_SCC_AGE"] = "nosokesenior"


feature_extraction(df=train)


cat_cols, num_cols, cat_but_car = grab_col_names(train)

# drop unnecessary columns after feature extraction
drop_list = ["id", "NObeyesdad"]

train = train.drop(drop_list, axis=1)

train.head()

#############################################
# Endoing (One Hot Encoding)
############################################

cat_cols, num_cols, cat_but_car = grab_col_names(train)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first
    )
    return dataframe


train = one_hot_encoder(train, cat_cols, drop_first=True)

train.head()

cat_cols, num_cols, cat_but_car = grab_col_names(train)

# uppercase for column names
train.columns = [col.upper() for col in train.columns]

train.head()

#############################################
# Modelling (LightGBM)
#############################################

X = train.drop("NOBEYESDAD_ENCODED", axis=1)
y = train["NOBEYESDAD_ENCODED"]

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()


cv_results = cross_validate(
    lgbm_model,
    X,
    y,
    cv=10,
    scoring=["accuracy", "recall_macro", "roc_auc_ovr", "f1_micro"],
    n_jobs=-1,
)

cv_results = pd.DataFrame(cv_results)
cv_results

# grid search for hyperparameter tuning
# lgbm best parameters range for grid search
lgbm_params = {
    "n_estimators": [100, 200, 500, 1000],
    "subsample": [0.6, 0.8, 1.0],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.1, 0.01, 0.02, 0.05],
    "min_child_samples": [5, 10, 20],
}

# grid search for hyperparameter tuning
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(
    X, y
)

# best parameters for lgbm
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(
    X, y
)

cv_results = cross_validate(
    lgbm_final,
    X,
    y,
    cv=10,
    scoring=["accuracy", "recall_macro", "roc_auc_ovr", "f1_micro"],
    n_jobs=-1,
)


cv_results = pd.DataFrame(cv_results)
cv_results

cv_results["test_accuracy"].mean()
cv_results["test_recall_macro"].mean()
cv_results["test_roc_auc_ovr"].mean()
cv_results["test_f1_micro"].mean()


#############################################
# Plot importance
#############################################


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": features.columns}
    )
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(lgbm_final, X, num=20, save=False)


#############################################
# Validation Curve
#############################################


# val_curve_params function for validation curve
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        cv=cv,
    )

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color="b")

    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)


# validation curve for n_estimators
val_curve_params(
    lgbm_final,
    X,
    y,
    param_name="n_estimators",
    param_range=[10, 50, 200, 500],
    scoring="roc_auc_ovr",
    cv=5,
)
