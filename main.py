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

train["NObeyesdad"].value_counts()
train["NObeyesdad_encoded"].value_counts()

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


import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", None)  # Increase the width of the output

for col in num_cols:
    # wrrite the variable name
    print(f"########################### {col} ###########################")
    num_summary(train, col, plot=True)


train.head()


# target variable analysis for categorical variables
def target_summary_with_cat(dataframe, target, categorical_col, graph=False):
    # label encoding for NObeyesdad column for visualization
    label_encoder = LabelEncoder()
    dataframe["NObeyesdad_encoded"] = label_encoder.fit_transform(
        dataframe["NObeyesdad"]
    )

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


train.shape

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


# outlier analysis for numerical variables
for col in num_cols:
    print(f"##################### {col} #####################")
    print(col, check_outlier(train, col, q1=0.05, q3=0.95))
