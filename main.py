import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
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
    Underweight: Less than 18.5
    Normal: 18.5 to 24.9
    Overweight: 25.0 to 29.9
    Obesity I: 30.0 to 34.9
    Obesity II: 35.0 to 39.9
    Obesity III: Higher than 40
"""

# importin data from csv files
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
sample_submission = pd.read_csv("./data/sample_submission.csv")


# checking data with check_train function
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


def target_summary_with_cat(dataframe, target, categorical_col):
    # check the target variable data type and convert to int by using label encoder
    if dataframe[target].dtypes == "object":
        dataframe[target] = LabelEncoder().fit_transform(dataframe[target])

    # calculate the percentage of each category
    target_counts = (
        dataframe.groupby(categorical_col)[target]
        .value_counts(normalize=True)
        .rename("percentage")
        .mul(100)
        .reset_index()
    )

    print(target_counts, end="\n\n")


for col in cat_cols:
    target_summary_with_cat(train, "NObeyesdad", col)



def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")


for col in num_cols:
    target_summary_with_num(train, "NObeyesdad", col)

# deneme 