import pandas as pd
import numpy as np
import math
import scipy.stats as sct
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.options.display.max_colwidth = None

df_ = pd.read_csv("Hotel_Reviews.csv")
#df_ = pd.read_csv("Kaggle_Hotel/Hotel_Reviews.csv")
df= df_.copy()
df = df.drop(columns=["Review_Date", "Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given","lat","lng"],axis=1)

scaler = MinMaxScaler(feature_range=(0, 5))
df['Reviewer_Score'] = scaler.fit_transform(df[['Reviewer_Score']])
df['Average_Score'] = scaler.fit_transform(df[['Average_Score']])

df["days_since_review"] = (df["days_since_review"]
                           .str.replace(" days", "")
                           .str.replace(" day", "")
                           .astype(int))



df["summary_min"] = df.groupby("Hotel_Name")["days_since_review"].transform("min")
df["summary_q1"] = df.groupby("Hotel_Name")["days_since_review"].transform(lambda x: x.quantile(0.25))
df["summary_q2"] = df.groupby("Hotel_Name")["days_since_review"].transform("median")
df["summary_q3"] = df.groupby("Hotel_Name")["days_since_review"].transform(lambda x: x.quantile(0.75))
df["summary_max"] = df.groupby("Hotel_Name")["days_since_review"].transform("max")

def weighted_average_by_time(df_group):
    q1, q2, q3 = df_group["summary_q1"].iloc[0], df_group["summary_q2"].iloc[0], df_group["summary_q3"].iloc[0]

    weighted_avg = (
            df_group.loc[df_group["days_since_review"] <= q1, "Reviewer_Score"].mean() * 0.28 +
            df_group.loc[(df_group["days_since_review"] > q1) & (
                        df_group["days_since_review"] <= q2), "Reviewer_Score"].mean() * 0.26 +
            df_group.loc[(df_group["days_since_review"] > q2) & (
                        df_group["days_since_review"] <= q3), "Reviewer_Score"].mean() * 0.24 +
            df_group.loc[df_group["days_since_review"] > q3, "Reviewer_Score"].mean() * 0.22
    )
    return weighted_avg

hotel_weighted_averages = df.groupby("Hotel_Name").apply(weighted_average_by_time).reset_index(name="Weighted_Average_Score")
df = df.merge(hotel_weighted_averages, on="Hotel_Name", how="left")

df["Reviewer_Nationality"] = df["Reviewer_Nationality"].str.strip().str.strip("'\"")
unique_countries = [country.strip().strip("'\"") for country in df["Reviewer_Nationality"].unique().tolist()]

def waighted_national(df_group, nation):
    selected_nation_score = df_group.loc[df_group["Reviewer_Nationality"].isin(nation), "Reviewer_Score"].mean()
    other_nations_score = df_group.loc[~df_group["Reviewer_Nationality"].isin(nation), "Reviewer_Score"].mean()
    weighted_avg = (selected_nation_score * 0.60) + (other_nations_score * 0.40)

    return weighted_avg

hotel_weighted_averages_by_nation = df.groupby("Hotel_Name").apply(waighted_national, nation=['United Kingdom']).reset_index(
    name="Weighted_Average_Score_By_Nation")

df = df.merge(hotel_weighted_averages_by_nation, on="Hotel_Name", how="left")

bins = [0, 1, 2, 3, 4, 6]
labels = [1, 2, 3, 4, 5]

df['Final_star'] = pd.cut(df['Reviewer_Score'], bins=bins, labels=labels, right=False)
df = pd.get_dummies(df, columns=['Final_star'], prefix='Star', dtype = "int64")

df['Negative_Review'] = df['Negative_Review'].replace([
    'Non', 'No', 'None', 'NA', 'Nothing', 'Nothin', 'Nothinb', 'Nothibg', 'Nothings', 'NOTHNG',
    'Nothink', 'Nothig', 'Nothingseccess', 'Nothimg', 'Nothinh', 'Nothg', 'Nothing7', 'Nothinge',
    'Notheng', 'Nothhing', 'Nothong', 'Nothjng', 'Nothingefs', 'Nothjnng', 'Nothingg', 'Nothung',
    'Nothinf', 'Nothingthe', 'NothingA', 'Nothiing', 'NOTHINGGGGGG', 'nothging', 'Nothting',
    'nothingone', 'Nothubg', 'Nothjg', 'NothingGra', 'Noththing', 'Nothening', 'NothingGreat',
    'Nothingreally', 'Nothlng'
], 'No Negative')

df['Positive_Review'] = df['Positive_Review'].replace([
    'nothing', 'Nothink', 'Nothin', 'Nothings', 'Not', 'Notjing', 'Nothng', 'NA', 'Na', 'nathing'
], 'No Positive')

df['Negative_Flag'] = df['Negative_Review'].apply(lambda x: 0 if x == 'No Negative' else 1)
df['Positive_Flag'] = df['Positive_Review'].apply(lambda x: 0 if x == 'No Positive' else 1)
df["Total_count"] = df["Negative_Flag"] + df["Positive_Flag"]
df = df[df['Total_count'] != 0]

df_grouped = (df.groupby("Hotel_Name").agg(
    Negative_Review=("Negative_Flag", "sum"),
    Positive_Review=("Positive_Flag", "sum"))
    .sort_values(by="Positive_Review", ascending=False)
).reset_index()

df_grouped['Positive_Ratio'] = df_grouped['Positive_Review'] / (df_grouped['Negative_Review'] + df_grouped['Positive_Review'])
df = df.merge(df_grouped, on="Hotel_Name", how="left")

Final_df = (
    df.groupby('Hotel_Name').agg(
        Hotel_Address=('Hotel_Address', 'first'),
        Weighted_Average_Score=('Weighted_Average_Score', 'first'),
        Weighted_Average_Score_By_Nation=('Weighted_Average_Score_By_Nation', 'first'),
        Total_Number_of_Reviews=('Total_Number_of_Reviews', 'first'),
        Star_Rating=('Reviewer_Score', 'mean'),
        Star_1=('Star_1', 'sum'),
        Star_2=('Star_2', 'sum'),
        Star_3=('Star_3', 'sum'),
        Star_4=('Star_4', 'sum'),
        Star_5=('Star_5', 'sum'),
        Negative_Review=("Negative_Flag", "sum"),
        Positive_Review=("Positive_Flag", "sum"),
        Positive_Ratio=("Positive_Ratio", 'first')
    )
    .sort_values(by="Weighted_Average_Score", ascending=False)
    .reset_index()
)

Final_df["Comment_Count_Score"] = Final_df["Total_Number_of_Reviews"] - Final_df["Negative_Review"] + Final_df["Positive_Review"]
Final_df["Log_Comment_Count_Score"] = np.log1p(Final_df["Comment_Count_Score"])
scaler = MinMaxScaler(feature_range=(1, 5))
Final_df["Comment_Count_Score"] = scaler.fit_transform(Final_df[["Log_Comment_Count_Score"]])

def weighted_sorting_score(dataframe, w1=30, w2=70,):
    return (dataframe["Comment_Count_Score"] * w1 / 100 +
            dataframe["Weighted_Average_Score"] * w2 / 100)

Final_df["weighted_sorting_score"] = weighted_sorting_score(Final_df)

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = sct.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

Final_df["bar_score"] = Final_df.apply(lambda x: bayesian_average_rating(x[["Star_1",
                                                                "Star_2",
                                                                "Star_3",
                                                                "Star_4",
                                                                "Star_5"]]), axis=1)

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["Star_1",
                                                                     "Star_2",
                                                                     "Star_3",
                                                                     "Star_4",
                                                                     "Star_5"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return (bar_score*bar_w/100) + (wss_score*wss_w/100)

Final_df["hybrid_sorting_score"] = hybrid_sorting_score(Final_df)

Final_df['Country_name'] = Final_df['Hotel_Address'].apply(    #counrty olarak aldıktan sonra aslıda otel adresinide silebiliriz sadece ülke kalır
    lambda x: 'United Kingdom' if 'United Kingdom' in x else x.split()[-1])

def recommend_hotels(dataframe, country_name, nationality=None):
    # Ülkeye göre filtreleme
    filtered_hotels = dataframe[dataframe['Country_name'] == country_name]

    # Eğer bir milliyet girildiyse, Weighted_Average_Score_By_Nation kullanarak sıralama yap
    if nationality:
        recommended_hotels = filtered_hotels.sort_values(by="Weighted_Average_Score_By_Nation", ascending=False).head(10)
        print(f"{country_name} ülkesindeki '{nationality}' milliyeti için en iyi 10 otel:")
    else:
        # Milliyet girilmezse, hybrid_sorting_score'a göre sıralama yap
        recommended_hotels = filtered_hotels.sort_values(by="hybrid_sorting_score", ascending=False).head(10)
        print(f"{country_name} ülkesindeki en iyi 10 otel:")

    # Sonuçları ekrana yazdır
    print(recommended_hotels[['Hotel_Name', 'hybrid_sorting_score', 'Weighted_Average_Score_By_Nation']])

# Kullanım örneği:
recommend_hotels(Final_df, "France")
recommend_hotels(Final_df, "France", nationality="United Kingdom")



#st.logo(
 #   "Images/Logo2.png",
 #   icon_image="Images/logo.png",
 #   size="large")





















