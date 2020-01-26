import pandas as pd
import numpy as np
import pprint
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


def get_data(filepath):
    data = pd.read_csv(filepath)
    return data


def encoder_category(data, columns):

    for col in columns:
        encoder = OrdinalEncoder(dtype=np.int)
        data[col] = encoder.fit_transform(data[col].values.reshape(-1, 1)) + 1
    return data


def replace_in_col(data, col_name, col_replace):
    if col_name in data.columns:
        data[col_name] = data[col_name].replace(col_replace)


def rename_col(data):
    rename_list = ["Id", "Nom", "Hote_id", "Nom_hote", "Departement", "Quartier", 'Latitude', 'Longitude', "Type_chambre",
                   "Prix", "Minimum_nuit", "Nombre_avis", "Dernier_avis", "Avis_par_mois", "Nombre_list_hote", "Disponabilitie_365"]

    columns = list(data.columns)
    if len(columns) == len(rename_list):
        d = dict(zip(columns, rename_list))
        data.rename(columns=d, inplace=True, errors="raise")


def encodeHot_cat(data, to_encode):
    onehot = OneHotEncoder(dtype=np.int, sparse=True)
    nominals = pd.DataFrame(
        onehot.fit_transform(data[to_encode].values.reshape(-1, 1)).toarray(),
        columns=['Depart_' + county for county in data[to_encode].unique()])
    data = pd.concat([data, nominals.reindex(data.index)], axis=1)
    data.drop(columns=[to_encode], axis=1, inplace=True)
    return data


def scaling(data, to_scale):
    standard_vec = StandardScaler(with_mean=False)
    nominals = pd.DataFrame(
        np.round(standard_vec.fit_transform(
            data[to_scale].values.reshape(-1, 1)), 2),
        columns=[to_scale+"_scale"])
    data = pd.concat([data, nominals.reindex(data.index)], axis=1)
    data.drop(columns=[to_scale], axis=1, inplace=True)
    data.dropna()
    del(nominals)
    return data


def cleaning(data):
    print("=========== Start Cleaning ===========")
    missing_data = pd.DataFrame(
        round(data.isnull().sum()/data.shape[0]*100, 2), columns=["Pourcentage (%)"])
    print(missing_data.sort_values(by="Pourcentage (%)", ascending=False))
    print("\n")
    print("The pourcentage of missing data in our data is ", missing_data.max())
    print("====1/ Deleting duplicated values====")
    duplicated_vals = data.duplicated().sum()
    print("\t There are ", duplicated_vals)
    print("\t Deleting duplicated data")
    data.drop_duplicates(inplace=True)
    print("\t Deleting Done !!")
    print("====2/ Deleting unnecessary columns====")
    print("\t Deleting the following columns: Nom, Hote_id, Nom_hote,Dernier_avis")
    data.drop(columns=["Nom", "Hote_id", "Nom_hote",
                       "Dernier_avis"], axis=1, inplace=True)
    print("====3/ Replacing NaN with 0====")
    print("\t Replacing NaN in columns Nombre_avis and Avis_par_mois to 0")
    data["Nombre_avis"].fillna(0, inplace=True)
    data["Avis_par_mois"].fillna(0, inplace=True)
    print("===4/ Drop the rest of Nan====")
    data.dropna(how="any", inplace=True)
    print("\t Done!!")
    print("=========== End of Cleaning ===========")


def get_processed_data(file):
    df = get_data(file)
    rename_col(df)
    cleaning(df)
    df = encoder_category(df, ["Quartier", "Type_chambre"])
    df = encodeHot_cat(df, "Departement")
    df = scaling(df, "Disponabilitie_365")
    df = scaling(df, "Nombre_avis")

    df.to_csv('./data/processed/preprocessing_data.csv', index=False)
    return df


get_processed_data('./data/raw/AB_NYC_2019.csv')
