import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


def get_data(filepath):
    """Read and generate the data and get a DataFrame.
        Args:
            filepath (str): File of the raw .csv data .
        Returns:
            data (pandas.core.frame.DataFrame): Data generating in csv file.
    """
    data = pd.read_csv(filepath)
    return data


def encoder_category(data, columns):
    """Encode the categorical type in a specific column to numbers.
            >> ['Male', 'Female] --> [1, 2]
        Args:
            data (pandas.core.frame.DataFrame): Data from the dataSet.
            columns (str): the column to encode
        Returns:
            data (pandas.core.frame.DataFrame): Data after encoding a specific colum.
    """
    encoder = OrdinalEncoder(dtype=np.int)
    for col in columns:
        data[col] = encoder.fit_transform(data[col].values.reshape(-1, 1)) + 1

    return data


def replace_in_col(data, col_name, col_replace):
    """Replace the data inside a column.
        Args:
            data (pandas.core.frame.DataFrame): Data from the dataSet.
            col_name (str): the name of column.
            col_replace (dict): the dictionary of all the specific names we want to replace.
    """
    if col_name in data.columns:
        data[col_name] = data[col_name].replace(col_replace)


def rename_col(data):
    """Rename the columns to french.
        Args:
            data (pandas.core.frame.DataFrame): Data from the dataSet.
    """
    rename_list = ["Id", "Nom", "Hote_id", "Nom_hote", "Departement", "Quartier", 'Latitude', 'Longitude', "Type_chambre",
                   "Prix", "Minimum_nuit", "Nombre_avis", "Dernier_avis", "Avis_par_mois", "Nombre_list_hote", "Disponabilitie_365"]

    columns = list(data.columns)
    if len(columns) == len(rename_list):
        d = dict(zip(columns, rename_list))
        data.rename(columns=d, inplace=True, errors="raise")


def encodeHot_cat(data, col_encode):
    """Encode the categorical column to seperate columns.
        Args:
            data (pandas.core.frame.DataFrame): Data from the dataSet.
            col_encode (str): the column to encode
        Returns:
            data (pandas.core.frame.DataFrame): Data after deleting the encoded column and adding new columns.
    """
    onehot = OneHotEncoder(dtype=np.int, sparse=True)
    nominals = pd.DataFrame(
        onehot.fit_transform(data[col_encode].values.reshape(-1, 1)).toarray(),
        columns=['Depart_' + county for county in data[col_encode].unique()])
    data = pd.concat([data, nominals.reindex(data.index)], axis=1)
    data.drop(columns=[col_encode], axis=1, inplace=True)
    del(nominals)

    return data


def scaling(data, col_scale):
    """Scaling a column in a standard way.
        Args:
            data (pandas.core.frame.DataFrame): Data from the dataSet.
            col_scale (str): the column to scale
        Returns:
            data (pandas.core.frame.DataFrame): Data after deleting the column and adding the scaled one.
        Note:
            for more detail on scaling: http://benalexkeen.com/feature-scaling-with-scikit-learn/
    """
    # Add the standardScaler
    standard_vec = StandardScaler(with_mean=False)

    # Round the data to 2 after point
    nominals = pd.DataFrame(
        np.round(standard_vec.fit_transform(
            data[col_scale].values.reshape(-1, 1)), 2),
        columns=[col_scale + "_scale"])
    data = pd.concat([data, nominals.reindex(data.index)], axis=1)
    data.drop(columns=[col_scale], axis=1, inplace=True)

    # Deleting the unused data
    del(nominals)

    return data


def cleaning(data):
    """Cleaning data to remove or replace the NaN values.
        Args:
            data (pandas.core.frame.DataFrame): Data from the dataSet.
    """
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
    print("\n")

def get_processed_data(file):
    """Processing the csv file to get cleaned and has all the numerical 
        columns and save it to another file.
        Args:
            file (str): Data from the dataSet.
        Returns:
            data (pandas.core.frame.DataFrame): Data after deleting the column and adding the scaled one.
    """
    print("I- Getting the data from",file.split("/")[-1])
    print("\n")
    df = get_data(file)
    
    print("II- Rename the data")
    print("\n")
    rename_col(df)
    
    print("III- Cleaning the data")
    print("\n")
    cleaning(df)
    
    print("IV- Encoding the data")
    print("\n")
    df = encoder_category(df, ["Quartier", "Type_chambre"])
    df = encodeHot_cat(df, "Departement")
    
    print("IV- Scaling the data")
    print("\n")
    df = scaling(df, "Disponabilitie_365")
    df = scaling(df, "Nombre_avis")

    print("IV- Saving the data to './data/processed/preprocessing_data.csv'")
    print("\n")
    df.to_csv('./data/processed/preprocessing_data.csv', index=False)
    
    return df


get_processed_data('./data/raw/AB_NYC_2019.csv')
