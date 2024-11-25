# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt


def data_exploration(df):
    """
    Function to provide a comprehensive overview of a pandas DataFrame.
    Intended for use in Jupyter Notebook environments.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to explore.
    """
    # General overview of the dataset
    print(f"The dataset has a total of {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"The number of duplicate rows is {df.duplicated().sum()}.")
    
    print("\n----------\n")
    
    # Null values and their percentages
    print("Columns with null values and their percentages:")
    null_percentages = (df.isnull().sum() / df.shape[0]) * 100
    null_percentages = null_percentages[null_percentages > 0].sort_values(ascending=False)
    if not null_percentages.empty:
        display(null_percentages)
    else:
        print("No null values found.")
    
    print("\n----------\n")
    
    # Main statistics for numerical variables
    print("Main statistics of numerical variables:")
    display(df.describe().T)
    
    print("\n----------\n")
    
    # Main statistics for categorical variables
    print("Main statistics of categorical variables:")
    display(df.describe(include=["O", "category"]).T)
    
    print("\n----------\n")
    
    # DataFrame information
    print("Features of the DataFrame:")
    df.info()


class Visualizer:
    """A class for performing various data visualization tasks on a pandas DataFrame."""

    def __init__(self, df):
        """
        Initializes the instance with a DataFrame.

        Parameters:
        - df (DataFrame): The DataFrame to be used in the instance.
        """
        self.df = df


    def separate_dataframes(self):
        """
        Separates the DataFrame into two based on data types.

        Returns:
        - (tuple): A tuple containing:
        - DataFrame: A DataFrame with numerical columns.
        - DataFrame: A DataFrame with object (string) columns.
        """
        return self.df.select_dtypes(include=np.number), self.df.select_dtypes(include="O")
    

    def plot_all_numerical(self, color="grey", size=(15, 5)):
        """
        Plots the distribution of numerical variables in the DataFrame.

        Parameters:
        - color (str, optional): The color of the histogram bars. Default is "grey".
        - size (tuple, optional): The size of the plot as (width, height). Default is (15, 5).

        Returns:
        - None
        """

        cols = self.separate_dataframes()[0].columns
        _, axes = plt.subplots(nrows = 2, ncols = math.ceil(len(cols)/2), figsize=size)
        axes = axes.flat

        for i, col in enumerate(cols):
            sns.histplot(x=col, data=self.df, ax=axes[i], color=color, bins=20)

        plt.suptitle("Distribution of numerical variables")
        plt.tight_layout()


    def plot_numeric_variable(self, var, color="grey", size=(10, 6)):
        """
        Plots the distribution of a specified numeric variable from the DataFrame.

        Parameters:
        - var (str): The name of the variable to plot. Must exist in the DataFrame.
        - color (str, optional): The color of the histogram bars. Default is "grey".
        - size (tuple, optional): The size of the plot as (width, height). Default is (10, 6).

        Raises:
        - ValueError: If the specified variable does not exist in the DataFrame.

        Returns:
        - None
        """

        # Check if the variable exists in the DataFrame
        if var not in self.df.columns:
            raise ValueError(f"The variable '{var}' does not exist in the DataFrame.")

        # Plot the distribution of the variable.
        plt.figure(figsize=size)
        sns.histplot(x=var, data=self.df, color=color, bins=20)

        plt.title(f"'{var}' distribution")
        plt.xlabel(var)
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


    def plot_all_categorical(self, color="grey", size=(40, 10)):
        """
        Plots the distribution of categorical variables in the DataFrame.

        Parameters:
        - color (str, optional): The color of the bars in the count plots. Default is "grey".
        - size (tuple, optional): The size of the plot as (width, height). Default is (40, 10).

        Returns:
        - None
        """

        df_cat = self.separate_dataframes()[1]

        _, axes = plt.subplots(2, math.ceil(len(df_cat.columns) / 2), figsize=size)
        axes = axes.flat

        for i, col in enumerate(df_cat.columns):
            sns.countplot(x=col, data=self.df, order=self.df[col].value_counts().index, ax=axes[i], color=color)
            axes[i].tick_params(rotation=90)
            axes[i].set_title(col)
            axes[i].set(xlabel=None)

        plt.tight_layout()
        plt.suptitle("Distribution of categorical variables")


    def plot_categorical_variable(self, var, color="grey", size=(10, 6)):
        """
        Plots the distribution of a specified categorical variable from the DataFrame.

        Parameters:
        - var (str): The name of the variable to plot. Must exist in the DataFrame.
        - color (str, optional): The color of the bars in the count plot. Default is "grey".
        - size (tuple, optional): The size of the plot as (width, height). Default is (10, 6).

        Raises:
        - ValueError: If the specified variable does not exist in the DataFrame.

        Returns:
        - None
        """

        # Check if the variable exists in the DataFrame.
        if var not in self.df.columns:
            raise ValueError(f"The variable '{var}' does not exist in the DataFrame.")

        # Plot the distribution of the variable.
        plt.figure(figsize=size)
        sns.countplot(
            x=var, 
            data=self.df, 
            order=self.df[var].value_counts().index,
            color=color
        )

        plt.title(f"Distribution of the categorical variable '{var}'.")
        plt.xlabel(var)
        plt.ylabel("Frequency")
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


    def plot_categorical_comparison(self, cat_var1, cat_var2, size=(12, 6), palette="rocket"):
        """
        Plots a comparison of two categorical variables from the DataFrame.

        Parameters:
        - cat_var1 (str): The name of the first categorical variable for the x-axis. Must exist in the DataFrame.
        - cat_var2 (str): The name of the second categorical variable for grouping (hue). Must exist in the DataFrame.
        - size (tuple, optional): The size of the plot as (width, height). Default is (12, 6).
        - palette (str, optional): The color palette for the plot. Default is "rocket".

        Raises:
        - ValueError: If either of the specified variables does not exist in the DataFrame.

        Returns:
        - None
        """

        # Verify if the variables exist in the DataFrame
        if cat_var1 not in self.df.columns:
            raise ValueError(f"The variable '{cat_var1}' does not exist in the DataFrame.")

        if cat_var2 not in self.df.columns:
            raise ValueError(f"The variable '{cat_var2}' does not exist in the DataFrame.")

        plt.figure(figsize=size)
        sns.countplot(data=self.df, x=cat_var1, hue=cat_var2, palette=palette)

        plt.title(f"{cat_var1} vs {cat_var2}")
        plt.xlabel(cat_var1)
        plt.ylabel("Frequency")
        plt.legend(title=cat_var2, loc='lower right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


    def plot_relationship(self, target_variable, size=(40, 12), color="grey"):
        """
        Plots the relationship between the target variable and other variables in the DataFrame.

        Parameters:
        - target_variable (str): The name of the target variable to analyze relationships with. Must exist in the DataFrame.
        - size (tuple, optional): The size of the plot as (width, height). Default is (40, 12).
        - color (str, optional): The color of the plots. Default is "grey".

        Returns:
        - None
        """

        df_num = self.separate_dataframes()[0].columns

        fig, axes = plt.subplots(3, int(len(self.df.columns) / 3), figsize=size)
        axes = axes.flat

        for i, col in enumerate(self.df.columns):

            if col == target_variable:
                fig.delaxes(axes[i])

            elif col in df_num:
                sns.scatterplot(x=target_variable, 
                                y=col, 
                                data=self.df, 
                                color=color, 
                                ax=axes[i])
                axes[i].set_title(col)
                axes[i].set(xlabel=None)
            else:
                sns.barplot(x=col, y=target_variable, data=self.df, ax=axes[i], color=color)
                axes[i].tick_params(rotation=90)
                axes[i].set_title(col)
                axes[i].set(xlabel=None)

        plt.tight_layout()
    

    def temporal_analysis(self, target_variable, temporal_variable, interval, color="black", order=None):
        """
        Performs a temporal analysis of the target variable based on a specified time interval.

        Parameters:
        - target_variable (str): The name of the variable to analyze. Must exist in the DataFrame.
        - temporal_variable (str): The name of the temporal variable to group by. Must exist in the DataFrame.
        - interval (str): The time interval for grouping. Acceptable values are "hour", "day", "month", or "year".
        - color (str, optional): The color of the line plot. Default is "black".
        - order (list, optional): A custom order for the intervals. Useful for non-chronological orderings (e.g., month names).

        Raises:
        - ValueError: If the specified variables do not exist or the interval is unrecognized.

        Returns:
        - None
        """

        # Check that the columns exist.
        if target_variable not in self.df.columns or temporal_variable not in self.df.columns:
            raise ValueError("One or both specified variables do not exist in the DataFrame.")

        # Convert to datetime
        self.df[temporal_variable] = pd.to_datetime(self.df[temporal_variable])

        df = self.df.copy()

        # Group the data according to the specified interval.
        if interval == "hour":
            df["interval"] = self.df[temporal_variable].dt.hour
        elif interval == "day":
            df["interval"] = self.df[temporal_variable].dt.day
        elif interval == "month":
            df["interval"] = self.df[temporal_variable].dt.month_name()
        elif interval == "year":
            df["interval"] = self.df[temporal_variable].dt.year
        else:
            raise ValueError("Unrecognized interval. Use 'hour', 'day', 'month', or 'year'.")

        # If an order is provided, apply categorization.
        if order:
            df["interval"] = pd.Categorical(self.df["interval"], categories=order, ordered=True)

        # Creata graphics
        plt.figure(figsize=(15, 5))

        sns.lineplot(x="interval", 
                    y=target_variable, 
                    data=df, 
                    color=color)

        # Add a line for the mean of the response variable.
        mean_value = self.df[target_variable].mean()
        plt.axhline(mean_value, color="green", linestyle="--", label=f"{target_variable} mean values")

        plt.xlabel(interval.capitalize())
        plt.ylabel(target_variable)
        plt.title(f"Temporal analysis of {target_variable} by {interval}")
        plt.legend()
        sns.despine()
        plt.tight_layout()
        plt.show()


    def outliers_detection(self, color = "grey"):
        """
        Detects and visualizes outliers in the numerical variables of the DataFrame using box plots.

        Parameters:
        - color (str, optional): The color of the box plots. Default is "grey".

        Returns:
        - None
        """

        cols = self.separate_dataframes()[0].columns

        fig, axes = plt.subplots(2, ncols = math.ceil(len(cols)/2), figsize=(15,5))
        axes = axes.flat

        for i, col in enumerate(cols):

            sns.boxplot(x=col,
                        data=self.df, 
                        ax=axes[i], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})

        if len(cols) % 2 != 0:
            fig.delaxes(axes[-1])

        plt.tight_layout()


    def correlation(self, size=(7, 5)):
        """
        Displays a heatmap showing the correlation between numerical variables in the DataFrame.

        Parameters:
        - size (tuple, optional): The size of the plot as (width, height). Default is (7, 5).

        Returns:
        - None
        """

        plt.figure(figsize=size)

        mask = np.triu(np.ones_like(self.df.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.df.corr(numeric_only = True), 
                    annot = True,
                    vmin=-1,
                    vmax=1,
                    cmap="viridis",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
        

    def grouped_boxplot(self, numeric_column, categorical_column, title="Boxplot", xlabel=None, ylabel=None):
        """
        Creates a grouped boxplot to visualize the distribution of a numeric variable across the levels of a categorical variable.

        Parameters:
        - numeric_column (str): The name of the numeric column to plot. Must exist in the DataFrame.
        - categorical_column (str): The name of the categorical column to group by. Must exist in the DataFrame.
        - title (str, optional): The title of the plot. Default is "Boxplot".
        - xlabel (str, optional): Custom label for the x-axis. Defaults to the name of the categorical column if not provided.
        - ylabel (str, optional): Custom label for the y-axis. Defaults to the name of the numeric column if not provided.

        Raises:
        - ValueError: If the specified columns do not exist in the DataFrame.

        Returns:
        - None
        """
        # Check that the columns exist.
        if numeric_column not in self.df.columns or categorical_column not in self.df.columns:
            raise ValueError("One or both specified columns do not exist in the DataFrame.")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=categorical_column, y=numeric_column, data=self.df, palette="mako")

        plt.title(title, fontsize=16)
        plt.xlabel(xlabel if xlabel else categorical_column, fontsize=14)
        plt.ylabel(ylabel if ylabel else numeric_column, fontsize=14)

        plt.tight_layout()
        plt.show()


class Desbalanceo:

    def __init__(self, dataframe, variable_dependiente):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente

    def visualizar_clase(self, color="orange", edgecolor="black"):
        plt.figure(figsize=(8, 5))  # para cambiar el tamaño de la figura
        fig = sns.countplot(data=self.dataframe, 
                            x=self.variable_dependiente,  
                            color=color,  
                            edgecolor=edgecolor)
        fig.set(xticklabels=["No", "Yes"])
        plt.show()


    def balancear_clases_pandas(self, metodo):

        # Contar las muestras por clase
        contar_clases = self.dataframe[self.variable_dependiente].value_counts()
        clase_mayoritaria = contar_clases.idxmax()
        clase_minoritaria = contar_clases.idxmin()

        # Separar las clases
        df_mayoritaria = self.dataframe[self.dataframe[self.variable_dependiente] == clase_mayoritaria]
        df_minoritaria = self.dataframe[self.dataframe[self.variable_dependiente] == clase_minoritaria]

        if metodo == "downsampling":
            # Submuestrear la clase mayoritaria
            df_majority_downsampled = df_mayoritaria.sample(contar_clases[clase_minoritaria], random_state=42)
            # Combinar los subconjuntos
            df_balanced = pd.concat([df_majority_downsampled, df_minoritaria])

        elif metodo == "upsampling":
            # Sobremuestrear la clase minoritaria
            df_minority_upsampled = df_minoritaria.sample(contar_clases[clase_mayoritaria], replace=True, random_state=42)
            # Combinar los subconjuntos
            df_balanced = pd.concat([df_mayoritaria, df_minority_upsampled])

        else:
            raise ValueError("Método no reconocido. Use 'downsampling' o 'upsampling'.")

        return df_balanced

    def balancear_clases_imblearn(self, metodo):

        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        if metodo == "RandomOverSampler":
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)

        elif metodo == "RandomUnderSampler":
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)

        else:
            raise ValueError("Método no reconocido. Use 'RandomOverSampler' o 'RandomUnderSampler'.")

        df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled
    
    def balancear_clases_smote(self):
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled

    def balancear_clases_smote_tomek(self):
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled

