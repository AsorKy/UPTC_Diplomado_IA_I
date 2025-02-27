#-------------------------------------------------------------------------------
#------------------------------ Dependencies -----------------------------------
#-------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import missingno
import statsmodels.api as sm
from sklearn.impute import KNNImputer
import scipy.stats as sts
from scipy.stats import ks_2samp
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PolynomialFeatures

#-------------------------------------------------------------------------------
#------------------------------ Histogram and KDE ------------------------------
#-------------------------------------------------------------------------------

def histograma_frecuencias(data, parametro, bins, title, x_label='data (X)', y_label='Frecuencia', kde=True, data_label=None, color='blue'):
    
  plt.figure(figsize=(8, 4))
  sns.histplot(data, bins=bins, kde=kde, color=color, label=data_label)
  plt.axvline(parametro, color=color, linestyle='dashed', linewidth=2)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  plt.grid(linestyle='--', alpha=0.7)
  plt.show()
  
#--------------------------------------------------------------------------------
#--------------------------------- Histogram-Box --------------------------------
#--------------------------------------------------------------------------------

def histogram_box(dataframe, variable):
  df = dataframe
  fig, ax = plt.subplots(1,2,figsize=(14,5))
  # histogram_box plot
  sns.histplot(data=df, x=str(variable), kde=True, ax=ax[0])
  sns.boxplot(data=df, x=str(variable))
  # mean
  mean = df[str(variable)].mean()
  ax[0].axvline(mean, color='red')
  # stats-box
  variance = round(df[str(variable)].var(),2)
  kurtosis = round(df[str(variable)].kurt(),2)
  skewness = round(df[str(variable)].skew(),2)

  textstr = '\n'.join( (r'mean=%.2f' % (mean, ), r'Variance=%.2f' % (variance, ), r'Kurtosis=%.2f' % (kurtosis, ), r'Skewness=%.2f' % (skewness, )) )
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
  ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
  ax[0].set_title(f'Histogram of {variable}')
  ax[0].set_xlabel(f'{variable}')
  ax[0].set_ylabel("Frecuency")
  ax[0].grid(True)
  ax[1].set_title(f'Boxplot of {variable}')
  ax[1].set_xlabel(f'{variable}')
  ax[1].set_ylabel("Frecuencias")
  ax[1].grid(True)

  plt.subplots_adjust(hspace=0.5)
  plt.show()


#-------------------------------------------------------------------------------
#------------------- Histogram and Central Tendency Metrics --------------------
#-------------------------------------------------------------------------------

def histograma_frecuencias_centrales(data,variable,bins,title,x_label='data (X)', y_label='Frecuencia', kde=True):
  plt.figure(figsize=(8,4))
  sns.histplot(data = data, bins=bins, x=variable, kde=True, color='blue', label='data')
  plt.axvline( x = data[variable].mean(), color = 'red', linestyle = 'dashed', label='mean')
  plt.axvline( x = data[variable].median(), color = 'green', linestyle = 'dashed', label='median')
  plt.axvline( x = data[variable].mode()[0],color = 'black', linestyle = 'dashed', label='mode')
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.grid(linestyle='--', alpha=0.7)
  plt.legend()
  plt.show()


#-------------------------------------------------------------------------------
#------------------------ Histogram and Scatter plot ---------------------------
#-------------------------------------------------------------------------------


def histograma_dispersion(data,variable,bins,title, title_box,x_label='data (X)', y_label='Frecuencia', kde=True):
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    sns.histplot(data = data, x=variable, bins=bins, kde=True, color='blue', ax = ax[0])
    sns.boxplot( data = data, x=variable, ax = ax[1])
    
    mean = data[variable].mean()
    q1 = data[variable].quantile(0.25)
    q2 = data[variable].quantile(0.5)
    q3 = data[variable].quantile(0.75)
    low_lim = q1 - 1.5*(q3-q1)
    up_lim  = q3 + 1.5*(q3-q1)
    
    sigma = data[variable].std()
    q3_sigma = mean + sigma
    q1_sigma = mean - sigma

    ax[0].axvline(x = low_lim, color='black', linestyle='dashed', label='low limit(IQR)')
    ax[0].axvline(x = up_lim, color='black', linestyle='dashed', label='upper limit(IQR)')
    ax[0].axvline(x = data[variable].min(), color='red', linestyle='dashed', label='x_min')

    ax[0].axvline(x = q1, color='green', linestyle='dashed', label='Q1')
    ax[0].axvline(x = q2, color='blue', linestyle='dashed', label='Q2')
    ax[0].axvline(x = q3, color='green', linestyle='dashed', label='Q3')
    
    ax[1].axvline(x = low_lim, color='black', linestyle='dashed', label='low limit(IQR)')
    ax[1].axvline(x = up_lim, color='black', linestyle='dashed', label='upper limit(IQR)')
    ax[1].axvline(x = data[variable].min(), color='red', linestyle='dashed', label='x_min')

    ax[1].axvline(x = q1_sigma, color='green', linestyle='dashed', label='media - sigma')
    ax[1].axvline(x = q1, color='purple', linestyle='dashed', label='Q1')
    ax[1].axvline(x = q2, color='blue', linestyle='dashed', label='Q2')
    ax[1].axvline(x = q3_sigma, color='green', linestyle='dashed', label='media + sigma')
    ax[1].axvline(x = q3, color='purple', linestyle='dashed', label='Q3')

    ax[0].set_title(title)
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)
    ax[0].legend()
    ax[1].legend()
    ax[1].set_title(title_box)
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_label)
    ax[0].grid(linestyle='--', alpha=0.7)
    ax[1].grid(linestyle='--', alpha=0.7)
    plt.show()
    
#-------------------------------------------------------------------------------
#--------------------- Imputation Scatter Plot Comparison ----------------------
#-------------------------------------------------------------------------------

def imputation_scatter_plot(data:pd.DataFrame, df_imputed:pd.DataFrame, variables, title='Imputation Scatter plot of '):
    null_indices = data[data.isnull().any(axis=1)].index
    df_imputed = df_imputed.iloc[null_indices]
    plt.scatter(data[variables[0]], data[variables[1]], color='blue', alpha=0.7, label='Original Data')
    plt.scatter(df_imputed[variables[0]], df_imputed[variables[1]], color='orange', alpha=0.7, label='Imputed Data')
    plt.title(f'{title} {variables[0]} vd {variables[1]}')    
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    
#-------------------------------------------------------------------------------
#--------------------------- Anscombe Dataset ----------------------------------
#-------------------------------------------------------------------------------


def Anscombe():
    # Cuarteto de Anscombe
    x1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
    y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])

    x2 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
    y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])

    x3 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
    y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])

    x4 = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19])
    y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50])

    # Calcular estadísticas descriptivas
    media_x = np.mean(x1)
    varianza_x = np.var(x1, ddof=1)
    estadisticas = {
        'Media de x': media_x,
        'Varianza de x': varianza_x,
        'Media de y1': np.mean(y1),
        'Varianza de y1': np.var(y1, ddof=1),
        'Media de y2': np.mean(y2),
        'Varianza de y2': np.var(y2, ddof=1),
        'Media de y3': np.mean(y3),
        'Varianza de y3': np.var(y3, ddof=1),
        'Media de y4': np.mean(y4),
        'Varianza de y4': np.var(y4, ddof=1),
    }
    

    # Crear un DataFrame para almacenar los resultados
    df_stats = pd.DataFrame(estadisticas, index=['Valores'])
    print(df_stats)

    # Graficar los datos
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Conjunto de datos 1
    axs[0, 0].scatter(x1, y1)
    axs[0, 0].plot(np.unique(x1), np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)), color='red')
    axs[0, 0].set_title('Conjunto de datos 1')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y1')
    axs[0, 0].grid(linestyle='--', alpha=0.7)

    # Conjunto de datos 2
    axs[0, 1].scatter(x2, y2)
    axs[0, 1].plot(np.unique(x2), np.poly1d(np.polyfit(x2, y2, 1))(np.unique(x2)), color='red')
    axs[0, 1].set_title('Conjunto de datos 2')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y2')
    axs[0, 1].grid(linestyle='--', alpha=0.7)

    # Conjunto de datos 3
    axs[1, 0].scatter(x3, y3)
    axs[1, 0].plot(np.unique(x3), np.poly1d(np.polyfit(x3, y3, 1))(np.unique(x3)), color='red')
    axs[1, 0].set_title('Conjunto de datos 3')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y3')
    axs[1, 0].grid(linestyle='--', alpha=0.7)

    # Conjunto de datos 4
    axs[1, 1].scatter(x4, y4)
    axs[1, 1].plot(np.unique(x4), np.poly1d(np.polyfit(x4, y4, 1))(np.unique(x4)), color='red')
    axs[1, 1].set_title('Conjunto de datos 4')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y4')
    axs[1, 1].grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------
#-----------------------Discrete Marginal Distribution -------------------------
#-------------------------------------------------------------------------------


def graficar_marginales_discretas(df, X, Y, P_X, P_Y):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Distribución para X
    axes[0].bar(df[X].unique(), P_X, color='skyblue', edgecolor='black')
    axes[0].set_title('P(X) Marginal')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('P(X)')
    axes[0].set_xticks(df[X].unique())
    axes[0].set_xticklabels(df[X].unique(), rotation=45)
    axes[0].grid(linestyle='--', alpha=0.7)

    # Distribución para Y
    axes[1].bar(df[Y].unique(), P_Y, color='skyblue', edgecolor='black')
    axes[1].set_title('P(Y) Marginal')
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('P(Y)')
    axes[1].set_xticks(df[Y].unique())
    axes[1].set_xticklabels(df[Y].unique(), rotation=45)
    axes[1].grid(linestyle='--', alpha=0.7)

    # Mostrar los gráficos
    plt.tight_layout()
    plt.show()
    

#-------------------------------------------------------------------------------
#-------------------- Continuous Marginal Distribution -------------------------
#-------------------------------------------------------------------------------


def graficar_marginal_continua(X_marginal, Y_marginal, P_X, P_Y):
    plt.figure(figsize=(10, 4))

    # Marginal de X
    plt.subplot(1, 2, 1)
    plt.plot(X_marginal, P_X, label=f'P(X)')
    plt.title('Distribución Marginal de X')
    plt.xlabel('X')
    plt.ylabel('P(X)')
    plt.grid(True)
    plt.grid(linestyle='--', alpha=0.7)

    # Marginal de Y
    plt.subplot(1, 2, 2)
    plt.plot(Y_marginal, P_Y, label=f'P(Y)')
    plt.title('Distribución Marginal de Y')
    plt.xlabel('Y')
    plt.ylabel('P(Y)')
    plt.grid(True)
    plt.grid(linestyle='--', alpha=0.7)

    plt.show()    

#-------------------------------------------------------------------------------
#---------------------------- Density Contour ----------------------------------
#-------------------------------------------------------------------------------
    
def graficar_densidad(X, Y, Z):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Densidad de Probabilidad')
    plt.title('Densidad de Probabilidad Bivariada')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.show()
    

#-------------------------------------------------------------------------------
#--------------------------- Density 3D Plot -----------------------------------
#-------------------------------------------------------------------------------

def graficar_densidad_3d(X, Y, Z):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Gráfico de superficie 3D
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Densidad de Probabilidad Bivariada en 3D')
    ax.set_xlabel('Variable 1')
    ax.set_ylabel('Variable 2')
    ax.set_zlabel('Densidad de Probabilidad')
    
    plt.show()


#-------------------------------------------------------------------------------
#------------------- Conditional density evaluated at x_i ----------------------
#-------------------------------------------------------------------------------
    
    
def densidad_condicional_xi(Y_values, prob_condicional_Y, x_i):
    plt.figure(figsize=(8, 4))
    plt.plot(Y_values, prob_condicional_Y, label=f'P(Y | X = {x_i})')
    plt.xlabel('Y')
    plt.ylabel('Probabilidad Condicional')
    plt.title(f'Probabilidad Condicional P(Y | X = {x_i})')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    
    
#--------------------------------------------------------------------------------
#------------------ Conditional density evaluated at y_j ------------------------
#--------------------------------------------------------------------------------
    
    
def densidad_condicional_yj(X_values, prob_condicional_X, y_j):
    plt.figure(figsize=(8, 4))
    plt.plot(X_values, prob_condicional_X, label=f'P(X | Y = {y_j})')
    plt.xlabel('Y')
    plt.ylabel('Probabilidad Condicional')
    plt.title(f'Probabilidad Condicional P(X | Y = {y_j})')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
 

#--------------------------------------------------------------------------------
#------------------------ Correlation Data example ------------------------------
#--------------------------------------------------------------------------------


def ejemplo_correlacion():
    # Generar datos sintéticos
    np.random.seed(0)

    # Correlación Positiva
    X_pos = np.arange(0, 10, 0.5)
    Y_pos = 2 * X_pos + np.random.normal(0.2, 1, len(X_pos))

    # Correlación Nula
    X_null = np.arange(0, 10, 0.5)
    Y_null = np.random.normal(0, 0.2, len(X_null))

    # Correlación Negativa
    X_neg = np.arange(0, 10, 0.5)
    Y_neg = -2 * X_neg + np.random.normal(0.2, 1, len(X_neg))

    # Graficar los resultados
    plt.figure(figsize=(15, 5))

    # Correlación Positiva
    plt.subplot(1, 3, 1)
    plt.scatter(X_pos, Y_pos, color='green')
    plt.title('Correlación Positiva')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(linestyle='--', alpha=0.7)

    # Correlación Nula
    plt.subplot(1, 3, 2)
    plt.scatter(X_null, Y_null, color='blue')
    plt.title('Correlación Nula')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(linestyle='--', alpha=0.7)

    # Correlación Negativa
    plt.subplot(1, 3, 3)
    plt.scatter(X_neg, Y_neg, color='red')
    plt.title('Correlación Negativa')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return X_pos, Y_pos, X_null, Y_null, X_neg, Y_neg


#--------------------------------------------------------------------------------
#------------------------- Multivariate Gaussian 3D plot ------------------------
#--------------------------------------------------------------------------------


def graficar_gauss_3D(mu, cov, x_ints, y_ints):
    # Crear una malla de puntos para evaluar la densidad
    x = np.linspace(x_ints[0], x_ints[1], 100)
    y = np.linspace(y_ints[0], y_ints[1], 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Calcular la densidad utilizando la función multivariada normal
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)

    # Graficar la distribución en 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

    # Etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Densidad')
    ax.set_title('Distribución Gaussiana Multivariable')

    plt.show()
    
 
#--------------------------------------------------------------------------------
#------------------------- Gaussian Ellipses Graph ------------------------------
#--------------------------------------------------------------------------------
 

def graficar_elipsoides_gauss(mean, cov_positive, cov_zero, cov_negative):
    # Crear una malla de puntos
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Crear las distribuciones gaussianas multivariables
    rv_positive = multivariate_normal(mean, cov_positive)
    rv_zero = multivariate_normal(mean, cov_zero)
    rv_negative = multivariate_normal(mean, cov_negative)

    # Calcular las densidades para cada tipo de covarianza
    Z_positive = rv_positive.pdf(pos)
    Z_zero = rv_zero.pdf(pos)
    Z_negative = rv_negative.pdf(pos)

    # Plotear los mapas de contorno
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Covarianza positiva
    axs[0].contour(X, Y, Z_positive, cmap='viridis')
    axs[0].set_title('Covarianza Positiva')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].grid(linestyle='--', alpha=0.7)

    # Covarianza nula
    axs[1].contour(X, Y, Z_zero, cmap='viridis')
    axs[1].set_title('Covarianza Nula')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].grid(linestyle='--', alpha=0.7)

    # Covarianza negativa
    axs[2].contour(X, Y, Z_negative, cmap='viridis')
    axs[2].set_title('Covarianza Negativa')
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')
    axs[2].grid(linestyle='--', alpha=0.7)

    plt.show()


#--------------------------------------------------------------------------------
#---------------------- Multinomial distribution graph --------------------------
#--------------------------------------------------------------------------------

def graficar_multinomial(X, Y, probs, valid):
    # Crear gráfico 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar los puntos válidos
    ax.bar3d(X[valid], Y[valid], np.zeros_like(probs[valid]), 1, 1, probs[valid], shade=True, color='cyan')

    # Etiquetas
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability')
    ax.set_title('3D Plot of Multinomial Distribution')

    plt.show()  
    
    
#--------------------------------------------------------------------------------
#-------------------------- Percentage of null data -----------------------------
#--------------------------------------------------------------------------------

class Data_Processing:
  def __init__(self):
    pass

  # Percentage of null values
  def null_classifier(self,dataframe):
    series = dataframe.isnull().sum() * 100 / dataframe.shape[0]
    nulls  = {'Variable': series.index, 'Percentage of null values': series.values}
    nulls = pd.DataFrame(nulls).sort_values(by = 'Percentage of null values', ascending = False)
    return nulls

  # Simple column selector
  def column_selector(self, dataframe, column_names):
    df = dataframe
    selected_columns = [col for col in column_names if col in df.columns]
    selected_df = df[selected_columns]
    return selected_df

  # Column selector by null percentage
  def null_df_filter(self, dataframe, threshold):
    df = dataframe
    nulls = self.null_classifier(df)
    filtered = nulls[nulls['Percentage of null values'] < threshold]['Variable']
    df_filtered = df[filtered]
    return df_filtered

#--------------------------------------------------------------------------------
#------------------------------ Nullity Bar Plot --------------------------------
#--------------------------------------------------------------------------------

def nullity_plot(df, threshold=30):
    tot = len(df)
    nullity = (df.isnull().sum() * 100) / tot
    nullity = pd.DataFrame({'Variable':nullity.index, 'Nullity Percentage': nullity.values})
    nullity = nullity.sort_values(by='Nullity Percentage', ascending=False)
    plt.bar(nullity['Variable'], nullity['Nullity Percentage'])
    plt.axhline(y=threshold, color='r', linestyle='dashed', label='Threshold')
    plt.title('Nullity Percentage')
    plt.xlabel('Variables')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()


#--------------------------------------------------------------------------------
#----------------------------- Test de imputación -------------------------------
#--------------------------------------------------------------------------------

def imputation_test(original_dataframe, imputed_dataframe, problematic_columns):
  df_before = original_dataframe
  df_after = imputed_dataframe
  statistics = pd.DataFrame({})
  statistics['Item'] = ['t-test', 'p-value', 'ks_test', 'p_value_ks','observed mean', 'imputed mean','observed variance', 'imputed variance', 'mean difference', 'var difference']

  for column in problematic_columns:
    observed_values = df_before[column].dropna()
    imputed_values = df_after[column].dropna()

    mean_observed = observed_values.mean()
    mean_imputed = imputed_values.mean()
    var_observed = observed_values.var()
    imp_observed = imputed_values.var()
    var_difference = abs(var_observed - imp_observed)
    mean_difference = abs(mean_observed - mean_imputed)


    # two-sided t-test
    t_stat, p_value = sts.ttest_ind(observed_values, imputed_values, equal_var = False)
    # ks-test
    ks_stat, p_value_ks = ks_2samp(observed_values, imputed_values)
    statistics[column] = [t_stat, p_value, ks_stat, p_value_ks ,mean_observed, mean_imputed, var_observed, imp_observed, mean_difference, var_difference]


  return statistics

#--------------------------------------------------------------------------------
#--------------------- Plot de validación de imputación -------------------------
#--------------------------------------------------------------------------------

class imputation_plots:

  def __init__(self):
    pass

  def plot_inputation_comparison(self, observed_df, imputed_df, problematic_columns, subplot_layout, size):
    rows, cols = subplot_layout
    r, c = rows, cols
    fig, axs = plt.subplots(rows,cols, figsize = size)

    def plot_function(position,obs_val, imp_val, col_name, x_limits):
      mean_obs = np.mean(obs_val)
      mean_imp = np.mean(imp_val)
      i , j = position[0] , position[1]
      ax = axs[i,j]
      sns.histplot(observed_values, label='Original',kde=True, color='blue', ax=ax)
      sns.histplot(imputed_values, label='Imputed',kde=True, color='orange', ax=ax)
      ax.axvline(mean_obs, color='black')
      ax.axvline(mean_imp, color='red')
      ax.set_title(f'Comparison for {column}')
      ax.set_xlabel(f'{column}')
      ax.set_ylabel('Frequency')
      ax.grid(True)
      ax.set_xlim(x_limits[0], x_limits[1])
      ax.legend()

    for k, column in enumerate(problematic_columns):

      observed_values = observed_df[column].dropna()
      imputed_values = imputed_df[column].dropna()
      limits = [min(observed_values.min(),imputed_values.min()),
                max(observed_values.max(),imputed_values.max())]

      i = rows - r
      j = cols - c

      plot_function([i,j], observed_values, imputed_values, column, limits)
      c -= 1
      if c > 0:
        r = r
      else:
        r -= 1
        c = cols

    plt.show()