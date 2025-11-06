"""
Funções auxiliares para o projeto Titanic Survival Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle


def load_data(filepath):
    """
    Carrega o dataset do Titanic
    
    Parameters:
    -----------
    filepath : str
        Caminho para o arquivo CSV
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com os dados carregados
    """
    df = pd.read_csv(filepath)
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def check_missing_values(df):
    """
    Verifica valores faltantes no dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a ser analisado
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com contagem e percentual de valores faltantes
    """
    missing = pd.DataFrame({
        'Total': df.isnull().sum(),
        'Percentual': (df.isnull().sum() / len(df)) * 100
    })
    missing = missing[missing['Total'] > 0].sort_values('Total', ascending=False)
    return missing


def plot_survival_rate(df, column, figsize=(12, 5)):
    """
    Plota taxa de sobrevivência por uma variável categórica
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com os dados
    column : str
        Nome da coluna para análise
    figsize : tuple
        Tamanho da figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Contagem
    pd.crosstab(df[column], df['Survived']).plot(
        kind='bar', ax=axes[0], color=['#d62728', '#2ca02c']
    )
    axes[0].set_title(f'Sobrevivência por {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Contagem')
    axes[0].legend(['Não Sobreviveu', 'Sobreviveu'])
    axes[0].tick_params(axis='x', rotation=0)
    
    # Taxa
    survival_rate = df.groupby(column)['Survived'].mean()
    survival_rate.plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title(f'Taxa de Sobrevivência por {column}')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Taxa de Sobrevivência')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()


def preprocess_titanic(df):
    """
    Pré-processa o dataset do Titanic
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame original
        
    Returns:
    --------
    pd.DataFrame
        DataFrame pré-processado
    """
    data = df.copy()
    
    # Preencher valores faltantes
    data['Age'].fillna(data['Age'].median(), inplace=True)
    
    if 'Embarked' in data.columns:
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    if 'Fare' in data.columns:
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Feature Engineering
    if 'SibSp' in data.columns and 'Parch' in data.columns:
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Criar faixas etárias
    data['AgeGroup'] = pd.cut(
        data['Age'], 
        bins=[0, 12, 18, 35, 60, 100], 
        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']
    )
    
    # Extrair título do nome
    if 'Name' in data.columns:
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
        # Agrupar títulos raros
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 
                      'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        data['Title'] = data['Title'].replace(rare_titles, 'Rare')
        data['Title'] = data['Title'].replace('Mlle', 'Miss')
        data['Title'] = data['Title'].replace('Ms', 'Miss')
        data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    return data


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plota matriz de confusão
    
    Parameters:
    -----------
    y_true : array-like
        Valores verdadeiros
    y_pred : array-like
        Valores preditos
    labels : list
        Lista de labels para os eixos
    """
    if labels is None:
        labels = ['Não Sobreviveu', 'Sobreviveu']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.tight_layout()
    plt.show()


def print_model_metrics(y_true, y_pred, y_proba=None):
    """
    Imprime métricas do modelo
    
    Parameters:
    -----------
    y_true : array-like
        Valores verdadeiros
    y_pred : array-like
        Valores preditos
    y_proba : array-like, optional
        Probabilidades preditas
    """
    from sklearn.metrics import (accuracy_score, precision_score, 
                                 recall_score, f1_score, roc_auc_score)
    
    print("="*60)
    print("MÉTRICAS DO MODELO")
    print("="*60)
    
    print(f"\nAcurácia:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precisão:  {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    
    if y_proba is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    
    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*60)
    print(classification_report(y_true, y_pred, 
                               target_names=['Não Sobreviveu', 'Sobreviveu']))


def save_model(model, filepath):
    """
    Salva modelo usando pickle
    
    Parameters:
    -----------
    model : object
        Modelo a ser salvo
    filepath : str
        Caminho do arquivo
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo salvo em: {filepath}")


def load_model(filepath):
    """
    Carrega modelo usando pickle
    
    Parameters:
    -----------
    filepath : str
        Caminho do arquivo
        
    Returns:
    --------
    object
        Modelo carregado
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo carregado de: {filepath}")
    return model


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plota importância das features para modelos lineares
    
    Parameters:
    -----------
    model : sklearn model
        Modelo treinado (deve ter atributo coef_)
    feature_names : list
        Lista com nomes das features
    top_n : int
        Número de features a exibir
    """
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in importance['Coefficient']]
    plt.barh(importance['Feature'], importance['Coefficient'], color=colors)
    plt.xlabel('Coeficiente')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Features Mais Importantes')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    plt.show()


# Configuração padrão para plots
def setup_plot_style():
    """
    Configura estilo padrão para plots
    """
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    print("Estilo de plots configurado!")


if __name__ == "__main__":
    print("Módulo de utilidades carregado com sucesso!")
    print("\nFunções disponíveis:")
    print("  - load_data()")
    print("  - check_missing_values()")
    print("  - plot_survival_rate()")
    print("  - preprocess_titanic()")
    print("  - plot_confusion_matrix()")
    print("  - print_model_metrics()")
    print("  - save_model() / load_model()")
    print("  - plot_feature_importance()")
    print("  - setup_plot_style()")

