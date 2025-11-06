# 🚢 Titanic Survival Prediction - Regressão Logística

Projeto de Data Science para prever a sobrevivência de passageiros do Titanic utilizando Regressão Logística.

## 📁 Estrutura do Projeto

```
datascience/
│
├── input/                          # Dados de entrada
│   └── titanic.csv                 # Dataset do Titanic
│
├── notebooks/                      # Notebooks Jupyter
│   ├── 01_eda.ipynb               # Análise Exploratória de Dados
│   └── 02_modelagem.ipynb         # Modelagem e Avaliação
│
├── src/                           # Código fonte Python
│   └── utils.py                   # Funções auxiliares
│
├── output/                        # Resultados e modelos salvos
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
│
├── requirements.txt               # Dependências do projeto
├── .gitignore                     # Arquivos ignorados pelo Git
└── README.md                      # Este arquivo
```

## 🎯 Objetivo

Desenvolver um modelo de classificação binária para prever se um passageiro sobreviveu ou não ao desastre do Titanic, baseado em características como:
- Classe socioeconômica (Pclass)
- Sexo
- Idade
- Número de familiares a bordo
- Tarifa paga
- Porto de embarque

## 📊 Notebooks

### 1️⃣ `01_eda.ipynb` - Análise Exploratória de Dados
Análise completa do dataset incluindo:
- Visão geral dos dados
- Análise de valores faltantes
- Estatísticas descritivas
- Visualizações exploratórias
- Matriz de correlação
- Identificação de padrões

### 2️⃣ `02_modelagem.ipynb` - Modelagem
Pipeline completo de Machine Learning:
- Pré-processamento dos dados
- Feature Engineering
- Divisão treino/teste
- Treinamento da Regressão Logística
- Avaliação do modelo
- Otimização de hiperparâmetros
- Interpretação dos resultados

## 🚀 Como Usar

### Pré-requisitos
- Python 3.8 ou superior
- pip

### Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd datascience
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute os notebooks:
```bash
jupyter notebook
```

## 📈 Resultados

O modelo de Regressão Logística alcançou:
- **Acurácia**: ~80%
- **ROC-AUC**: ~0.85
- **Precisão**: ~0.80
- **Recall**: ~0.75

### Features Mais Importantes
1. **Sexo**: Mulheres tiveram maior taxa de sobrevivência
2. **Classe**: Passageiros da 1ª classe sobreviveram mais
3. **Título**: Mr., Mrs., Miss têm impactos diferentes
4. **Tarifa**: Correlacionada positivamente com sobrevivência

## 🔍 Insights Principais

- ⚠️ **"Mulheres e crianças primeiro"** foi uma política real
- 💰 Classe socioeconômica teve grande impacto na sobrevivência
- 👨‍👩‍👧‍👦 Tamanho da família influenciou as chances de sobrevivência
- 🚢 Porto de embarque teve correlação com sobrevivência

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **Matplotlib & Seaborn**: Visualização
- **Scikit-learn**: Machine Learning
- **Jupyter**: Ambiente interativo

## 📝 Próximos Passos

- [ ] Testar outros algoritmos (Random Forest, XGBoost, SVM)
- [ ] Implementar ensemble methods
- [ ] Feature engineering avançado
- [ ] Deploy do modelo (API REST)
- [ ] Dashboard interativo

## 👤 Autor

Rodrigo

## 📄 Licença

Este projeto é de código aberto e está disponível para fins educacionais.

---

**Nota**: Este é um projeto de estudo baseado no famoso dataset do Titanic, amplamente utilizado para aprendizado de Machine Learning.

