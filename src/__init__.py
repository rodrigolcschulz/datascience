"""
Módulo de utilidades para o projeto Titanic Survival Prediction
"""

from .utils import (
    load_data,
    check_missing_values,
    plot_survival_rate,
    preprocess_titanic,
    plot_confusion_matrix,
    print_model_metrics,
    save_model,
    load_model,
    plot_feature_importance,
    setup_plot_style
)

__version__ = '1.0.0'
__all__ = [
    'load_data',
    'check_missing_values',
    'plot_survival_rate',
    'preprocess_titanic',
    'plot_confusion_matrix',
    'print_model_metrics',
    'save_model',
    'load_model',
    'plot_feature_importance',
    'setup_plot_style'
]

