# Проект: Прогнозирование цен на недвижимость (House Prices)

Финальный проект по машинному обучению, реализующий комплексный модульный пайплайн предобработки данных, генерации признаков (Feature Engineering), обучения ансамбля регрессионных моделей (ML + DL) и финального стекинга.

## 📊 Основные результаты эксперимента

Все метрики качества рассчитаны с использованием честной Out-of-Fold (OOF) кросс-валидации на 5 фолдах без утечки данных (Data Leakage). 
Целевая переменная оптимизировалась в логарифмическом масштабе ln(1 + x), поэтому основной метрикой является **RMSLE** (Root Mean Squared Logarithmic Error).

### Результаты валидации моделей (Этап 1: Solo Models)
```text
dummy : OOF RMSLE = 0.3962 │ by folds: 0.3961 ± 0.0066
lr    : OOF RMSLE = 0.1149 │ by folds: 0.1146 ± 0.0089
knn   : OOF RMSLE = 0.1605 │ by folds: 0.1603 ± 0.0073
tree  : OOF RMSLE = 0.1638 │ by folds: 0.1634 ± 0.0107
rf    : OOF RMSLE = 0.1415 │ by folds: 0.1411 ± 0.0102
lgbm  : OOF RMSLE = 0.1194 │ by folds: 0.1191 ± 0.0075
xgb   : OOF RMSLE = 0.1186 │ by folds: 0.1183 ± 0.0082
cb    : OOF RMSLE = 0.1187 │ by folds: 0.1185 ± 0.0081
nn    : OOF RMSLE = 0.3438 │ by folds: 0.3425 ± 0.0295
```

### Сводная таблица финального ансамбля (Финальный рейтинг)

| Модель / Архитектура Ансамбля | OOF RMSLE | Характеристика |
| :--- | :---: | :--- |
| 🏆 **Advanced Stacking (Meta-Model)** | **0.1122** | **Лучший стабильный ансамбль** |
| Averaging Ensemble | 0.1134 | Простое усреднение |
| Linear Regression (ElasticNet Tuned) | 0.1149 | Лучшая соло-модель (сильный Feature Eng.) |
| XGBoost | 0.1186 | Градиентный бустинг (DMatrix) |
| CatBoost | 0.1187 | Градиентный бустинг (Категории) |
| LightGBM | 0.1194 | Гистограммный бустинг |
| RandomForest | 0.1415 | Бэггинг над деревьями решений |
| KNeighbors | 0.1605 | Метрический алгоритм (KNN) |
| Decision Tree | 0.1638 | Одиночное решающее дерево |
| PyTorch Neural Network (NN3) | 0.3438 | Полносвязная нейросеть (Deep Learning) |
| Dummy Baseline | 0.3962 | Наивный регрессор (константное среднее) |

## 📁 Структура репозитория
```text
├── data/                  # Исходные датасеты Kaggle (train.csv, test.csv)
├── notebooks/             # Jupyter Notebooks (EDA, анализ выбросов Z-Score/IQR)
├── output/                # Артефакты работы (submission.csv, metrics.json)
├── src/                   # Исходный код модулей пайплайна
│   ├── data.py            # Очистка, OOF Target Encoding и обработка пропусков
│   ├── train.py        # Универсальная OOF-валидация регрессоров (run_model)
│   ├── models.py          # Инициализация ML моделей и архитектура PyTorch NN3
│   └── ensemble.py        # Математика стекинга и блендинга предсказаний
├── main.py                # Главный исполняемый скрипт конвейера
├── config.py              # Конфигурация гиперпараметров (OmegaConf)
└── requirements.txt       # Зависимости проекта (.venv)
```

## 🚀 Инструкция по запуску проекта

Скачайте `train.csv` и `test.csv` с страницы соревнования [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) и поместите их в папку `data/`.

Для воспроизведения полного цикла обучения, генерации признаков и сборки финального ансамбля выполните следующие команды в терминале:

### 1. Клонирование репозитория и переход в папку
```bash
git clone https://github.com/mukhamed-nnn/house_prices
cd house_prices
```

### 2. Настройка виртуального окружения и установка зависимостей
```bash
# Создание изолированного окружения
python -m venv .venv

# Активация для Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Активация для Linux / macOS:
source .venv/bin/activate

# Обновление менеджера пакетов и установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
```

> **Примечание про GPU:** `requirements.txt` нацелен на CUDA 12.6. Если у вас другая версия
> CUDA (проверить: `nvidia-smi`) или нет GPU вообще, замените индекс на подходящий
> (`cu121`/`cu124`) или используйте CPU-версию: `pip install torch` без `--extra-index-url`.

### 3. Запуск основного пайплайна
Убедитесь, что оригинальные файлы `train.csv` и `test.csv` скачаны с Kaggle и находятся в папке `data/`. Запустите конвейер одной командой:
```bash
python main.py
```
После завершения работы скрипта итоговый файл, содержащий обратное экспонирование цен из логарифмов (`expm1`), будет сохранен по пути: `./output/submission.csv`.
