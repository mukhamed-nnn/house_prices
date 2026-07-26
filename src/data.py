"""Загрузка исходных данных и построение признаков для House Prices."""
import pandas as pd

def load_raw_data(path_to_train: str, path_to_test: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path_to_train)
    test = pd.read_csv(path_to_test)
    return train, test

def preprocessing(train, test, cv):
    train = train[~train['Id'].isin([1299, 524])].reset_index(drop=True)
    train = train[train['GrLivArea'] <= 4000]
    df = pd.concat([train, test], sort=False).reset_index(drop=True)
    train_mask = df['SalePrice'].notna()

    # ЗАПОЛНЕНИЕ ПРОПУСКОВ
    none_columns_categorical = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType',
    ]
    for col in none_columns_categorical:
        df[col] = df[col].fillna('None')
        
    none_columns_numeric = [
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea',
    ]
    for col in none_columns_numeric:
        df[col] = df[col].fillna(0)

    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    train_medians = df[train_mask].groupby('Neighborhood')['LotFrontage'].median()
    df['LotFrontage'] = df['LotFrontage'].fillna(df['Neighborhood'].map(train_medians))
    df['LotFrontage'] = df['LotFrontage'].fillna(df.loc[train_mask, 'LotFrontage'].median())

    mode_fill_columns = ['MSZoning', 'Utilities', 'Functional', 'Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'SaleType']
    for col in mode_fill_columns:
        train_mode = df.loc[train_mask, col].mode()[0]
        df[col] = df[col].fillna(train_mode)

    df['MSSubClass'] = df['MSSubClass'].astype('category')

    # Совокупные площади
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    # Ванные комнаты
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    # Возраст объекта на момент продажи
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    # Бинарные флаги - наличие объекта
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['WasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['IsNewHouse'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    # Взаимодействия качества и площади
    df['Qual_TotalSF'] = df['OverallQual'] * df['TotalSF']
    df['Qual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']

    quality_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'None': 0}
    quality_columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

    for col in quality_columns:
        df[col] = df[col].map(quality_map)

    # Отдельные шкалы с иным набором градаций
    df['BsmtExposure'] = df['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
    df['BsmtFinType1'] = df['BsmtFinType1'].map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
    df['BsmtFinType2'] = df['BsmtFinType2'].map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
    df['GarageFinish'] = df['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
    df['Fence'] = df['Fence'].map({'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4})
    df['PavedDrive'] = df['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
    df['Functional'] = df['Functional'].map({'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7})

    # Раз преобразованы в числа — исключаем их из категориального пайплайна OHE/target encoding
    already_ordinal = quality_columns + ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish', 'Fence', 'PavedDrive', 'Functional']
    
    exclude_columns = ['Id', 'SalePrice'] + already_ordinal
    THRESHOLD = 6
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_features = [col for col in num_features if col not in exclude_columns]
    cat_columns = df.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    cat_columns = [col for col in cat_columns if col not in exclude_columns]
    ohe_features = []
    target_features = []
    for col in cat_columns:
        n_unique = df.loc[train_mask, col].nunique()
        if n_unique <= THRESHOLD:
            ohe_features.append(col)
        else:
            target_features.append(col)

    df = df.drop(columns=['Id'])
    df_cat = df.copy()
    df = pd.get_dummies(df, columns=ohe_features)
    df = target_encode_cv(df, train_mask, target_features, y=df.loc[train_mask, 'SalePrice'], cv=cv, smoothing=10)
    

    train, test = df[train_mask].copy(), df[~train_mask].copy()
    train_cat, test_cat = df_cat[train_mask].copy(), df_cat[~train_mask].copy()
    test = test.drop(columns=['SalePrice'])
    test_cat = test_cat.drop(columns=['SalePrice'])

    return train, test, train_cat, test_cat


def target_encode_cv(df, train_mask, target_features, y, cv, smoothing=10):
    """
    Честный (без утечки) target encoding для категориальных признаков высокой кардинальности.

    Train: каждая строка получает OOF-закодированное значение — среднее по таргету
    внутри категории, посчитанное БЕЗ фолда, в который попала сама строка.
    Test: кодируется средним по категории, посчитанным на всём train.

    smoothing сглаживает редкие категории к глобальному среднему:
    encoded = (count * category_mean + smoothing * global_mean) / (count + smoothing)
    """
    df = df.copy()
    global_mean = y.mean()
    train_idx = df.index[train_mask]
    test_idx = df.index[~train_mask]

    for col in target_features:
        encoded = pd.Series(index=df.index, dtype=float)

        # --- Train: честные OOF-значения ---
        for tr_pos, val_pos in cv.split(train_idx):
            fold_train_idx = train_idx[tr_pos]
            fold_val_idx = train_idx[val_pos]

            stats = pd.DataFrame({'y': y.loc[fold_train_idx], 'cat': df.loc[fold_train_idx, col]}) \
                        .groupby('cat')['y'].agg(['mean', 'count'])
            smoothed = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)

            encoded.loc[fold_val_idx] = df.loc[fold_val_idx, col].map(smoothed)

        # категории, не встретившиеся в конкретном обучающем фолде, — глобальным средним
        encoded.loc[train_idx] = encoded.loc[train_idx].fillna(global_mean)

        # --- Test: кодируем средним по всему train ---
        stats_full = pd.DataFrame({'y': y, 'cat': df.loc[train_idx, col]}).groupby('cat')['y'].agg(['mean', 'count'])
        smoothed_full = (stats_full['count'] * stats_full['mean'] + smoothing * global_mean) / (stats_full['count'] + smoothing)
        encoded.loc[test_idx] = df.loc[test_idx, col].map(smoothed_full).fillna(global_mean)

        df[col + '_te'] = encoded

    return df.drop(columns=target_features)