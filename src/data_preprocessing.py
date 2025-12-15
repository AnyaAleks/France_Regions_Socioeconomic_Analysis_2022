# src/data_preprocessing.py
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def preprocess_data(data):
    """Предварительная обработка региональных данных INSEE"""
    print("Предобработка региональных данных INSEE...")

    if data.empty:
        print("✗ Нет данных для обработки")
        return data

    print(f"  Исходный размер: {data.shape[0]} строк, {data.shape[1]} колонок")

    data_clean = data.copy()

    print("\n1. ПРОВЕРКА И ОЧИСТКА ДАННЫХ:")
    print("-" * 40)

    # 1.1 Проверка дубликатов
    if data_clean.duplicated().any():
        duplicates = data_clean.duplicated().sum()
        data_clean = data_clean.drop_duplicates()
        print(f"  ✓ Удалено {duplicates} дубликатов")

    # 1.2 Проверка пропущенных значений
    missing_values = data_clean.isnull().sum()
    if missing_values.any():
        print(f"  Пропущенные значения:")
        for col, count in missing_values[missing_values > 0].items():
            percentage = (count / len(data_clean)) * 100
            print(f"    - {col}: {count} ({percentage:.1f}%)")

    # 1.3 Заполнение пропущенных значений
    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        na_count = data_clean[col].isna().sum()
        if na_count > 0:
            # Заполняем медианой, если не все значения пропущены
            if na_count < len(data_clean):
                median_val = data_clean[col].median()
                data_clean[col] = data_clean[col].fillna(median_val)
                print(f"  ✓ Заполнено {na_count} пропусков в '{col}' (медиана: {median_val:.2f})")

    # 1.4 Обработка выбросов для ключевых показателей
    key_indicators = ['gdp_per_capita', 'unemployment_rate', 'average_salary', 'population_density']

    for indicator in key_indicators:
        if indicator in data_clean.columns:
            q1 = data_clean[indicator].quantile(0.25)
            q3 = data_clean[indicator].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = data_clean[(data_clean[indicator] < lower_bound) | (data_clean[indicator] > upper_bound)]
            if len(outliers) > 0:
                print(f"  ⚠ Найдено {len(outliers)} выбросов в '{indicator}'")
                # Можно заменить на граничные значения или оставить

    print("\n2. СТАНДАРТИЗАЦИЯ И НОРМАЛИЗАЦИЯ:")
    print("-" * 40)

    # 2.1 Стандартизация числовых признаков (z-score)
    cols_to_standardize = ['gdp_per_capita', 'average_salary', 'population_density']

    for col in cols_to_standardize:
        if col in data_clean.columns:
            mean_val = data_clean[col].mean()
            std_val = data_clean[col].std()
            if std_val > 0:
                data_clean[f'{col}_zscore'] = (data_clean[col] - mean_val) / std_val
                print(f"  ✓ Стандартизирован '{col}' (z-score)")

    # 2.2 Нормализация в диапазон 0-1 (min-max scaling)
    cols_to_normalize = ['higher_education_rate', 'dev_index']

    for col in cols_to_normalize:
        if col in data_clean.columns:
            min_val = data_clean[col].min()
            max_val = data_clean[col].max()
            range_val = max_val - min_val

            if range_val > 0:
                data_clean[f'{col}_normalized'] = (data_clean[col] - min_val) / range_val
                print(f"  ✓ Нормализован '{col}' (0-1)")

    # 2.3 Логарифмирование для сильно скошенных распределений
    skewed_cols = ['total_population', 'total_gdp', 'main_dwellings']

    for col in skewed_cols:
        if col in data_clean.columns:
            # Проверяем, что все значения положительные
            if (data_clean[col] > 0).all():
                data_clean[f'log_{col}'] = np.log1p(data_clean[col])
                print(f"  ✓ Применен логарифм к '{col}'")

    print("\n3. СОЗДАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ:")
    print("-" * 40)

    # 3.1 Категоризация регионов по уровню развития
    if 'dev_index' in data_clean.columns:
        # Квинтили для разделения на 5 групп
        data_clean['development_level'] = pd.qcut(
            data_clean['dev_index'],
            q=5,
            labels=['Очень низкий', 'Низкий', 'Средний', 'Высокий', 'Очень высокий']
        )
        print(f"  ✓ Создана категория 'development_level'")

        # Простая бинарная категоризация
        median_dev = data_clean['dev_index'].median()
        data_clean['is_developed'] = data_clean['dev_index'] > median_dev
        print(f"  ✓ Создана категория 'is_developed'")

    # 3.2 Категоризация по ВВП на душу
    if 'gdp_per_capita' in data_clean.columns:
        gdp_bins = [0, 25000, 35000, 45000, 100000]
        gdp_labels = ['Низкий', 'Ниже среднего', 'Выше среднего', 'Высокий']
        data_clean['gdp_category'] = pd.cut(
            data_clean['gdp_per_capita'],
            bins=gdp_bins,
            labels=gdp_labels,
            include_lowest=True
        )
        print(f"  ✓ Создана категория 'gdp_category'")

    # 3.3 Категоризация по безработице
    if 'unemployment_rate' in data_clean.columns:
        unemployment_bins = [0, 8, 10, 12, 100]
        unemployment_labels = ['Очень низкая', 'Низкая', 'Средняя', 'Высокая']
        data_clean['unemployment_category'] = pd.cut(
            data_clean['unemployment_rate'],
            bins=unemployment_bins,
            labels=unemployment_labels,
            include_lowest=True
        )
        print(f"  ✓ Создана категория 'unemployment_category'")

    print("\n4. СОЗДАНИЕ ВЗАИМОДЕЙСТВИЙ И ПОЛИНОМИАЛЬНЫХ ПРИЗНАКОВ:")
    print("-" * 40)

    # 4.1 Взаимодействие между ключевыми показателями
    if all(col in data_clean.columns for col in ['gdp_per_capita', 'higher_education_rate']):
        data_clean['gdp_edu_interaction'] = data_clean['gdp_per_capita'] * data_clean['higher_education_rate']
        print(f"  ✓ Создано взаимодействие: gdp_edu_interaction")

    if all(col in data_clean.columns for col in ['gdp_per_capita', 'unemployment_rate']):
        data_clean['gdp_unemp_interaction'] = data_clean['gdp_per_capita'] / (data_clean['unemployment_rate'] + 1)
        print(f"  ✓ Создано взаимодействие: gdp_unemp_interaction")

    # 4.2 Полиномиальные признаки
    if 'gdp_per_capita' in data_clean.columns:
        data_clean['gdp_squared'] = data_clean['gdp_per_capita'] ** 2
        print(f"  ✓ Создан полиномиальный признак: gdp_squared")

    if 'higher_education_rate' in data_clean.columns:
        data_clean['edu_squared'] = data_clean['higher_education_rate'] ** 2
        print(f"  ✓ Создан полиномиальный признак: edu_squared")

    print("\n5. ФИНАЛЬНАЯ ОЧИСТКА И ПОДГОТОВКА:")
    print("-" * 40)

    # 5.1 Удаление временных колонок
    cols_to_drop = ['region_code_x', 'region_code_y', 'region_code']
    for col in cols_to_drop:
        if col in data_clean.columns:
            data_clean = data_clean.drop(columns=[col])

    # 5.2 Сортировка по региону
    data_clean = data_clean.sort_values('region')

    # 5.3 Сброс индекса
    data_clean = data_clean.reset_index(drop=True)

    # 5.4 Проверка итоговой структуры
    print(f"  Количественные признаки: {len(data_clean.select_dtypes(include=[np.number]).columns)}")
    print(f"  Категориальные признаки: {len(data_clean.select_dtypes(include=['category', 'bool']).columns)}")
    print(f"  Текстовые признаки: {len(data_clean.select_dtypes(include=['object']).columns)}")

    print(f"\n✓ ИТОГОВЫЙ РАЗМЕР: {data_clean.shape[0]} строк, {data_clean.shape[1]} колонок")
    print(f"  Ключевые показатели для анализа:")

    key_cols = [
        'region', 'total_population', 'gdp_per_capita', 'unemployment_rate',
        'higher_education_rate', 'average_salary', 'population_density',
        'development_level', 'total_gdp', 'dev_index'
    ]

    available_key_cols = [col for col in key_cols if col in data_clean.columns]

    for col in available_key_cols[:10]:  # Покажем первые 10
        print(f"    - {col}")

    if len(available_key_cols) > 10:
        print(f"    ... и еще {len(available_key_cols) - 10} показателей")

    return data_clean


def create_summary_statistics(data):
    """Создание сводной статистики по данным"""
    print("\nСоздание сводной статистики...")

    summary = {}

    # Базовые статистики для числовых колонок
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        summary[col] = {
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'q1': data[col].quantile(0.25),
            'q3': data[col].quantile(0.75)
        }

    # Для категориальных колонок
    categorical_cols = data.select_dtypes(include=['category']).columns

    for col in categorical_cols:
        summary[col] = {
            'unique_values': data[col].nunique(),
            'most_common': data[col].mode().iloc[0] if not data[col].mode().empty else None,
            'counts': dict(data[col].value_counts().head())
        }

    return summary


# Для тестирования модуля
if __name__ == "__main__":
    # Создаем тестовые данные
    test_data = pd.DataFrame({
        'code': ['11', '24', '27'],
        'region': ['Île-de-France', 'Centre-Val de Loire', 'Bourgogne-Franche-Comté'],
        'total_population': [12200000, 2500000, 2800000],
        'gdp_per_capita': [65000, 29000, 30000],
        'unemployment_rate': [8.5, 10.5, 11.0],
        'higher_education_rate': [25.5, 18.2, 19.0]
    })

    print("Тестирование предобработки данных...")
    processed = preprocess_data(test_data)

    if not processed.empty:
        print("\nОбработанные данные:")
        print(processed.head())

        print("\nСводная статистика:")
        stats = create_summary_statistics(processed)
        for col, stat in stats.items():
            if 'mean' in stat:
                print(f"{col}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")