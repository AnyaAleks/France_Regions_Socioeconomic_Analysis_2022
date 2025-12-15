# src/data_aggregation.py
import pandas as pd
import numpy as np


def aggregate_data(data):
    """Агрегация и анализ региональных данных"""
    print("Агрегация региональных данных INSEE...")

    if data.empty:
        print("✗ Нет данных для агрегации")
        return data

    print(f"  Исходные данные: {data.shape[0]} регионов, {data.shape[1]} показателей")

    aggregated = data.copy()

    print("\n1. РАСЧЕТ СВОДНОЙ СТАТИСТИКИ:")
    print("-" * 40)

    # 1.1 Базовые статистики по Франции в целом
    numeric_cols = aggregated.select_dtypes(include=[np.number]).columns

    print(f"\n  СВОДКА ПО ВСЕЙ ФРАНЦИИ (МЕТРОПОЛИЯ):")
    print(f"  {'Показатель':<25} {'Среднее':>12} {'Медиана':>12} {'Min':>12} {'Max':>12} {'Std':>12}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

    key_indicators = [
        'total_population', 'gdp_per_capita', 'unemployment_rate',
        'higher_education_rate', 'average_salary', 'population_density',
        'total_gdp', 'dev_index'
    ]

    france_summary = {}

    for indicator in key_indicators:
        if indicator in aggregated.columns:
            mean_val = aggregated[indicator].mean()
            median_val = aggregated[indicator].median()
            min_val = aggregated[indicator].min()
            max_val = aggregated[indicator].max()
            std_val = aggregated[indicator].std()

            france_summary[indicator] = {
                'mean': mean_val, 'median': median_val,
                'min': min_val, 'max': max_val, 'std': std_val
            }

            # Форматируем вывод в зависимости от типа показателя
            if indicator in ['total_population', 'total_gdp']:
                print(
                    f"  {indicator:<25} {mean_val / 1e6:>11.1f}M {median_val / 1e6:>11.1f}M {min_val / 1e6:>11.1f}M {max_val / 1e6:>11.1f}M {std_val / 1e6:>11.1f}M")
            elif indicator == 'gdp_per_capita':
                print(
                    f"  {indicator:<25} {mean_val:>11,.0f}€ {median_val:>11,.0f}€ {min_val:>11,.0f}€ {max_val:>11,.0f}€ {std_val:>11,.0f}€")
            elif indicator == 'average_salary':
                print(
                    f"  {indicator:<25} {mean_val:>11,.0f}€ {median_val:>11,.0f}€ {min_val:>11,.0f}€ {max_val:>11,.0f}€ {std_val:>11,.0f}€")
            elif indicator in ['unemployment_rate', 'higher_education_rate']:
                print(
                    f"  {indicator:<25} {mean_val:>11.1f}% {median_val:>11.1f}% {min_val:>11.1f}% {max_val:>11.1f}% {std_val:>11.1f}%")
            elif indicator == 'population_density':
                print(
                    f"  {indicator:<25} {mean_val:>11.0f} {median_val:>11.0f} {min_val:>11.0f} {max_val:>11.0f} {std_val:>11.0f}")
            else:
                print(
                    f"  {indicator:<25} {mean_val:>11.3f} {median_val:>11.3f} {min_val:>11.3f} {max_val:>11.3f} {std_val:>11.3f}")

    print("\n2. РАНЖИРОВАНИЕ РЕГИОНОВ:")
    print("-" * 40)

    # 2.1 Топ-5 регионов по ключевым показателям
    rankings = {}

    print(f"\n  ТОП-5 РЕГИОНОВ ПО ВВП НА ДУШУ:")
    if 'gdp_per_capita' in aggregated.columns:
        top_gdp = aggregated.nlargest(5, 'gdp_per_capita')[['region', 'gdp_per_capita']]
        rankings['top_gdp'] = top_gdp
        for idx, (_, row) in enumerate(top_gdp.iterrows(), 1):
            print(f"    {idx}. {row['region']}: {row['gdp_per_capita']:,.0f}€")

    print(f"\n  ТОП-5 РЕГИОНОВ ПО УРОВНЮ ОБРАЗОВАНИЯ:")
    if 'higher_education_rate' in aggregated.columns:
        top_edu = aggregated.nlargest(5, 'higher_education_rate')[['region', 'higher_education_rate']]
        rankings['top_education'] = top_edu
        for idx, (_, row) in enumerate(top_edu.iterrows(), 1):
            print(f"    {idx}. {row['region']}: {row['higher_education_rate']:.1f}%")

    print(f"\n  РЕГИОНЫ С САМОЙ НИЗКОЙ БЕЗРАБОТИЦЕЙ:")
    if 'unemployment_rate' in aggregated.columns:
        low_unemp = aggregated.nsmallest(5, 'unemployment_rate')[['region', 'unemployment_rate']]
        rankings['low_unemployment'] = low_unemp
        for idx, (_, row) in enumerate(low_unemp.iterrows(), 1):
            print(f"    {idx}. {row['region']}: {row['unemployment_rate']:.1f}%")

    print(f"\n  РЕГИОНЫ С САМОЙ ВЫСОКОЙ БЕЗРАБОТИЦЕЙ:")
    if 'unemployment_rate' in aggregated.columns:
        high_unemp = aggregated.nlargest(5, 'unemployment_rate')[['region', 'unemployment_rate']]
        rankings['high_unemployment'] = high_unemp
        for idx, (_, row) in enumerate(high_unemp.iterrows(), 1):
            print(f"    {idx}. {row['region']}: {row['unemployment_rate']:.1f}%")

    print("\n3. ГРУППИРОВКА И КЛАСТЕРИЗАЦИЯ:")
    print("-" * 40)

    # 3.1 Группировка по категориям развития
    if 'development_level' in aggregated.columns:
        development_groups = aggregated.groupby('development_level').agg({
            'region': 'count',
            'gdp_per_capita': 'mean',
            'unemployment_rate': 'mean',
            'higher_education_rate': 'mean'
        }).round(2)

        print(f"\n  ГРУППЫ ПО УРОВНЮ РАЗВИТИЯ:")
        print(f"  {'Уровень':<20} {'Регионов':>10} {'ВВП/чел':>10} {'Безработица':>12} {'Образование':>12}")
        print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 12}")

        for level, group in development_groups.iterrows():
            print(
                f"  {str(level):<20} {group['region']:>10} {group['gdp_per_capita']:>9,.0f}€ {group['unemployment_rate']:>11.1f}% {group['higher_education_rate']:>11.1f}%")

    # 3.2 Группировка по категориям ВВП
    if 'gdp_category' in aggregated.columns:
        gdp_groups = aggregated.groupby('gdp_category').agg({
            'region': 'count',
            'unemployment_rate': 'mean',
            'higher_education_rate': 'mean'
        }).round(2)

        print(f"\n  ГРУППЫ ПО УРОВНЮ ВВП:")
        print(f"  {'Категория ВВП':<20} {'Регионов':>10} {'Безработица':>12} {'Образование':>12}")
        print(f"  {'-' * 20} {'-' * 10} {'-' * 12} {'-' * 12}")

        for category, group in gdp_groups.iterrows():
            print(
                f"  {str(category):<20} {group['region']:>10} {group['unemployment_rate']:>11.1f}% {group['higher_education_rate']:>11.1f}%")

    print("\n4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ:")
    print("-" * 40)

    # 4.1 Матрица корреляций
    correlation_cols = [
        'gdp_per_capita', 'unemployment_rate', 'higher_education_rate',
        'average_salary', 'population_density', 'dev_index'
    ]

    available_corr_cols = [col for col in correlation_cols if col in aggregated.columns]

    if len(available_corr_cols) >= 2:
        correlation_matrix = aggregated[available_corr_cols].corr()

        print(f"\n  МАТРИЦА КОРРЕЛЯЦИЙ (Pearson):")
        print(f"  {'':<20}", end="")
        for col in available_corr_cols:
            print(f" {col[:10]:>10}", end="")
        print()

        for row in available_corr_cols:
            print(f"  {row:<20}", end="")
            for col in available_corr_cols:
                corr = correlation_matrix.loc[row, col]
                if row == col:
                    print(f" {'1.00':>10}", end="")
                else:
                    print(f" {corr:>10.2f}", end="")
            print()

        # Выделяем сильные корреляции
        print(f"\n  СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| > 0.7):")
        strong_correlations = []
        for i in range(len(available_corr_cols)):
            for j in range(i + 1, len(available_corr_cols)):
                col1, col2 = available_corr_cols[i], available_corr_cols[j]
                corr = correlation_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    direction = "положительная" if corr > 0 else "отрицательная"
                    strong_correlations.append((col1, col2, corr, direction))

        if strong_correlations:
            for col1, col2, corr, direction in strong_correlations:
                print(f"    {col1} ↔ {col2}: r = {corr:.2f} ({direction})")
        else:
            print(f"    Нет сильных корреляций")

    print("\n5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("-" * 40)

    # Добавляем расчетные колонки для анализа
    if 'total_population' in aggregated.columns and 'total_gdp' in aggregated.columns:
        # Доля в общем населении и ВВП
        total_pop_france = aggregated['total_population'].sum()
        total_gdp_france = aggregated['total_gdp'].sum()

        aggregated['population_share'] = (aggregated['total_population'] / total_pop_france * 100).round(2)
        aggregated['gdp_share'] = (aggregated['total_gdp'] / total_gdp_france * 100).round(2)

        print(f"  ✓ Добавлены доли в общем населении и ВВП")

    # Добавляем рейтинги
    for rank_name, rank_df in rankings.items():
        rank_col = f'rank_{rank_name}'
        for _, row in rank_df.iterrows():
            region = row['region']
            value = row[rank_df.columns[1]]
            aggregated.loc[aggregated['region'] == region, rank_col] = value

    print(f"\n✓ АГРЕГИРОВАННЫЕ ДАННЫХ: {aggregated.shape[0]} регионов, {aggregated.shape[1]} показателей")
    print(f"  Ключевые агрегированные показатели сохранены")

    return aggregated


def save_aggregated_results(data, output_dir='data/processed'):
    """Сохранение агрегированных результатов"""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем полные данные
    output_path = os.path.join(output_dir, 'regional_data_aggregated.csv')
    data.to_csv(output_path, index=False, encoding='utf-8-sig')

    # Создаем сводный отчет
    summary_path = os.path.join(output_dir, 'regional_summary.txt')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("СВОДНЫЙ ОТЧЕТ: Анализ регионов Франции\n")
        f.write("=" * 60 + "\n\n")

        f.write("ОБЩАЯ ИНФОРМАЦИЯ:\n")
        f.write(f"  Всего регионов: {len(data)}\n")
        f.write(f"  Всего показателей: {len(data.columns)}\n\n")

        f.write("КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:\n")
        key_cols = ['region', 'total_population', 'gdp_per_capita',
                    'unemployment_rate', 'higher_education_rate', 'average_salary']

        for col in key_cols:
            if col in data.columns:
                if col == 'region':
                    continue
                mean_val = data[col].mean()
                f.write(f"  {col}: среднее = {mean_val:,.2f}\n")

        f.write("\nТОП-3 РЕГИОНА ПО ВВП НА ДУШУ:\n")
        if 'gdp_per_capita' in data.columns:
            top3 = data.nlargest(3, 'gdp_per_capita')[['region', 'gdp_per_capita']]
            for _, row in top3.iterrows():
                f.write(f"  {row['region']}: {row['gdp_per_capita']:,.0f}€\n")

    print(f"✓ Результаты сохранены в {output_dir}/")

    return output_path


# Для тестирования модуля
if __name__ == "__main__":
    # Создаем тестовые данные
    test_data = pd.DataFrame({
        'region': ['Île-de-France', 'Centre-Val de Loire', 'Bourgogne-Franche-Comté',
                   'Normandie', 'Hauts-de-France', 'Grand Est', 'Pays de la Loire'],
        'total_population': [12200000, 2500000, 2800000, 3300000, 6000000, 5500000, 3800000],
        'gdp_per_capita': [65000, 29000, 30000, 28000, 27000, 31000, 33000],
        'unemployment_rate': [8.5, 10.5, 11.0, 10.8, 12.5, 9.8, 8.9],
        'higher_education_rate': [25.5, 18.2, 19.0, 17.8, 16.5, 19.5, 20.2],
        'average_salary': [3800, 2850, 2900, 2800, 2750, 3000, 3150],
        'development_level': pd.Categorical(['Очень высокий', 'Средний', 'Средний',
                                             'Низкий', 'Низкий', 'Средний', 'Высокий'],
                                            categories=['Очень низкий', 'Низкий', 'Средний',
                                                        'Высокий', 'Очень высокий'])
    })

    print("Тестирование агрегации данных...")
    aggregated = aggregate_data(test_data)

    if not aggregated.empty:
        print("\nАгрегированные данные:")
        print(aggregated.head())

        # Сохраняем результаты
        save_aggregated_results(aggregated, 'test_output')