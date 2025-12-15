# src/statistical_tests.py
import pandas as pd
import numpy as np
from scipy import stats
import os

def perform_statistical_tests(data):
    """Выполнение статистических тестов для анализа регионов Франции"""
    print("Выполнение статистических тестов...")
    
    if data.empty:
        print("✗ Нет данных для статистических тестов")
        return
    
    # Создаем папку для отчетов
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    reports_dir = os.path.join(project_root, 'results', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Фильтруем только исходные данные (убираем строку сводной статистики)
    if 'region' in data.columns:
        original_data = data[data['region'] != 'ВСЕ РЕГИОНЫ (СВОДКА)']
    else:
        original_data = data
    
    print(f"Анализируемые показатели: {list(original_data.columns)}")
    
    # Результаты тестов
    results = []
    
    print("\n" + "=" * 80)
    print("СТАТИСТИЧЕСКИЕ ТЕСТЫ")
    print("=" * 80)
    
    # 1. Тесты нормальности
    print("\n" + "-" * 80)
    print("1. ПРОВЕРКА НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ (Шапиро-Уилк)")
    print("-" * 80)
    
    # Выбираем только основные показатели (не статистические)
    main_columns = ['population', 'gdp_per_capita', 'unemployment_rate', 
                   'education_level', 'average_salary', 'total_gdp']
    
    available_columns = [col for col in main_columns if col in original_data.columns]
    
    for col in available_columns:
        values = original_data[col].dropna().values
        
        if len(values) >= 3 and len(values) <= 5000:
            try:
                stat, p_value = stats.shapiro(values)
                is_normal = p_value > 0.05
                
                result = {
                    'Показатель': col,
                    'Статистика W': f"{stat:.4f}",
                    'p-value': f"{p_value:.4f}",
                    'Вывод': 'Нормальное' if is_normal else 'Не нормальное'
                }
                results.append(result)
                
                print(f"\n{col}:")
                print(f"  Статистика W = {stat:.4f}")
                print(f"  p-value = {p_value:.4f}")
                print(f"  Вывод: {'Нормальное' if is_normal else 'Не нормальное'}")
            except Exception as e:
                print(f"\n{col}: Ошибка теста - {e}")
        else:
            print(f"\n{col}: Неприменим (n={len(values)})")
    
    # 2. Корреляционный анализ
    print("\n" + "-" * 80)
    print("2. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ (Спирмен)")
    print("-" * 80)
    
    if len(available_columns) >= 2:
        # Вычисляем корреляцию
        corr_matrix = original_data[available_columns].corr(method='spearman')
        
        print("\nМатрица корреляции:")
        print(corr_matrix.round(3))
        
        # Сохраняем значимые корреляции
        for i in range(len(available_columns)):
            for j in range(i+1, len(available_columns)):
                col1, col2 = available_columns[i], available_columns[j]
                corr_value = corr_matrix.loc[col1, col2]
                
                # Проверяем значимость
                n = len(original_data)
                if n > 2 and abs(corr_value) > 0:
                    t_stat = corr_value * np.sqrt((n-2)/(1-corr_value**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                    is_significant = p_value < 0.05
                    
                    if is_significant:
                        result = {
                            'Показатель': f"Корреляция {col1}-{col2}",
                            'Статистика': f"{corr_value:.3f}",
                            'p-value': f"{p_value:.4f}",
                            'Вывод': f"{'Положительная' if corr_value > 0 else 'Отрицательная'} корреляция"
                        }
                        results.append(result)
                        
                        print(f"\n{col1} - {col2}:")
                        print(f"  Коэффициент Спирмена = {corr_value:.3f}")
                        print(f"  p-value = {p_value:.4f}")
                        print(f"  Вывод: {'Значимая ' + ('положительная' if corr_value > 0 else 'отрицательная') + ' корреляция' if is_significant else 'Не значимая корреляция'}")
    
    # 3. Сравнение групп регионов
    if 'region' in original_data.columns and len(original_data) > 5:
        print("\n" + "-" * 80)
        print("3. СРАВНЕНИЕ РЕГИОНОВ ПО УРОВНЮ БЕЗРАБОТИЦЫ")
        print("-" * 80)
        
        # Разделяем на группы по уровню безработицы
        if 'unemployment_rate' in original_data.columns:
            median_unemployment = original_data['unemployment_rate'].median()
            
            low_unemployment = original_data[original_data['unemployment_rate'] < median_unemployment]
            high_unemployment = original_data[original_data['unemployment_rate'] >= median_unemployment]
            
            print(f"\nГруппы регионов:")
            print(f"  Низкая безработица (< {median_unemployment:.1f}%): {len(low_unemployment)} регионов")
            print(f"  Высокая безработица (≥ {median_unemployment:.1f}%): {len(high_unemployment)} регионов")
            
            # Сравниваем ВВП в двух группах
            if 'gdp_per_capita' in original_data.columns:
                if len(low_unemployment) >= 2 and len(high_unemployment) >= 2:
                    # Тест Манна-Уитни для независимых выборок
                    stat, p_value = stats.mannwhitneyu(
                        low_unemployment['gdp_per_capita'].dropna(),
                        high_unemployment['gdp_per_capita'].dropna(),
                        alternative='two-sided'
                    )
                    
                    result = {
                        'Показатель': 'Сравнение ВВП по группам безработицы',
                        'Статистика U': f"{stat:.2f}",
                        'p-value': f"{p_value:.4f}",
                        'Вывод': 'Различия есть' if p_value < 0.05 else 'Различий нет'
                    }
                    results.append(result)
                    
                    print(f"\nСравнение ВВП на душу населения:")
                    print(f"  Группа с низкой безработицей: среднее = {low_unemployment['gdp_per_capita'].mean():.0f} €")
                    print(f"  Группа с высокой безработицей: среднее = {high_unemployment['gdp_per_capita'].mean():.0f} €")
                    print(f"  Тест Манна-Уитни: U = {stat:.2f}, p = {p_value:.4f}")
                    print(f"  Вывод: {'Есть статистически значимые различия' if p_value < 0.05 else 'Нет статистически значимых различий'}")
    
    # 4. Регрессионный анализ (простой)
    if all(col in original_data.columns for col in ['gdp_per_capita', 'education_level']):
        print("\n" + "-" * 80)
        print("4. РЕГРЕССИОННЫЙ АНАЛИЗ")
        print("-" * 80)
        
        # Удаляем пропущенные значения
        clean_data = original_data[['gdp_per_capita', 'education_level']].dropna()
        
        if len(clean_data) >= 3:
            # Линейная регрессия
            x = clean_data['education_level'].values
            y = clean_data['gdp_per_capita'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            result = {
                'Показатель': 'Регрессия: ВВП ~ Образование',
                'Наклон (slope)': f"{slope:.2f}",
                'R²': f"{r_value**2:.4f}",
                'p-value': f"{p_value:.4f}",
                'Вывод': 'Значимая зависимость' if p_value < 0.05 else 'Незначимая зависимость'
            }
            results.append(result)
            
            print(f"\nЗависимость ВВП от уровня образования:")
            print(f"  Уравнение: ВВП = {intercept:.0f} + {slope:.0f} × Образование")
            print(f"  Коэффициент детерминации R² = {r_value**2:.4f}")
            print(f"  p-value = {p_value:.4f}")
            print(f"  Вывод: {'Есть значимая зависимость' if p_value < 0.05 else 'Нет значимой зависимости'}")
    
    # Сохраняем результаты в файл
    if results:
        results_df = pd.DataFrame(results)
        output_path = os.path.join(reports_dir, 'statistical_results.csv')
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Результаты тестов сохранены в: {output_path}")
        
        # Также сохраняем текстовый отчет
        txt_output_path = os.path.join(reports_dir, 'statistical_report.txt')
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write("СТАТИСТИЧЕСКИЙ ОТЧЕТ: Анализ регионов Франции\n")
            f.write("=" * 60 + "\n\n")
            
            for res in results:
                f.write(f"Показатель: {res.get('Показатель', '')}\n")
                for key, value in res.items():
                    if key != 'Показатель':
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"✓ Текстовый отчет сохранен в: {txt_output_path}")
    
    print("\n" + "=" * 80)
    print("СТАТИСТИЧЕСКИЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 80)
    
    return results