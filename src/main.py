# France_Regions_Socioeconomic_Analysis
# src/main.py

import sys
import os

# Добавляем текущую директорию в путь Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 80)
    print("\nАНАЛИЗ СОЦИАЛЬНО-ЭКОНОМИЧЕСКИХ ПОКАЗАТЕЛЕЙ РЕГИОНОВ ФРАНЦИИ")
    print("=" * 80)

    print("\nПути сохранения:")
    print("1. Обработанные данные: data/processed/")
    print("2. Графики: results/figures/")
    print("3. Статистические отчеты: results/reports/")
    print("4. Результаты тестов: results/tables/")

    # Импортируем модули
    try:
        from data_loading import load_data
        from data_preprocessing import preprocess_data
        from data_aggregation import aggregate_data
        from visualization import create_visualizations
        from statistical_tests import perform_statistical_tests  # ИСПРАВЛЕНО

        print("\n" + "=" * 80)
        print("ЭТАП 1: ЗАГРУЗКА ДАННЫХ")
        print("=" * 80)

        # Загружаем данные
        data = load_data()

        if data.empty:
            print("✗ Нет данных для анализа. Создаем тестовые данные...")
            # Создаем тестовые данные для демонстрации
            import pandas as pd
            import numpy as np

            # Создаем демо-данные регионов Франции
            regions = [
                'Île-de-France', 'Auvergne-Rhône-Alpes', 'Nouvelle-Aquitaine',
                'Occitanie', 'Hauts-de-France', 'Grand Est',
                'Pays de la Loire', 'Bretagne', 'Normandie',
                'Provence-Alpes-Côte d\'Azur', 'Bourgogne-Franche-Comté',
                'Centre-Val de Loire', 'Corse'
            ]

            data = pd.DataFrame({
                'region': regions,
                'population': np.random.randint(500000, 12000000, len(regions)),
                'gdp_per_capita': np.random.randint(20000, 50000, len(regions)),
                'unemployment_rate': np.random.uniform(5, 15, len(regions)),
                'education_level': np.random.uniform(60, 95, len(regions)),
                'average_salary': np.random.randint(2500, 4500, len(regions))
            })
            print("✓ Созданы демонстрационные данные")

        print("✓ Данные успешно загружены")

        # Обрабатываем данные
        print("\n" + "=" * 80)
        print("ЭТАП 2: ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ")
        print("=" * 80)
        processed_data = preprocess_data(data)
        print("✓ Данные успешно обработаны")

        # Агрегируем данные
        print("\n" + "=" * 80)
        print("ЭТАП 3: АГРЕГАЦИЯ ДАННЫХ")
        print("=" * 80)
        aggregated_data = aggregate_data(processed_data)
        print("✓ Данные успешно агрегированы")

        # Создаем визуализации
        print("\n" + "=" * 80)
        print("ЭТАП 4: ВИЗУАЛИЗАЦИЯ")
        print("=" * 80)
        create_visualizations(aggregated_data)
        print("✓ Визуализации созданы")

        # Выполняем статистические тесты
        print("\n" + "=" * 80)
        print("ЭТАП 5: СТАТИСТИЧЕСКИЕ ТЕСТЫ")
        print("=" * 80)
        perform_statistical_tests(aggregated_data)
        print("✓ Статистические тесты выполнены")

        print("\n" + "=" * 80)
        print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
        print("=" * 80)

    except ImportError as e:
        print(f"\nОШИБКА ИМПОРТА: {e}")
        print("Убедитесь, что все модули находятся в папке src/")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nНЕПРЕДВИДЕННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()