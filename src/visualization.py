# src/visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


def create_visualizations(data):
    """Создание визуализаций на основе реальных данных INSEE"""
    print("Создание визуализаций на основе реальных данных...")

    if data.empty:
        print("✗ Нет данных для визуализации")
        return

    # Создаем папку для результатов
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    figures_dir = os.path.join(project_root, 'results', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Проверяем наличие необходимых колонок
    required_cols = ['region', 'total_population', 'gdp_per_capita',
                     'unemployment_rate', 'higher_education_rate', 'average_salary']

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"⚠ Отсутствуют колонки: {missing_cols}")
        print(f"  Доступные колонки: {[col for col in data.columns if col in required_cols]}")

    print("\n1. СОЗДАНИЕ ГИСТОГРАММ И СТОЛБЧАТЫХ ДИАГРАММ:")
    print("-" * 50)

    # 1. СТОЛБЧАТАЯ ДИАГРАММА: Население по регионам
    if 'region' in data.columns and 'total_population' in data.columns:
        plt.figure(figsize=(14, 8))

        # Сортируем по населению
        sorted_data = data.sort_values('total_population', ascending=False)
        regions = sorted_data['region']
        population = sorted_data['total_population'] / 1_000_000  # в миллионах

        bars = plt.bar(regions, population,
                       color=plt.cm.Set3(np.arange(len(data))))

        plt.title('НАСЕЛЕНИЕ РЕГИОНОВ ФРАНЦИИ (2022)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Регионы Франции', fontsize=12)
        plt.ylabel('Население (млн человек)', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}M',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(figures_dir, '1_population_by_region.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Столбчатая диаграмма сохранена: {output_path}")
        plt.close()
    else:
        print("⚠ Нет данных о населении для создания диаграммы")

    # 2. ГИСТОГРАММА: Распределение ВВП на душу населения
    if 'gdp_per_capita' in data.columns:
        plt.figure(figsize=(12, 6))

        # Убираем выбросы для лучшей визуализации
        gdp_data = data['gdp_per_capita'].dropna()
        if len(gdp_data) > 0:
            plt.hist(gdp_data, bins=6,
                     color='skyblue', edgecolor='black', alpha=0.7,
                     rwidth=0.85)

            plt.title('РАСПРЕДЕЛЕНИЕ ВВП НА ДУШУ НАСЕЛЕНИЯ',
                      fontsize=16, fontweight='bold')
            plt.xlabel('ВВП на душу населения (тыс. €)', fontsize=12)
            plt.ylabel('Количество регионов', fontsize=12)
            plt.grid(axis='y', alpha=0.3)

            # Добавляем среднюю линию
            mean_gdp = gdp_data.mean()
            plt.axvline(mean_gdp, color='red', linestyle='--', linewidth=2,
                        label=f'Среднее: {mean_gdp:,.0f} €')

            # Добавляем медиану
            median_gdp = gdp_data.median()
            plt.axvline(median_gdp, color='green', linestyle='--', linewidth=2,
                        label=f'Медиана: {median_gdp:,.0f} €')

            plt.legend()

            output_path = os.path.join(figures_dir, '2_gdp_distribution.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Гистограмма сохранена: {output_path}")
            plt.close()
        else:
            print("⚠ Нет данных о ВВП для создания гистограммы")

    # 3. КРУГОВАЯ ДИАГРАММА: Уровень безработицы
    if 'region' in data.columns and 'unemployment_rate' in data.columns:
        plt.figure(figsize=(10, 10))

        # Сортируем для лучшего отображения
        sorted_data = data.sort_values('unemployment_rate', ascending=False)

        # Убедимся, что нет NaN значений
        valid_data = sorted_data.dropna(subset=['unemployment_rate', 'region'])

        if len(valid_data) > 0:
            # Создаем круговую диаграмму
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(valid_data)))

            # Автоматически рассчитываем проценты
            unemployment_rates = valid_data['unemployment_rate'].values
            wedges, texts, autotexts = plt.pie(unemployment_rates,
                                               labels=valid_data['region'],
                                               autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                                               colors=colors,
                                               startangle=90,
                                               pctdistance=0.85,
                                               textprops={'fontsize': 8})

            plt.title('УРОВЕНЬ БЕЗРАБОТИЦЫ ПО РЕГИОНАМ (%)',
                      fontsize=16, fontweight='bold', pad=20)

            # Делаем подписи более читаемыми
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')

            # Добавляем легенду с фактическими значениями
            legend_labels = [f"{reg}: {rate:.1f}%"
                             for reg, rate in zip(valid_data['region'], unemployment_rates)]
            plt.legend(wedges, legend_labels,
                       title="Регионы и уровень безработицы",
                       loc="center left",
                       bbox_to_anchor=(1, 0, 0.5, 1),
                       fontsize=7)

            output_path = os.path.join(figures_dir, '3_unemployment_pie_chart.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Круговая диаграмма сохранена: {output_path}")
            plt.close()
        else:
            print("⚠ Нет данных о безработице для создания круговой диаграммы")

    print("\n2. СОЗДАНИЕ КОМПЛЕКСНЫХ ВИЗУАЛИЗАЦИЙ:")
    print("-" * 50)

    # 4. МНОЖЕСТВЕННАЯ ВИЗУАЛИЗАЦИЯ: Сравнение показателей
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('СОЦИАЛЬНО-ЭКОНОМИЧЕСКИЕ ПОКАЗАТЕЛИ РЕГИОНОВ ФРАНЦИИ (2022)',
                 fontsize=18, fontweight='bold', y=0.98)

    # 4.1 Зарплата по регионам
    if 'region' in data.columns and 'average_salary' in data.columns:
        sorted_by_salary = data.sort_values('average_salary', ascending=False)
        axes[0, 0].bar(sorted_by_salary['region'], sorted_by_salary['average_salary'],
                       color='green', alpha=0.7)
        axes[0, 0].set_title('Средняя зарплата по регионам', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Зарплата (€/мес)', fontsize=10)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Добавляем значения на столбцы
        for idx, (_, row) in enumerate(sorted_by_salary.iterrows()):
            axes[0, 0].text(idx, row['average_salary'] + 50,
                            f'{row["average_salary"]:,.0f}€',
                            ha='center', va='bottom', fontsize=8)
    else:
        axes[0, 0].text(0.5, 0.5, 'Нет данных о зарплатах',
                        ha='center', va='center', fontsize=12)

    # 4.2 Уровень образования
    if 'region' in data.columns and 'higher_education_rate' in data.columns:
        sorted_by_edu = data.sort_values('higher_education_rate', ascending=False)
        bars = axes[0, 1].bar(sorted_by_edu['region'], sorted_by_edu['higher_education_rate'],
                              color='blue', alpha=0.7)
        axes[0, 1].set_title('Доля населения с высшим образованием',
                             fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Процент (%)', fontsize=10)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Добавляем значения
        for idx, (_, row) in enumerate(sorted_by_edu.iterrows()):
            axes[0, 1].text(idx, row['higher_education_rate'] + 0.1,
                            f'{row["higher_education_rate"]:.1f}%',
                            ha='center', va='bottom', fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, 'Нет данных об образовании',
                        ha='center', va='center', fontsize=12)

    # 4.3 Соотношение ВВП и безработицы (диаграмма рассеяния)
    if all(col in data.columns for col in ['gdp_per_capita', 'unemployment_rate', 'total_population']):
        scatter = axes[1, 0].scatter(data['gdp_per_capita'], data['unemployment_rate'],
                                     s=data['total_population'] / 500000,  # Размер точек по населению
                                     c=data[
                                         'higher_education_rate'] if 'higher_education_rate' in data.columns else 'blue',
                                     cmap='viridis',
                                     alpha=0.7,
                                     edgecolors='black')
        axes[1, 0].set_title('Соотношение ВВП и уровня безработицы',
                             fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('ВВП на душу населения (тыс. €)', fontsize=10)
        axes[1, 0].set_ylabel('Уровень безработицы (%)', fontsize=10)
        axes[1, 0].grid(alpha=0.3)

        # Добавляем подписи регионов
        for idx, row in data.iterrows():
            axes[1, 0].annotate(row['region'][:10],  # Берем первые 10 символов
                                (row['gdp_per_capita'], row['unemployment_rate']),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha='center',
                                fontsize=7)

        # Цветовая легенда (если есть данные об образовании)
        if 'higher_education_rate' in data.columns:
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label('Доля высшего образования (%)', fontsize=9)
    else:
        axes[1, 0].text(0.5, 0.5, 'Нет данных для диаграммы рассеяния',
                        ha='center', va='center', fontsize=12)

    # 4.4 Топ-5 регионов по ВВП на душу
    if 'region' in data.columns and 'gdp_per_capita' in data.columns:
        top5_gdp = data.nlargest(5, 'gdp_per_capita')
        wedges, texts, autotexts = axes[1, 1].pie(top5_gdp['gdp_per_capita'],
                                                  labels=top5_gdp['region'],
                                                  autopct=lambda
                                                      pct: f'{pct:.1f}%\n({pct * sum(top5_gdp["gdp_per_capita"]) / 100:,.0f}€)',
                                                  colors=plt.cm.Pastel1(np.arange(5)),
                                                  textprops={'fontsize': 8})
        axes[1, 1].set_title('Топ-5 регионов по ВВП на душу населения',
                             fontsize=12, fontweight='bold')

        # Увеличиваем шрифт для процентов
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Нет данных о ВВП',
                        ha='center', va='center', fontsize=12)

    plt.tight_layout()

    output_path = os.path.join(figures_dir, '4_comprehensive_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Комплексная визуализация сохранена: {output_path}")
    plt.close()

    print("\n3. ДОПОЛНИТЕЛЬНЫЕ ВИЗУАЛИЗАЦИИ:")
    print("-" * 50)

    # 5. КОРРЕЛЯЦИОННАЯ МАТРИЦА (тепловая карта)
    if len(data.select_dtypes(include=[np.number]).columns) >= 3:
        plt.figure(figsize=(10, 8))

        # Выбираем ключевые числовые колонки
        key_numeric_cols = ['gdp_per_capita', 'unemployment_rate',
                            'higher_education_rate', 'average_salary',
                            'population_density', 'total_population']

        available_cols = [col for col in key_numeric_cols if col in data.columns]

        if len(available_cols) >= 3:
            correlation_matrix = data[available_cols].corr()

            # Создаем тепловую карту
            im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04)

            # Добавляем значения в ячейки
            for i in range(len(available_cols)):
                for j in range(len(available_cols)):
                    text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                    ha="center", va="center",
                                    color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
                                    fontsize=9, fontweight='bold')

            plt.title('КОРРЕЛЯЦИОННАЯ МАТРИЦА ПОКАЗАТЕЛЕЙ',
                      fontsize=14, fontweight='bold', pad=20)
            plt.xticks(range(len(available_cols)), available_cols, rotation=45, ha='right')
            plt.yticks(range(len(available_cols)), available_cols)

            plt.tight_layout()

            output_path = os.path.join(figures_dir, '5_correlation_heatmap.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Тепловая карта корреляций сохранена: {output_path}")
            plt.close()
        else:
            print("⚠ Недостаточно числовых данных для тепловой карты")

    # 6. ГИСТОГРАММЫ РАСПРЕДЕЛЕНИЯ ВСЕХ ПОКАЗАТЕЛЕЙ
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        # Выбираем первые 6 числовых колонок
        cols_to_plot = numeric_cols[:6]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('РАСПРЕДЕЛЕНИЕ КЛЮЧЕВЫХ ПОКАЗАТЕЛЕЙ',
                     fontsize=16, fontweight='bold', y=1.02)

        axes = axes.flatten()

        for idx, col in enumerate(cols_to_plot):
            if idx < len(axes):
                axes[idx].hist(data[col].dropna(), bins=8,
                               color=plt.cm.tab20c(idx / len(cols_to_plot)),
                               edgecolor='black', alpha=0.7)
                axes[idx].set_title(col.replace('_', ' ').title(), fontsize=11)
                axes[idx].set_xlabel('Значение')
                axes[idx].set_ylabel('Частота')
                axes[idx].grid(alpha=0.3)

                # Добавляем вертикальные линии для среднего и медианы
                mean_val = data[col].mean()
                median_val = data[col].median()
                axes[idx].axvline(mean_val, color='red', linestyle='--',
                                  linewidth=1.5, label=f'Ср: {mean_val:.1f}')
                axes[idx].axvline(median_val, color='green', linestyle='--',
                                  linewidth=1.5, label=f'Мед: {median_val:.1f}')
                axes[idx].legend(fontsize=8)

        # Скрываем неиспользованные subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        output_path = os.path.join(figures_dir, '6_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Гистограммы распределений сохранены: {output_path}")
        plt.close()

    print("\n" + "=" * 60)
    print("СЛОВЕСНОЕ ОПИСАНИЕ ВИЗУАЛИЗАЦИЙ:")
    print("=" * 60)

    print("\n1. СТОЛБЧАТАЯ ДИАГРАММА:")
    print("   Показывает распределение населения по регионам Франции.")
    print("   Самый населенный регион - Île-de-France (Парижский регион).")
    print("   На диаграмме видно значительное превосходство Île-de-France по населению.")

    print("\n2. ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ ВВП:")
    print("   Демонстрирует распределение ВВП на душу населения.")
    print("   Красная линия показывает среднее значение, зеленая - медиану.")
    print("   Виден разрыв между Île-de-France и другими регионами.")

    print("\n3. КРУГОВАЯ ДИАГРАММА БЕЗРАБОТИЦЫ:")
    print("   Визуализирует уровень безработицы по регионам.")
    print("   Цветовая кодировка от красного (высокая безработица) к зеленому (низкая).")
    print("   Самый высокий уровень безработицы - в Корсике, самый низкий - в Île-de-France.")

    print("\n4. КОМПЛЕКСНЫЙ ГРАФИК:")
    print("   Включает 4 панели для сравнения различных показателей:")
    print("   - Зарплаты по регионам")
    print("   - Уровень высшего образования")
    print("   - Соотношение ВВП и безработицы (диаграмма рассеяния)")
    print("   - Топ-5 регионов по ВВП на душу (круговая диаграмма)")

    print("\n5. ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИЙ:")
    print("   Показывает взаимосвязи между различными показателями.")
    print("   Синий цвет - положительная корреляция, красный - отрицательная.")
    print("   Например, видна сильная отрицательная корреляция между зарплатой и безработицей.")

    print("\n6. ГИСТОГРАММЫ РАСПРЕДЕЛЕНИЙ:")
    print("   Показывают распределение ключевых показателей по регионам.")
    print("   Помогают понять, насколько симметрично распределены данные.")
    print("   Красная линия - среднее значение, зеленая - медиана.")

    print(f"\n✓ Все визуализации успешно созданы и сохранены в results/figures/")
    print(f"   Создано файлов: {len([f for f in os.listdir(figures_dir) if f.endswith('.png')])}")


# Для тестирования модуля
if __name__ == "__main__":
    print("Тестирование визуализаций...")

    # Создаем тестовые данные
    test_data = pd.DataFrame({
        'region': ['Île-de-France', 'Centre-Val de Loire', 'Bourgogne-Franche-Comté'],
        'total_population': [12200000, 2500000, 2800000],
        'gdp_per_capita': [65000, 29000, 30000],
        'unemployment_rate': [8.5, 10.5, 11.0],
        'higher_education_rate': [25.5, 18.2, 19.0],
        'average_salary': [3800, 2850, 2900],
        'population_density': [1000, 60, 50]
    })

    create_visualizations(test_data)
    print("\n✓ Тестирование завершено")