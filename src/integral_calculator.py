import tkinter as tk
from tkinter import ttk, messagebox
from sympy import integrate, Symbol, sympify, diff, lambdify
from decimal import Decimal, getcontext
from typing import Tuple, Optional, List, Callable, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from enum import Enum
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class IntegrationMethod(Enum):
    ADAPTIVE = "Адаптивный метод"
    TRAPEZOID = "Метод трапеций"
    SIMPSON = "Метод Симпсона"
    GAUSS = "Метод Гаусса"

@dataclass
class IntegrationResult:
    """Класс для хранения результатов интегрирования"""
    value: Decimal
    points: int
    is_stable: bool = True
    method: IntegrationMethod = IntegrationMethod.ADAPTIVE

class IntegrationConfig:
    """Класс для хранения конфигурации интегрирования"""
    def __init__(self, 
                 min_interval_size: float = 0.5,
                 max_depth: int = 10,
                 precision: int = 50,
                 max_points: int = 150,
                 min_points: int = 20,
                 stability_threshold: float = 1e10,
                 convergence_threshold: float = 0.01):
        self.min_interval_size = Decimal(str(min_interval_size))
        self.max_depth = max_depth
        self.precision = precision
        self.max_points = max_points
        self.min_points = min_points
        self.stability_threshold = Decimal(str(stability_threshold))
        self.convergence_threshold = Decimal(str(convergence_threshold))
        getcontext().prec = precision

class IntegralCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Калькулятор интегралов")
        self.root.geometry("800x600")
        self.config = IntegrationConfig()
        self._setup_ui()
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    def __del__(self):
        self._executor.shutdown(wait=True)

    def _get_cached_result(self, key: str) -> Optional[IntegrationResult]:
        """Получение результата из кэша"""
        with self._cache_lock:
            return self._cache.get(key)

    def _set_cached_result(self, key: str, result: IntegrationResult):
        """Сохранение результата в кэш"""
        with self._cache_lock:
            self._cache[key] = result

    def _clear_cache(self):
        """Очистка кэша"""
        with self._cache_lock:
            self._cache.clear()

    def _calculate_with_method(self, method: IntegrationMethod, expr, x, lower_bound: float, upper_bound: float) -> IntegrationResult:
        """Вычисление интеграла указанным методом"""
        try:
            cache_key = f"{method.value}_{expr}_{lower_bound}_{upper_bound}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result

            # Проверяем корректность функции
            try:
                # Пробуем вычислить значение функции в начальной точке
                test_val = expr.subs(x, lower_bound).evalf()
                if isinstance(test_val, complex):
                    raise ValueError("Функция возвращает комплексные значения")
            except Exception as e:
                raise ValueError(f"Ошибка в функции: {str(e)}")

            if method == IntegrationMethod.ADAPTIVE:
                result = calculate_adaptive(expr, x, lower_bound, upper_bound, self.config)
            elif method == IntegrationMethod.TRAPEZOID:
                result = calculate_trapezoid(expr, x, lower_bound, upper_bound, self.config)
            elif method == IntegrationMethod.SIMPSON:
                result = calculate_simpson(expr, x, lower_bound, upper_bound, self.config)
            else:  # GAUSS
                result = calculate_gauss(expr, x, lower_bound, upper_bound, self.config)
            
            result.method = method
            self._set_cached_result(cache_key, result)
            return result
        except Exception as e:
            raise ValueError(f"Ошибка при вычислении методом {method.value}: {str(e)}")

    def _setup_ui(self):
        """Настройка пользовательского интерфейса"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Поле для ввода функции
        ttk.Label(main_frame, text="Функция f(x):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.expression_var = tk.StringVar()
        self.expression_entry = ttk.Entry(main_frame, textvariable=self.expression_var, width=50)
        self.expression_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Поля для пределов интегрирования
        ttk.Label(main_frame, text="Нижний предел:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.lower_bound_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.lower_bound_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(main_frame, text="Верхний предел:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.upper_bound_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.upper_bound_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Выбор метода интегрирования
        ttk.Label(main_frame, text="Метод интегрирования:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.method_var = tk.StringVar(value=IntegrationMethod.ADAPTIVE.value)
        self.method_combo = ttk.Combobox(main_frame, textvariable=self.method_var, 
                                       values=[method.value for method in IntegrationMethod],
                                       state="readonly", width=20)
        self.method_combo.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Поле для минимального размера интервала
        ttk.Label(main_frame, text="Мин. размер интервала:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.min_interval_var = tk.StringVar(value=str(self.config.min_interval_size))
        ttk.Entry(main_frame, textvariable=self.min_interval_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Кнопки
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        ttk.Button(buttons_frame, text="Вычислить", command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Построить график", command=self.plot_graph).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Сравнить методы", command=self.compare_methods).pack(side=tk.LEFT, padx=5)
        
        # Поле для вывода результата
        result_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="5")
        result_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.result_text = tk.Text(result_frame, height=15, width=70)
        self.result_text.grid(row=0, column=0, pady=5)
        
        # Добавляем примеры функций
        self._setup_examples(main_frame)

    def _setup_examples(self, main_frame):
        """Настройка примеров функций"""
        examples_frame = ttk.LabelFrame(main_frame, text="Примеры функций", padding="5")
        examples_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        examples = [
            "x*cos(7*(x**2)) (тестовая функция)",
            "x**2 (квадрат x)",
            "x**3 (куб x)",
            "sin(x) (синус)",
            "cos(x) (косинус)",
            "exp(x) (экспонента)",
            "log(x) (логарифм)",
            "x**4 + 2*x**2 - x + 1 (полином)"
        ]
        
        for i, example in enumerate(examples):
            ttk.Label(examples_frame, text=example).grid(row=i//2, column=i%2, sticky=tk.W, padx=5, pady=2)

    def _validate_input(self) -> Tuple[float, float, float]:
        """Валидация входных данных"""
        try:
            # Проверяем, что поля не пустые
            if not self.expression_var.get().strip():
                raise ValueError("Введите функцию")
            if not self.lower_bound_var.get().strip():
                raise ValueError("Введите нижний предел")
            if not self.upper_bound_var.get().strip():
                raise ValueError("Введите верхний предел")
            if not self.min_interval_var.get().strip():
                raise ValueError("Введите минимальный размер интервала")

            # Преобразуем и проверяем значения
            try:
                lower_bound = float(self.lower_bound_var.get().replace(',', '.'))
                upper_bound = float(self.upper_bound_var.get().replace(',', '.'))
                min_interval_size = float(self.min_interval_var.get().replace(',', '.'))
            except ValueError:
                raise ValueError("Некорректный формат числа")

            # Проверяем корректность значений
            if lower_bound >= upper_bound:
                raise ValueError("Нижний предел должен быть меньше верхнего")
            if min_interval_size <= 0:
                raise ValueError("Минимальный размер интервала должен быть положительным")
            if min_interval_size >= (upper_bound - lower_bound):
                raise ValueError("Минимальный размер интервала должен быть меньше длины интервала интегрирования")
                
            return lower_bound, upper_bound, min_interval_size
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            raise

    def calculate(self):
        """Основной метод вычисления интеграла"""
        try:
            lower_bound, upper_bound, min_interval_size = self._validate_input()
            self.config.min_interval_size = Decimal(str(min_interval_size))
            
            expression = self.expression_var.get()
            x = Symbol('x')
            expr = sympify(expression)
            
            # Выбираем метод интегрирования
            selected_method = next(method for method in IntegrationMethod 
                                if method.value == self.method_var.get())
            
            # Вычисляем интеграл выбранным методом
            result = self._calculate_with_method(selected_method, expr, x, lower_bound, upper_bound)
            
            # Выводим результаты
            self._display_results(expression, lower_bound, upper_bound, result)
            
        except Exception as e:
            self._handle_error(e)

    def _display_results(self, expression: str, lower_bound: float, upper_bound: float, 
                        result: IntegrationResult):
        """Отображение результатов вычисления"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Вычисление интеграла {expression} от {lower_bound} до {upper_bound}\n")
        self.result_text.insert(tk.END, "-" * 80 + "\n\n")
        self.result_text.insert(tk.END, f"1. Метод интегрирования: {result.method.value}\n")
        self.result_text.insert(tk.END, f"2. Приближенное значение: {float(result.value):.15f}\n")
        self.result_text.insert(tk.END, f"\n3. Параметры:\n")
        self.result_text.insert(tk.END, f"   - Минимальный размер интервала: {float(self.config.min_interval_size):.6f}\n")
        self.result_text.insert(tk.END, f"   - Общее количество точек: {result.points}\n")
        if not result.is_stable:
            self.result_text.insert(tk.END, "\nВнимание: обнаружены признаки нестабильности в вычислениях!\n")

    def _handle_error(self, error: Exception):
        """Обработка ошибок"""
        error_message = str(error)
        if isinstance(error, ValueError):
            messagebox.showerror("Ошибка", error_message)
        elif "Invalid input" in error_message:
            messagebox.showerror("Ошибка", "Неверный формат функции. Проверьте синтаксис.")
        elif "Division by zero" in error_message:
            messagebox.showerror("Ошибка", "Деление на ноль невозможно.")
        elif "complex" in error_message.lower():
            messagebox.showerror("Ошибка", "Функция возвращает комплексные значения.")
        elif "nan" in error_message.lower() or "inf" in error_message.lower():
            messagebox.showerror("Ошибка", "Функция возвращает недопустимые значения (NaN или Inf).")
        else:
            messagebox.showerror("Ошибка", f"Ошибка при вычислении: {error_message}")

    def plot_graph(self):
        """Построение графика функции"""
        try:
            # Получаем входные данные
            expression = self.expression_var.get()
            lower_bound = float(self.lower_bound_var.get())
            upper_bound = float(self.upper_bound_var.get())
            
            # Создаем символьную функцию
            x = Symbol('x')
            expr = sympify(expression)
            
            # Создаем числовую функцию для построения графика
            f = lambda x_val: float(expr.subs(x, x_val))
            
            # Создаем массив точек для построения графика
            x_vals = np.linspace(lower_bound, upper_bound, 1000)
            y_vals = [f(x) for x in x_vals]
            
            # Создаем новое окно для графика
            graph_window = tk.Toplevel(self.root)
            graph_window.title("График функции")
            graph_window.geometry("800x600")
            
            # Создаем фигуру matplotlib
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            # Строим график
            ax.plot(x_vals, y_vals, 'b-', label=f'f(x) = {expression}')
            
            # Добавляем сетку и легенду
            ax.grid(True)
            ax.legend()
            
            # Добавляем заголовки
            ax.set_title('График функции')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            
            # Создаем холст для отображения графика
            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении графика: {str(e)}")

    def compare_methods(self):
        """Сравнение всех методов интегрирования"""
        try:
            lower_bound, upper_bound, min_interval_size = self._validate_input()
            self.config.min_interval_size = Decimal(str(min_interval_size))
            
            expression = self.expression_var.get()
            x = Symbol('x')
            expr = sympify(expression)
            
            # Создаем новое окно для сравнения
            compare_window = tk.Toplevel(self.root)
            compare_window.title("Сравнение методов интегрирования")
            compare_window.geometry("800x600")
            
            # Создаем текстовое поле для результатов
            result_text = tk.Text(compare_window, height=30, width=90)
            result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            # Заголовок
            result_text.insert(tk.END, f"Сравнение методов интегрирования для функции {expression}\n")
            result_text.insert(tk.END, f"Интервал интегрирования: [{lower_bound}, {upper_bound}]\n")
            result_text.insert(tk.END, "-" * 100 + "\n\n")
            
            # Сравниваем все методы параллельно
            futures = []
            for method in IntegrationMethod:
                future = self._executor.submit(
                    self._calculate_with_method,
                    method, expr, x, lower_bound, upper_bound
                )
                futures.append((method, future))
            
            # Собираем результаты
            results = []
            for method, future in futures:
                try:
                    result = future.result(timeout=30)  # таймаут 30 секунд
                    results.append(result)
                except Exception as e:
                    result_text.insert(tk.END, f"Ошибка при вычислении методом {method.value}: {str(e)}\n\n")
            
            # Выводим результаты
            for i, result in enumerate(results, 1):
                result_text.insert(tk.END, f"{i}. {result.method.value}:\n")
                result_text.insert(tk.END, f"   Приближенное значение: {float(result.value):.15f}\n")
                result_text.insert(tk.END, f"   Количество точек:     {result.points}\n")
                if not result.is_stable:
                    result_text.insert(tk.END, "   Внимание: обнаружены признаки нестабильности!\n")
                result_text.insert(tk.END, "\n")
            
            # Делаем текст только для чтения
            result_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self._handle_error(e)

@lru_cache(maxsize=128)
def estimate_complexity(expr, x, start: float, end: float, config: IntegrationConfig) -> int:
    """Оценивает сложность функции на заданном интервале"""
    first_derivative = diff(expr, x)
    second_derivative = diff(expr, x, 2)
    
    mid = (start + end) / 2
    d1_start = abs(float(first_derivative.subs(x, start).evalf()))
    d1_mid = abs(float(first_derivative.subs(x, mid).evalf()))
    d1_end = abs(float(first_derivative.subs(x, end).evalf()))
    
    d2_start = abs(float(second_derivative.subs(x, start).evalf()))
    d2_mid = abs(float(second_derivative.subs(x, mid).evalf()))
    d2_end = abs(float(second_derivative.subs(x, end).evalf()))
    
    max_d1 = max(d1_start, d1_mid, d1_end)
    max_d2 = max(d2_start, d2_mid, d2_end)
    
    interval_size = end - start
    complexity = (max_d1 + max_d2/10) * interval_size
    points = max(config.min_points, min(config.max_points, 
                                      int(config.min_points * complexity / 10)))
    return points

def calculate_subinterval(expr, x, start: float, end: float, N: int, 
                         config: IntegrationConfig) -> IntegrationResult:
    """Вычисляет интеграл на подинтервале с контролем сходимости"""
    if N < config.min_points:
        N = config.min_points
    
    h1 = Decimal(str(end - start)) / Decimal(str(N))
    h2 = h1 / Decimal('2')
    N2 = N * 2
    
    def compute_with_step(h: Decimal, num_points: int) -> Optional[Tuple[Decimal, int]]:
        nodes = [Decimal(str(start)) + i*h for i in range(num_points+1)]
        values = []
        
        for node in nodes:
            node_float = float(node)
            try:
                # Вычисляем значения функции и производных
                f_val = expr.subs(x, node_float).evalf(config.precision)
                f_prime = diff(expr, x).subs(x, node_float).evalf(config.precision)
                f_double_prime = diff(expr, x, 2).subs(x, node_float).evalf(config.precision)
                
                # Проверяем, что все значения являются действительными числами
                if any(isinstance(val, complex) for val in [f_val, f_prime, f_double_prime]):
                    print(f"Предупреждение: обнаружено комплексное число в точке {node_float}")
                    return None, num_points
                
                # Конвертируем в Decimal
                f_val = Decimal(str(float(f_val)))
                f_prime = Decimal(str(float(f_prime)))
                f_double_prime = Decimal(str(float(f_double_prime)))
                
                if any(abs(val) > config.stability_threshold for val in [f_val, f_prime, f_double_prime]):
                    return None, num_points
                
                values.append((f_val, f_prime, f_double_prime))
            except (ValueError, TypeError, ZeroDivisionError) as e:
                print(f"Ошибка при вычислении в точке {node_float}: {str(e)}")
                return None, num_points
        
        approx_value = Decimal('0')
        for i in range(num_points):
            terms = [
                h/Decimal('2') * values[i][0],
                h*h/Decimal('10') * values[i][1],
                h*h*h/Decimal('120') * values[i][2],
                h/Decimal('2') * values[i+1][0],
                -h*h/Decimal('10') * values[i+1][1],
                h*h*h/Decimal('120') * values[i+1][2]
            ]
            
            if any(abs(term) > config.stability_threshold for term in terms):
                return None, num_points
            
            approx_value += sum(terms)
        
        return approx_value, num_points
    
    result1, points1 = compute_with_step(h1, N)
    if result1 is None:
        return IntegrationResult(Decimal('0'), points1, False)
    
    result2, points2 = compute_with_step(h2, N2)
    if result2 is None:
        return IntegrationResult(Decimal('0'), points2, False)
    
    rel_diff = abs((result2 - result1) / (result2 if abs(result2) > Decimal('1e-10') else Decimal('1')))
    if rel_diff > config.convergence_threshold:
        return IntegrationResult(Decimal('0'), N2, False)
    
    return IntegrationResult(result2, points2)

def calculate_adaptive(expr, x, start: float, end: float, 
                      config: IntegrationConfig, depth: int = 0) -> IntegrationResult:
    """Рекурсивно вычисляет интеграл с адаптивным разбиением"""
    if depth >= config.max_depth:
        print(f"Достигнута максимальная глубина рекурсии на интервале [{start:.2f}, {end:.2f}]")
        return IntegrationResult(Decimal('0'), 0, False)
    
    interval_size = end - start
    if interval_size < float(config.min_interval_size):
        N = estimate_complexity(expr, x, start, end, config)
        result = calculate_subinterval(expr, x, start, end, N, config)
        if result.is_stable:
            print(f"Малый интервал [{start:.2f}, {end:.2f}]: {result.points} точек, значение = {float(result.value):.6f}")
        else:
            print(f"Предупреждение: нестабильный результат на интервале [{start:.2f}, {end:.2f}]")
        return result
    
    N = estimate_complexity(expr, x, start, end, config)
    result = calculate_subinterval(expr, x, start, end, N, config)
    if result.is_stable and N < 80:
        print(f"Простой интервал [{start:.2f}, {end:.2f}]: {result.points} точек, значение = {float(result.value):.6f}")
        return result
    
    print(f"Разбиваем интервал [{start:.2f}, {end:.2f}]")
    third = interval_size / 3
    points1 = start + third
    points2 = end - third
    
    results = []
    total_points = 0
    is_stable = True
    
    for sub_start, sub_end in [(start, points1), (points1, points2), (points2, end)]:
        sub_result = calculate_adaptive(expr, x, sub_start, sub_end, config, depth + 1)
        results.append(sub_result)
        total_points += sub_result.points
        is_stable &= sub_result.is_stable
    
    total_result = sum(r.value for r in results)
    if abs(total_result) > config.stability_threshold:
        print(f"Предупреждение: нестабильный результат при объединении интервалов [{start:.2f}, {end:.2f}]")
        return IntegrationResult(Decimal('0'), total_points, False)
    
    return IntegrationResult(total_result, total_points, is_stable)

def calculate_trapezoid(expr, x, start: float, end: float, config: IntegrationConfig) -> IntegrationResult:
    """Вычисление интеграла методом трапеций"""
    try:
        N = estimate_complexity(expr, x, start, end, config)
        h = Decimal(str(end - start)) / Decimal(str(N))
        
        # Создаем числовую функцию для быстрого вычисления
        f = lambdify(x, expr, 'numpy')
        x_vals = np.linspace(start, end, N+1)
        
        try:
            y_vals = f(x_vals)
        except Exception as e:
            raise ValueError(f"Ошибка при вычислении значений функции: {str(e)}")
        
        # Проверяем на комплексные числа и нестабильность
        if np.any(np.iscomplex(y_vals)):
            raise ValueError("Функция возвращает комплексные значения")
        if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
            raise ValueError("Функция возвращает недопустимые значения (NaN или Inf)")
        
        # Нормализуем значения для улучшения стабильности
        max_abs_val = np.max(np.abs(y_vals))
        if max_abs_val > float(config.stability_threshold):
            y_vals = y_vals / max_abs_val
            scale_factor = Decimal(str(max_abs_val))
        else:
            scale_factor = Decimal('1')
        
        # Вычисляем интеграл с повышенной точностью
        result = Decimal('0')
        for i in range(N):
            x1 = Decimal(str(x_vals[i]))
            x2 = Decimal(str(x_vals[i+1]))
            y1 = Decimal(str(float(y_vals[i])))
            y2 = Decimal(str(float(y_vals[i+1])))
            
            # Используем формулу трапеций с повышенной точностью
            segment = (x2 - x1) * (y1 + y2) / Decimal('2')
            result += segment
        
        # Применяем масштабный коэффициент
        result *= scale_factor
        
        return IntegrationResult(result, N, True, IntegrationMethod.TRAPEZOID)
    except Exception as e:
        return IntegrationResult(Decimal('0'), N, False, IntegrationMethod.TRAPEZOID)

def calculate_simpson(expr, x, start: float, end: float, config: IntegrationConfig) -> IntegrationResult:
    """Вычисление интеграла методом Симпсона"""
    try:
        N = estimate_complexity(expr, x, start, end, config)
        if N % 2 == 1:
            N += 1
        h = Decimal(str(end - start)) / Decimal(str(N))
        
        # Создаем числовую функцию для быстрого вычисления
        f = lambdify(x, expr, 'numpy')
        x_vals = np.linspace(start, end, N+1)
        
        try:
            y_vals = f(x_vals)
        except Exception as e:
            raise ValueError(f"Ошибка при вычислении значений функции: {str(e)}")
        
        # Проверяем на комплексные числа и нестабильность
        if np.any(np.iscomplex(y_vals)):
            raise ValueError("Функция возвращает комплексные значения")
        if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
            raise ValueError("Функция возвращает недопустимые значения (NaN или Inf)")
        
        # Нормализуем значения для улучшения стабильности
        max_abs_val = np.max(np.abs(y_vals))
        if max_abs_val > float(config.stability_threshold):
            y_vals = y_vals / max_abs_val
            scale_factor = Decimal(str(max_abs_val))
        else:
            scale_factor = Decimal('1')
        
        # Вычисляем интеграл с повышенной точностью
        result = Decimal('0')
        for i in range(0, N, 2):
            x0 = Decimal(str(x_vals[i]))
            x1 = Decimal(str(x_vals[i+1]))
            x2 = Decimal(str(x_vals[i+2]))
            y0 = Decimal(str(float(y_vals[i])))
            y1 = Decimal(str(float(y_vals[i+1])))
            y2 = Decimal(str(float(y_vals[i+2])))
            
            # Используем формулу Симпсона с повышенной точностью
            segment = (x2 - x0) * (y0 + Decimal('4') * y1 + y2) / Decimal('6')
            result += segment
        
        # Применяем масштабный коэффициент
        result *= scale_factor
        
        return IntegrationResult(result, N, True, IntegrationMethod.SIMPSON)
    except Exception as e:
        return IntegrationResult(Decimal('0'), N, False, IntegrationMethod.SIMPSON)

def calculate_gauss(expr, x, start: float, end: float, config: IntegrationConfig) -> IntegrationResult:
    """Вычисление интеграла методом Гаусса"""
    try:
        N = estimate_complexity(expr, x, start, end, config)
        if N % 2 == 1:
            N += 1
        
        # Коэффициенты и узлы для квадратурной формулы Гаусса
        gauss_points = [
            (Decimal('0.5773502691896257'), Decimal('1.0')),
            (Decimal('-0.5773502691896257'), Decimal('1.0')),
            (Decimal('0.7745966692414834'), Decimal('0.5555555555555556')),
            (Decimal('0.0'), Decimal('0.8888888888888888')),
            (Decimal('-0.7745966692414834'), Decimal('0.5555555555555556'))
        ]
        
        # Создаем числовую функцию для быстрого вычисления
        f = lambdify(x, expr, 'numpy')
        
        a = Decimal(str(start))
        b = Decimal(str(end))
        result = Decimal('0')
        max_abs_val = Decimal('0')
        
        # Сначала находим максимальное значение для нормализации
        for i in range(N):
            x1 = a + i * (b - a) / Decimal(str(N))
            x2 = a + (i + 1) * (b - a) / Decimal(str(N))
            h = (x2 - x1) / Decimal('2')
            
            for point, _ in gauss_points:
                try:
                    x_val = float(x1 + h * (point + Decimal('1')))
                    f_val = f(x_val)
                    
                    if np.iscomplex(f_val):
                        raise ValueError("Функция возвращает комплексные значения")
                    if np.isnan(f_val) or np.isinf(f_val):
                        raise ValueError("Функция возвращает недопустимые значения (NaN или Inf)")
                    
                    abs_val = Decimal(str(abs(float(f_val))))
                    max_abs_val = max(max_abs_val, abs_val)
                except Exception as e:
                    raise ValueError(f"Ошибка при вычислении в точке {x_val}: {str(e)}")
        
        # Нормализуем значения если необходимо
        scale_factor = Decimal('1')
        if max_abs_val > config.stability_threshold:
            scale_factor = max_abs_val
            max_abs_val = Decimal('1')
        
        # Вычисляем интеграл с нормализованными значениями
        for i in range(N):
            x1 = a + i * (b - a) / Decimal(str(N))
            x2 = a + (i + 1) * (b - a) / Decimal(str(N))
            h = (x2 - x1) / Decimal('2')
            
            for point, weight in gauss_points:
                try:
                    x_val = float(x1 + h * (point + Decimal('1')))
                    f_val = f(x_val)
                    
                    # Нормализуем значение
                    if scale_factor != Decimal('1'):
                        f_val = float(f_val) / float(scale_factor)
                    
                    result += h * weight * Decimal(str(float(f_val)))
                except Exception as e:
                    raise ValueError(f"Ошибка при вычислении в точке {x_val}: {str(e)}")
        
        # Применяем масштабный коэффициент
        result *= scale_factor
        
        return IntegrationResult(result, N, True, IntegrationMethod.GAUSS)
    except Exception as e:
        return IntegrationResult(Decimal('0'), N, False, IntegrationMethod.GAUSS)

def main():
    root = tk.Tk()
    app = IntegralCalculator(root)
    root.mainloop()

if __name__ == "__main__":
    main() 