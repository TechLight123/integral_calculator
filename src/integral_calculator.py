import tkinter as tk
from tkinter import ttk, messagebox
from sympy import integrate, Symbol, sympify, diff
from decimal import Decimal, getcontext
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class IntegrationResult:
    """Класс для хранения результатов интегрирования"""
    value: Decimal
    points: int
    is_stable: bool = True

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
        
        # Поле для минимального размера интервала
        ttk.Label(main_frame, text="Мин. размер интервала:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.min_interval_var = tk.StringVar(value=str(self.config.min_interval_size))
        ttk.Entry(main_frame, textvariable=self.min_interval_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Кнопка вычисления
        ttk.Button(main_frame, text="Вычислить", command=self.calculate).grid(row=4, column=0, columnspan=3, pady=20)
        
        # Поле для вывода результата
        result_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="5")
        result_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.result_text = tk.Text(result_frame, height=15, width=70)
        self.result_text.grid(row=0, column=0, pady=5)
        
        # Добавляем примеры функций
        self._setup_examples(main_frame)

    def _setup_examples(self, main_frame):
        """Настройка примеров функций"""
        examples_frame = ttk.LabelFrame(main_frame, text="Примеры функций", padding="5")
        examples_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
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
            lower_bound = float(self.lower_bound_var.get())
            upper_bound = float(self.upper_bound_var.get())
            min_interval_size = float(self.min_interval_var.get())
            
            if lower_bound >= upper_bound:
                raise ValueError("Нижний предел должен быть меньше верхнего")
            if min_interval_size <= 0:
                raise ValueError("Минимальный размер интервала должен быть положительным")
                
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
            
            # Вычисляем точное значение интеграла
            exact_integral = integrate(expr, (x, lower_bound, upper_bound))
            exact_value = float(exact_integral.evalf())
            
            # Вычисляем интеграл с адаптивным разбиением
            result = calculate_adaptive(expr, x, lower_bound, upper_bound, self.config)
            
            # Выводим результаты
            self._display_results(expression, lower_bound, upper_bound, exact_value, result)
            
        except Exception as e:
            self._handle_error(e)

    def _display_results(self, expression: str, lower_bound: float, upper_bound: float, 
                        exact_value: float, result: IntegrationResult):
        """Отображение результатов вычисления"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Вычисление интеграла {expression} от {lower_bound} до {upper_bound}\n")
        self.result_text.insert(tk.END, "-" * 80 + "\n\n")
        self.result_text.insert(tk.END, f"1. Приближенное значение: {float(result.value):.15f}\n")
        self.result_text.insert(tk.END, f"2. Точное значение:       {exact_value:.15f}\n")
        self.result_text.insert(tk.END, f"3. Абсолютная ошибка:    {abs(exact_value - float(result.value)):.2e}\n")
        self.result_text.insert(tk.END, f"4. Относительная ошибка: {abs(exact_value - float(result.value))/abs(exact_value)*100:.2e}%\n")
        self.result_text.insert(tk.END, f"\n5. Параметры:\n")
        self.result_text.insert(tk.END, f"   - Минимальный размер интервала: {float(self.config.min_interval_size):.6f}\n")
        self.result_text.insert(tk.END, f"   - Общее количество точек: {result.points}\n")
        if not result.is_stable:
            self.result_text.insert(tk.END, "\nВнимание: обнаружены признаки нестабильности в вычислениях!\n")

    def _handle_error(self, error: Exception):
        """Обработка ошибок"""
        error_message = str(error)
        if "Invalid input" in error_message:
            messagebox.showerror("Ошибка", "Неверный формат функции. Проверьте синтаксис.")
        elif "Division by zero" in error_message:
            messagebox.showerror("Ошибка", "Деление на ноль невозможно.")
        else:
            messagebox.showerror("Ошибка", f"Ошибка при вычислении: {error_message}")

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
                f_val = Decimal(str(float(expr.subs(x, node_float).evalf(config.precision))))
                f_prime = Decimal(str(float(diff(expr, x).subs(x, node_float).evalf(config.precision))))
                f_double_prime = Decimal(str(float(diff(expr, x, 2).subs(x, node_float).evalf(config.precision))))
                
                if any(abs(val) > config.stability_threshold for val in [f_val, f_prime, f_double_prime]):
                    return None, num_points
                
                values.append((f_val, f_prime, f_double_prime))
            except (ValueError, TypeError, ZeroDivisionError):
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

def main():
    root = tk.Tk()
    app = IntegralCalculator(root)
    root.mainloop()

if __name__ == "__main__":
    main() 