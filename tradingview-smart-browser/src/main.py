"""
Умный браузер для анализа графиков TradingView
С функциями:
- Захват скриншотов графика
- Распознавание паттернов (тренды, уровни поддержки/сопротивления)
- AI-анализ технического состояния
- Сохранение истории анализов
"""

import sys
import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSplitter, QTextEdit, QFileDialog,
    QMessageBox, QToolBar, QStatusBar, QProgressBar, QLineEdit
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QUrl, QTimer, QSize, QThread, pyqtSignal, QRect
from PyQt6.QtGui import QIcon, QAction, QPixmap, QImage, QScreen
import numpy as np
import asyncio


class ScreenshotWorker(QThread):
    """Worker для захвата скриншотов через mss (захват области экрана)"""
    finished = pyqtSignal(str)  # Путь к файлу или ошибка
    
    def __init__(self, browser_window):
        super().__init__()
        self.browser_window = browser_window
    
    def run(self):
        try:
            import mss
            import mss.tools
            from PIL import Image
            import numpy as np
            
            # Получаем координаты и размер виджета браузера
            browser_widget = self.browser_window.browser
            if not browser_widget:
                self.finished.emit("ERROR: Browser widget not found")
                return
            
            # Конвертируем координаты в экранные
            screen = browser_widget.screen()
            if not screen:
                self.finished.emit("ERROR: No screen found")
                return
            
            geometry = browser_widget.geometry()
            global_geometry = browser_widget.mapToGlobal(geometry.topLeft())
            
            # Создаем область для захвата
            with mss.mss() as sct:
                monitor = {
                    "left": global_geometry.x(),
                    "top": global_geometry.y(),
                    "width": geometry.width(),
                    "height": geometry.height()
                }
                
                # Делаем скриншот
                screenshot = sct.grab(monitor)
                
                # Создаем директорию для скриншотов
                screenshot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'screenshots')
                os.makedirs(screenshot_dir, exist_ok=True)
                
                # Генерируем имя файла
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tradingview_{timestamp}.png"
                filepath = os.path.join(screenshot_dir, filename)
                
                # Конвертируем BGRA в RGB и сохраняем через PIL
                # mss возвращает данные в формате BGRA, нужно конвертировать в RGB
                img_array = np.frombuffer(screenshot.bgra, dtype=np.uint8).reshape(
                    screenshot.size[1], screenshot.size[0], 4
                )
                # Меняем местами синий и красный каналы (BGRA -> RGB), отбрасываем alpha
                img_rgb = Image.fromarray(img_array[:, :, [2, 1, 0]], mode='RGB')
                img_rgb.save(filepath, 'PNG')
                
                # Проверяем результат
                if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                    self.finished.emit(filepath)
                else:
                    self.finished.emit(f"ERROR: File empty or too small: {filepath}")
                    
        except Exception as e:
            self.finished.emit(f"ERROR: {str(e)}")


class ChartAnalyzer:
    """Анализатор графиков с компьютерным зрением"""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_screenshot(self, image_path: str) -> dict:
        """Анализирует скриншот графика"""
        try:
            import cv2
            from PIL import Image
            
            print(f"🔍 [Analyzer] Загрузка изображения: {image_path}")
            
            # Загрузка изображения (используем imdecode для поддержки кириллицы в пути)
            img_array = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"❌ [Analyzer] Не удалось загрузить изображение через cv2.imdecode()")
                print(f"   Путь к файлу: {image_path}")
                print(f"   Файл существует: {os.path.exists(image_path)}")
                if os.path.exists(image_path):
                    print(f"   Размер файла: {os.path.getsize(image_path)} байт")
                return {"error": "Не удалось загрузить изображение"}
            
            print(f"✅ [Analyzer] Изображение загружено успешно. Размер: {img.shape}")
            
            # Конвертация в различные цветовые пространства для анализа
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            print(f"📊 [Analyzer] Границы обнаружены. Количество активных пикселей: {np.sum(edges > 0)}")
            
            # Анализ гистограммы яркости для определения свечей
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Определение доминирующих цветов (для бычьих/медвежьих свечей)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Простой анализ тренда по направлению движения цены
            trend = self._detect_trend(edges, img)
            print(f"📈 [Analyzer] Определен тренд: {trend}")
            
            # Поиск горизонтальных линий (уровни поддержки/сопротивления)
            support_resistance = self._find_horizontal_levels(edges)
            print(f"📊 [Analyzer] Найдено уровней: поддержка={len(support_resistance.get('support', []))}, сопротивление={len(support_resistance.get('resistance', []))}")
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "trend": trend,
                "support_levels": support_resistance.get("support", []),
                "resistance_levels": support_resistance.get("resistance", []),
                "volatility": self._calculate_volatility(gray),
                "image_path": image_path,
                "confidence": 0.75  # Условная уверенность
            }
            
            self.analysis_history.append(analysis)
            print(f"✅ [Analyzer] Анализ завершен успешно")
            return analysis
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ [Analyzer] Ошибка анализа: {str(e)}\n{error_details}")
            return {"error": f"Ошибка анализа: {str(e)}"}
    
    def _detect_trend(self, edges, img) -> str:
        """Определяет направление тренда"""
        height, width = edges.shape[:2]
        
        # Разделяем изображение на левую и правую части
        left_part = edges[:, :width//2]
        right_part = edges[:, width//2:]
        
        # Считаем количество активных пикселей (границы свечей)
        left_activity = np.sum(left_part > 0)
        right_activity = np.sum(right_part > 0)
        
        # Простая эвристика: если справа больше активности - возможен рост
        if right_activity > left_activity * 1.1:
            return "BULLISH 📈"
        elif left_activity > right_activity * 1.1:
            return "BEARISH 📉"
        else:
            return "SIDEWAYS ➡️"
    
    def _find_horizontal_levels(self, edges) -> dict:
        """Ищет горизонтальные уровни поддержки и сопротивления"""
        height, width = edges.shape
        
        # Проецируем ребра на вертикальную ось
        vertical_projection = np.sum(edges > 0, axis=1)
        
        # Находим пики в проекции (горизонтальные линии)
        levels = []
        threshold = np.mean(vertical_projection) * 1.5
        
        for y in range(height):
            if vertical_projection[y] > threshold:
                # Проверяем, не слишком ли близко к предыдущему уровню
                if not levels or abs(y - levels[-1]) > 20:
                    levels.append(y)
        
        # Разделяем на поддержку и сопротивление (условно)
        mid_point = height // 2
        support = [level for level in levels if level > mid_point]
        resistance = [level for level in levels if level <= mid_point]
        
        return {
            "support": support[:3],  # Топ 3 уровня
            "resistance": resistance[:3]
        }
    
    def _calculate_volatility(self, gray_image) -> float:
        """Вычисляет волатильность по стандартному отклонению"""
        return float(np.std(gray_image)) / 255.0
    
    def get_analysis_summary(self) -> str:
        """Возвращает текстовое резюме последнего анализа"""
        if not self.analysis_history:
            return "Нет данных для анализа"
        
        last = self.analysis_history[-1]
        summary = f"""
📊 АНАЛИЗ ГРАФИКА
─────────────────
Время: {last['timestamp'][:19]}
Тренд: {last['trend']}
Волатильность: {last['volatility']:.2%}

Уровни сопротивления: {len(last['resistance_levels'])}
Уровни поддержки: {len(last['support_levels'])}

💡 Рекомендация: 
{'Рассмотреть покупку при подтверждении бычьего тренда' if 'BULLISH' in last['trend'] else 'Рассмотреть продажу при подтверждении медвежьего тренда' if 'BEARISH' in last['trend'] else 'Ожидать пробоя диапазона'}
"""
        return summary


class SmartBrowser(QMainWindow):
    """Основное окно умного браузера"""
    
    def __init__(self):
        super().__init__()
        self.analyzer = ChartAnalyzer()
        self.current_url = "https://www.tradingview.com/chart/"
        self.init_ui()
        self.load_tradingview()
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("🧠 Smart TradingView Browser")
        self.setGeometry(100, 100, 1400, 900)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Панель инструментов
        self.create_toolbar()
        
        # Разделитель для браузера и панели анализа
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Браузер
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl(self.current_url))
        splitter.addWidget(self.browser)
        
        # Панель анализа
        analysis_panel = self.create_analysis_panel()
        splitter.addWidget(analysis_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # Статус бар
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Готов к работе | Откройте график TradingView")
        
        # Таймер для автообновления статуса
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)
    
    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = QToolBar("Инструменты")
        self.addToolBar(toolbar)
        
        # Кнопка захвата скриншота
        screenshot_action = QAction("📸 Скриншот", self)
        screenshot_action.triggered.connect(self.capture_screenshot)
        toolbar.addAction(screenshot_action)
        
        # Кнопка анализа
        analyze_action = QAction("🔍 Анализировать", self)
        analyze_action.triggered.connect(self.analyze_chart)
        toolbar.addAction(analyze_action)
        
        # Кнопка сохранения
        save_action = QAction("💾 Сохранить", self)
        save_action.triggered.connect(self.save_analysis)
        toolbar.addAction(save_action)
        
        # Кнопка обновления
        refresh_action = QAction("🔄 Обновить", self)
        refresh_action.triggered.connect(lambda: self.browser.reload())
        toolbar.addAction(refresh_action)
        
        toolbar.addSeparator()
        
        # Поле ввода URL
        self.url_input = QLineEdit()
        self.url_input.setText(self.current_url)
        self.url_input.returnPressed.connect(self.navigate_to_url)
        self.url_input.setPlaceholderText("Введите URL...")
        self.url_input.setMinimumWidth(300)
        toolbar.addWidget(self.url_input)
    
    def create_analysis_panel(self) -> QWidget:
        """Создание панели анализа"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Заголовок
        title = QLabel("📊 Панель Анализа")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Область предпросмотра скриншота
        self.screenshot_preview = QLabel()
        self.screenshot_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screenshot_preview.setMinimumHeight(200)
        self.screenshot_preview.setStyleSheet("border: 2px solid #ccc; background: #f5f5f5;")
        self.screenshot_preview.setText("Скриншот будет здесь")
        layout.addWidget(self.screenshot_preview)
        
        # Кнопки действий
        btn_layout = QHBoxLayout()
        
        self.btn_capture = QPushButton("📸 Сделать скриншот")
        self.btn_capture.clicked.connect(self.capture_screenshot)
        btn_layout.addWidget(self.btn_capture)
        
        self.btn_analyze = QPushButton("🔍 Анализировать")
        self.btn_analyze.clicked.connect(self.analyze_chart)
        btn_layout.addWidget(self.btn_analyze)
        
        layout.addLayout(btn_layout)
        
        # Область результатов анализа
        self.analysis_result = QTextEdit()
        self.analysis_result.setReadOnly(True)
        self.analysis_result.setPlaceholderText("Результаты анализа появятся здесь...")
        self.analysis_result.setMinimumHeight(300)
        layout.addWidget(self.analysis_result)
        
        # Индикатор прогресса
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # История анализов
        history_label = QLabel("📜 История:")
        history_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(history_label)
        
        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        self.history_list.setMaximumHeight(150)
        layout.addWidget(self.history_list)
        
        return panel
    
    def load_tradingview(self):
        """Загрузка TradingView"""
        self.browser.setUrl(QUrl(self.current_url))
        self.statusBar.showMessage("Загрузка TradingView...")
    
    def navigate_to_url(self):
        """Навигация по введенному URL"""
        url = self.url_input.text()
        if not url.startswith("http"):
            url = "https://" + url
        self.browser.setUrl(QUrl(url))
        self.statusBar.showMessage(f"Переход на: {url}")
    
    def capture_screenshot(self):
        """Захват скриншота через mss (захват области экрана)"""
        try:
            self.statusBar.showMessage("Захват скриншота... Пожалуйста, подождите")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(30)
            
            # Запускаем worker для скриншота
            self.screenshot_worker = ScreenshotWorker(self)
            self.screenshot_worker.finished.connect(self.on_screenshot_finished)
            self.screenshot_worker.start()
            
        except Exception as e:
            self.statusBar.showMessage(f"Ошибка: {str(e)}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Ошибка", f"Не удалось сделать скриншот:\n{str(e)}")
    
    def on_screenshot_finished(self, result):
        """Обработка результата захвата скриншота"""
        if result.startswith("ERROR:"):
            self.statusBar.showMessage(f"Ошибка: {result}")
            QMessageBox.critical(self, "Ошибка скриншота", result)
            self.progress_bar.setVisible(False)
        else:
            try:
                # Скриншот успешно создан
                filepath = result
                filename = os.path.basename(filepath)
                
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    print(f"✅ Скриншот сохранен: {filepath}, размер: {file_size} байт")
                    
                    # Загружаем и показываем превью
                    pixmap = QPixmap(filepath)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(
                            self.screenshot_preview.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        self.screenshot_preview.setPixmap(scaled_pixmap)
                        
                        self.last_screenshot = filepath
                        self.progress_bar.setValue(100)
                        QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
                        
                        self.statusBar.showMessage(f"✅ Скриншот сохранен: {filename}")
                        
                        # Уведомление
                        msg = QMessageBox(self)
                        msg.setIcon(QMessageBox.Icon.Information)
                        msg.setWindowTitle("Скриншот готов")
                        msg.setText(f"Скриншот сохранен:\n{filename}")
                        msg.setInformativeText("Хотите открыть файл?")
                        msg.setStandardButtons(QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Ok)
                        msg.setDefaultButton(QMessageBox.StandardButton.Ok)
                        
                        ret = msg.exec()
                        if ret == QMessageBox.StandardButton.Open:
                            self.open_screenshot_file(filepath)
                
                self.progress_bar.setVisible(False)
            except Exception as e:
                self.statusBar.showMessage(f"Ошибка: {str(e)}")
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Ошибка", f"Не удалось сделать скриншот:\n{str(e)}")
    
    def open_screenshot_file(self, filepath):
        """Открывает скриншот в стандартном приложении просмотра изображений"""
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Windows":
                os.startfile(filepath)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", filepath])
            else:  # Linux
                subprocess.run(["xdg-open", filepath])
        except Exception as e:
            QMessageBox.warning(self, "Предупреждение", f"Не удалось открыть файл автоматически:\n{e}\n\nПуть: {filepath}")
    
    def analyze_chart(self):
        """Анализ текущего скриншота"""
        if not hasattr(self, 'last_screenshot') or not os.path.exists(self.last_screenshot):
            QMessageBox.warning(
                self,
                "Нет скриншота",
                "Сначала сделайте скриншот графика!"
            )
            return
        
        try:
            self.statusBar.showMessage("Анализ графика...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(30)
            
            # Проверяем существование файла и его размер
            if not os.path.exists(self.last_screenshot):
                raise FileNotFoundError(f"Файл скриншота не найден: {self.last_screenshot}")
            
            file_size = os.path.getsize(self.last_screenshot)
            print(f"🔍 Начинаю анализ файла: {self.last_screenshot}, размер: {file_size} байт")
            
            # Выполняем анализ
            analysis = self.analyzer.analyze_screenshot(self.last_screenshot)
            
            print(f"📊 Результат анализа: {analysis}")
            
            self.progress_bar.setValue(70)
            
            if "error" in analysis:
                error_msg = f"❌ Ошибка анализа:\n{analysis['error']}"
                print(error_msg)
                self.analysis_result.setText(error_msg)
            else:
                # Отображаем результаты
                summary = self.analyzer.get_analysis_summary()
                print(f"✅ Анализ успешен: {summary[:100]}...")
                self.analysis_result.setText(summary)
                
                # Обновляем историю
                self.update_history_display()
                
                self.statusBar.showMessage("Анализ завершен успешно!")
            
            self.progress_bar.setValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            error_trace = f"Ошибка анализа: {str(e)}\nТип: {type(e).__name__}"
            print(error_trace)
            import traceback
            traceback.print_exc()
            self.statusBar.showMessage(f"Ошибка анализа: {str(e)}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Ошибка", f"Не удалось проанализировать график:\n{str(e)}")
    
    def update_history_display(self):
        """Обновление отображения истории анализов"""
        history_text = ""
        for i, analysis in enumerate(reversed(self.analyzer.analysis_history[-5:]), 1):
            history_text += f"{i}. {analysis['timestamp'][11:19]} - {analysis['trend']}\n"
        self.history_list.setText(history_text if history_text else "История пуста")
    
    def save_analysis(self):
        """Сохранение текущего анализа"""
        if not self.analysis_result.toPlainText():
            QMessageBox.warning(self, "Нет данных", "Нет анализа для сохранения")
            return
        
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить анализ",
                "",
                "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.analysis_result.toPlainText())
                    f.write("\n\n---\n")
                    f.write(f"Скриншот: {getattr(self, 'last_screenshot', 'Не указан')}\n")
                    f.write(f"Дата экспорта: {datetime.now().isoformat()}\n")
                
                self.statusBar.showMessage(f"Анализ сохранен: {os.path.basename(filename)}")
                QMessageBox.information(self, "Успешно", "Анализ сохранен!")
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{str(e)}")
    
    def update_status(self):
        """Периодическое обновление статуса"""
        current_url = self.browser.url().toString()
        if "tradingview" in current_url.lower():
            self.statusBar.showMessage("TradingView активен | Готов к анализу")
        else:
            self.statusBar.showMessage(f"Просмотр: {current_url[:50]}...")


def main():
    """Точка входа приложения"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Установка темной темы (опционально)
    # app.setStyleSheet(dark_stylesheet)
    
    window = SmartBrowser()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
