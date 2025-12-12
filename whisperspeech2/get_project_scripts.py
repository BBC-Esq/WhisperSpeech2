import os
import sys
import yaml
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QPushButton,
    QWidget,
    QScrollArea,
    QLabel,
    QSpinBox,
    QGroupBox,
)
from PySide6.QtCore import Qt, QUrl, QThread, Signal
from PySide6.QtGui import QDesktopServices


class FileScanWorker(QThread):
    finished = Signal(list)
    progress = Signal(str)

    def __init__(self, current_directory, scan_depth, excluded_folders, excluded_files, active_extensions):
        super().__init__()
        self.current_directory = current_directory
        self.scan_depth = scan_depth
        self.excluded_folders = excluded_folders
        self.excluded_files = excluded_files
        self.active_extensions = active_extensions

    def run(self):
        collected = []
        if not self.active_extensions:
            self.finished.emit(collected)
            return

        def should_exclude_folder(folder_name):
            return folder_name.lower() in self.excluded_folders

        def should_exclude_file(file_path):
            file_name = file_path.name.lower()
            return file_name in {f.lower() for f in self.excluded_files}

        def scan_directory(directory, current_depth, max_depth):
            try:
                self.progress.emit(f"Scanning: {directory.name}")
                for item in directory.iterdir():
                    if item.is_file() and item.suffix.lower() in self.active_extensions:
                        if not should_exclude_file(item):
                            collected.append(str(item.resolve()))
                    elif item.is_dir() and current_depth < max_depth:
                        if not should_exclude_folder(item.name):
                            scan_directory(item, current_depth + 1, max_depth)
            except PermissionError:
                pass

        scan_directory(self.current_directory, 0, self.scan_depth)
        self.finished.emit(sorted(collected))


class FileCompilerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_file = "settings.yaml"
        self.checkboxes = {}
        self.excluded_folders = {
            'lib',
            'venv',
            'env',
            '__pycache__',
            '.git',
            '.vscode',
            'node_modules',
            'build',
            'dist',
            'site-packages',
        }
        self.excluded_files = {
            '__init__.py',
            'check_packages.py',
            'replace_sourcecode.py',
            'get_project_scripts.py',
            'setup_windows.py',
            'settings.yaml',
            'themes.yaml',
        }
        self.scan_depth = 0
        self.current_directory = Path.cwd()
        self.python_extensions = {".py", ".qml", ".yaml", ".qss", ".en-sample", ".in", ".ini", ".mlang-sample", ".scss", ".whisperspeech", ".yml"}
        self.typescript_extensions = {".ts", ".tsx", ".js", ".json", ".css"}
        self.website_extensions = {".html", ".htm", ".css", ".js", ".jsx", ".json", ".xml", ".svg", ".scss", ".sass", ".less", ".php", ".asp", ".aspx", ".vue"}
        self.include_python = True
        self.include_typescript = False
        self.include_website = False
        self.file_paths = []
        self.excluded_extensions = []
        self.root_dir = str(self.current_directory)
        self.project_name = self.current_directory.name
        self.scan_worker = None
        self.saved_checkbox_states = {}
        self.init_ui()
        self.load_config()
        self.refresh_scan()

    def _get_active_extensions(self):
        extensions = set()
        if self.include_python:
            extensions.update(self.python_extensions)
        if self.include_typescript:
            extensions.update(self.typescript_extensions)
        if self.include_website:
            extensions.update(self.website_extensions)
        return extensions

    def _get_excluded_extensions(self):
        excluded_exts = set()
        target_exts = self._get_active_extensions()
        
        def should_exclude_folder(folder_name):
            return folder_name.lower() in self.excluded_folders
        
        def scan_for_extensions(directory, current_depth, max_depth):
            try:
                for item in directory.iterdir():
                    if item.is_file():
                        if item.name.lower() not in {f.lower() for f in self.excluded_files}:
                            ext = item.suffix.lower()
                            if ext and ext not in target_exts:
                                excluded_exts.add(ext)
                    elif item.is_dir() and current_depth < max_depth:
                        if not should_exclude_folder(item.name):
                            scan_for_extensions(item, current_depth + 1, max_depth)
            except PermissionError:
                pass
        
        scan_for_extensions(self.current_directory, 0, self.scan_depth)
        return sorted(excluded_exts)

    def _find_common_root(self):
        if not self.file_paths:
            return str(self.current_directory)
        return str(self.current_directory)

    def _get_relative_path(self, file_path):
        try:
            rel_path = Path(file_path).relative_to(self.current_directory)
            return str(rel_path).replace('\\', '/')
        except ValueError:
            return file_path

    def _get_file_char_count(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        return len(content)

    def _calculate_total_chars(self):
        total_chars = 0
        selected_count = 0
        for file_path, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                selected_count += 1
                total_chars += self._get_file_char_count(file_path)
        return selected_count, total_chars

    def _update_total_display(self):
        selected_count, total_chars = self._calculate_total_chars()
        self.total_chars_label.setText(f"Selected: {selected_count} files, {total_chars:,} characters")

    def _update_excluded_info(self):
        self.excluded_extensions = self._get_excluded_extensions()
        excluded_text = f"Excluded folders: {', '.join(sorted(self.excluded_folders))}\nExcluded files: {', '.join(sorted(self.excluded_files))}"
        if self.excluded_extensions:
            excluded_text += f"\nExcluded file extensions: {', '.join(self.excluded_extensions)}"
        self.excluded_info_label.setText(excluded_text)

    def _get_mode_display(self):
        modes = []
        if self.include_python:
            modes.append("Python")
        if self.include_typescript:
            modes.append("TypeScript")
        if self.include_website:
            modes.append("Website")
        if modes:
            return ", ".join(modes)
        else:
            return "No file types selected"

    def init_ui(self):
        self.setWindowTitle("Source File Compiler")
        self.setGeometry(100, 100, 900, 1100)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        project_info = QLabel(f"Project: {self.project_name}\nRoot: {self.current_directory}")
        project_info.setStyleSheet("font-weight: bold; font-size: 12px; margin: 10px; padding: 10px; border-radius: 5px;")
        layout.addWidget(project_info)
        scan_group = QGroupBox("Settings")
        scan_layout = QHBoxLayout(scan_group)
        depth_label = QLabel("Scan Depth:")
        self.depth_spinbox = QSpinBox()
        self.depth_spinbox.setMinimum(0)
        self.depth_spinbox.setMaximum(10)
        self.depth_spinbox.setValue(self.scan_depth)
        self.depth_spinbox.setToolTip("0 = current directory only, 1 = include subfolders, etc.")
        self.refresh_btn = QPushButton("Refresh Scan")
        self.refresh_btn.clicked.connect(self.refresh_scan)
        self.refresh_btn.setToolTip("Rescan files with current settings")
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setFixedHeight(24)
        self.select_all_btn.setFixedWidth(90)
        self.select_all_btn.clicked.connect(self.select_all)
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.setFixedHeight(24)
        self.deselect_all_btn.setFixedWidth(90)
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        scan_layout.addWidget(depth_label)
        scan_layout.addWidget(self.depth_spinbox)
        scan_layout.addWidget(self.refresh_btn)
        scan_layout.addStretch()
        scan_layout.addWidget(self.select_all_btn)
        scan_layout.addWidget(self.deselect_all_btn)
        layout.addWidget(scan_group)
        file_type_layout = QHBoxLayout()
        file_type_label = QLabel("File Types:")
        self.python_checkbox = QCheckBox("Python")
        self.typescript_checkbox = QCheckBox("TypeScript")
        self.website_checkbox = QCheckBox("Website")
        self.python_checkbox.setChecked(self.include_python)
        self.typescript_checkbox.setChecked(self.include_typescript)
        self.website_checkbox.setChecked(self.include_website)

        def on_file_type_change():
            self.include_python = self.python_checkbox.isChecked()
            self.include_typescript = self.typescript_checkbox.isChecked()
            self.include_website = self.website_checkbox.isChecked()
            self.refresh_scan()

        self.python_checkbox.stateChanged.connect(on_file_type_change)
        self.typescript_checkbox.stateChanged.connect(on_file_type_change)
        self.website_checkbox.stateChanged.connect(on_file_type_change)
        file_type_layout.addWidget(file_type_label)
        file_type_layout.addWidget(self.python_checkbox)
        file_type_layout.addWidget(self.typescript_checkbox)
        file_type_layout.addWidget(self.website_checkbox)
        file_type_layout.addStretch()
        layout.addLayout(file_type_layout)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-weight: bold; color: #FF9800; margin: 5px;")
        layout.addWidget(self.status_label)
        self.file_count_label = QLabel(f"Found {len(self.file_paths)} {self._get_mode_display()} files")
        self.file_count_label.setStyleSheet("font-weight: bold; margin: 5px;")
        layout.addWidget(self.file_count_label)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(2)
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)
        self.copy_btn = QPushButton("Copy Selected Files to Clipboard")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        self.copy_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white; padding: 10px;")
        layout.addWidget(self.copy_btn)
        excluded_text = f"Excluded folders: {', '.join(sorted(self.excluded_folders))}\nExcluded files: {', '.join(sorted(self.excluded_files))}"
        if self.excluded_extensions:
            excluded_text += f"\nExcluded file extensions: {', '.join(self.excluded_extensions)}"
        self.excluded_info_label = QLabel(excluded_text)
        self.excluded_info_label.setStyleSheet("font-size: 10px; color: gray; margin: 5px;")
        self.excluded_info_label.setWordWrap(True)
        layout.addWidget(self.excluded_info_label)
        self.total_chars_label = QLabel("Selected: 0 files, 0 characters")
        self.total_chars_label.setStyleSheet("font-size: 12px; margin: 5px; font-weight: bold; color: #2196F3;")
        layout.addWidget(self.total_chars_label)

    def _populate_file_list(self):
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.checkboxes.clear()
        for file_path in self.file_paths:
            rel_path = self._get_relative_path(file_path)
            char_count = self._get_file_char_count(file_path)
            container = QWidget()
            hl = QHBoxLayout(container)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)
            checkbox = QCheckBox(f"{rel_path} ({char_count:,} chars)")
            if file_path in self.saved_checkbox_states:
                checkbox.setChecked(self.saved_checkbox_states[file_path])
            else:
                checkbox.setChecked(True)
            checkbox.setToolTip(file_path)
            checkbox.stateChanged.connect(self._update_total_display)
            self.checkboxes[file_path] = checkbox
            hl.addWidget(checkbox)
            link = QLabel(f'<a href="{file_path}">{rel_path}</a>')
            link.setTextInteractionFlags(Qt.TextBrowserInteraction)
            link.setOpenExternalLinks(False)
            link.linkActivated.connect(self.open_file)
            hl.addWidget(link)
            self.scroll_layout.addWidget(container)
        self._update_total_display()

    def open_file(self, file_path: str):
        QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

    def _disable_controls(self):
        self.refresh_btn.setEnabled(False)
        self.depth_spinbox.setEnabled(False)
        self.python_checkbox.setEnabled(False)
        self.typescript_checkbox.setEnabled(False)
        self.website_checkbox.setEnabled(False)
        self.select_all_btn.setEnabled(False)
        self.deselect_all_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)

    def _enable_controls(self):
        self.refresh_btn.setEnabled(True)
        self.depth_spinbox.setEnabled(True)
        self.python_checkbox.setEnabled(True)
        self.typescript_checkbox.setEnabled(True)
        self.website_checkbox.setEnabled(True)
        self.select_all_btn.setEnabled(True)
        self.deselect_all_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)

    def refresh_scan(self):
        if self.scan_worker is not None and self.scan_worker.isRunning():
            return

        self.scan_depth = self.depth_spinbox.value()
        self.root_dir = str(self.current_directory)

        self._disable_controls()
        self.status_label.setText("Scanning files...")

        active_extensions = self._get_active_extensions()
        self.scan_worker = FileScanWorker(
            self.current_directory,
            self.scan_depth,
            self.excluded_folders,
            self.excluded_files,
            active_extensions
        )
        self.scan_worker.progress.connect(self._on_scan_progress)
        self.scan_worker.finished.connect(self._on_scan_finished)
        self.scan_worker.start()

    def _on_scan_progress(self, message):
        self.status_label.setText(message)

    def _on_scan_finished(self, file_paths):
        self.file_paths = file_paths
        self.status_label.setText("")
        self.file_count_label.setText(f"Found {len(self.file_paths)} {self._get_mode_display()} files")
        self._populate_file_list()
        self._update_excluded_info()
        self._enable_controls()
        self.save_config()
        self.scan_worker = None

    def select_all(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)
        self._update_total_display()

    def deselect_all(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
        self._update_total_display()

    def copy_to_clipboard(self):
        selected_files = [path for path, checkbox in self.checkboxes.items() if checkbox.isChecked()]
        if not selected_files:
            return
        compiled_text = f"# {self.project_name} - Source Code\n\n"
        compiled_text += f"Project structure with {len(selected_files)} files:\n\n"
        file_groups = {}
        for file_path in selected_files:
            rel_path = self._get_relative_path(file_path)
            directory = str(Path(rel_path).parent).replace('\\', '/') if Path(rel_path).parent != Path('.') else 'root'
            if directory not in file_groups:
                file_groups[directory] = []
            file_groups[directory].append((file_path, rel_path))
        for directory, files in sorted(file_groups.items()):
            compiled_text += f"## {directory}/\n"
            for _, rel_path in files:
                compiled_text += f"- {Path(rel_path).name}\n"
            compiled_text += "\n"
        compiled_text += "---\n\n"
        for i, file_path in enumerate(selected_files):
            rel_path = self._get_relative_path(file_path)
            compiled_text += f"## File: {rel_path}\n\n"
            fence_lang = "python" if file_path.endswith('.py') else "text"
            compiled_text += f"```{fence_lang}\n"
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    file_content = file.read()
            compiled_text += file_content
            compiled_text += "\n```\n"
            if i < len(selected_files) - 1:
                compiled_text += "\n---\n\n"
        clipboard = QApplication.clipboard()
        clipboard.setText(compiled_text)
        self.save_config()
        try:
            with open("compiled_files.txt", "w", encoding="utf-8") as f:
                f.write(compiled_text)
        except:
            pass

    def save_config(self):
        config = {}
        for file_path, checkbox in self.checkboxes.items():
            config[file_path] = checkbox.isChecked()
        config['_scan_depth'] = self.scan_depth
        config['_include_python'] = self.include_python
        config['_include_typescript'] = self.include_typescript
        config['_include_website'] = self.include_website
        with open(self.config_file, 'w') as file:
            yaml.dump(config, file)

    def load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file) or {}
                if '_scan_depth' in config and isinstance(config['_scan_depth'], int):
                    self.scan_depth = config['_scan_depth']
                    self.depth_spinbox.setValue(self.scan_depth)
                if '_include_python' in config:
                    self.include_python = config['_include_python']
                    self.python_checkbox.setChecked(self.include_python)
                if '_include_typescript' in config:
                    self.include_typescript = config['_include_typescript']
                    self.typescript_checkbox.setChecked(self.include_typescript)
                if '_include_website' in config:
                    self.include_website = config['_include_website']
                    self.website_checkbox.setChecked(self.include_website)
                self.saved_checkbox_states = {k: v for k, v in config.items() if not k.startswith('_')}
        except FileNotFoundError:
            pass

    def closeEvent(self, event):
        self.save_config()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = FileCompilerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()