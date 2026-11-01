"""Shared Light / Dark colors. Theme changes affect presentation only."""
from __future__ import annotations

from pathlib import Path
from PySide6.QtGui import QColor, QPalette


COLORS = {
    'dark': dict(window='#0b1018', panel='#111a27', input='#121a28', text='#eaf0fa',
                 muted='#a0aec0', border='#303c52', button='#1a2637', hover='#23344c',
                 disabled='#0d131d', disabled_text='#718096', scroll='#34445e',
                 accent='#16c79a', accent_hover='#12ad86', accent_text='#07120f',
                 selected='#204d45', selected_text='#b5f4df', badge='#15253a',
                 badge_text='#7de3c8', canvas='#090d14', canvas_text='#a0aec0',
                 plot='#0d1723', axis='#afbdd0', raw='#8698b2', curve='#eab675',
                 marker='#6fe1bd', progress='#16877b', progress_text='#f0fffc'),
    'light': dict(window='#f1f4f7', panel='#ffffff', input='#ffffff', text='#202c3a',
                  muted='#536476', border='#c8d2dc', button='#edf2f6', hover='#dfe8ef',
                  disabled='#e9eef2', disabled_text='#768390', scroll='#b4c1cd',
                  accent='#2563eb', accent_hover='#1d4ed8', accent_text='#ffffff',
                  selected='#e7efff', selected_text='#1e40af', badge='#e7efff',
                  badge_text='#1e40af', canvas='#969ca3', canvas_text='#202832',
                  plot='#ffffff', axis='#536476', raw='#71849a', curve='#ac5e17',
                  marker='#2563eb', progress='#bdd5fb', progress_text='#1e3a5f'),
}


def theme_name(value) -> str:
    return value if isinstance(value, str) and value in COLORS else 'dark'


def colors_for(value) -> dict:
    return COLORS[theme_name(value)]


def palette_for(value) -> QPalette:
    c = colors_for(value)
    palette = QPalette()
    roles = dict(Window='window', WindowText='text', Base='input', AlternateBase='window',
                 ToolTipBase='panel', ToolTipText='text', Text='text', Button='button',
                 ButtonText='text', BrightText='accent_text', Highlight='accent',
                 HighlightedText='accent_text', PlaceholderText='muted', Link='marker',
                 Light='panel', Midlight='hover', Mid='border', Dark='scroll', Shadow='border')
    for name, key in roles.items():
        role = getattr(QPalette.ColorRole, name)
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            palette.setColor(group, role, QColor(c[key]))
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(c[key]))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText,
                 QPalette.ColorRole.PlaceholderText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(c['disabled_text']))
    for role in (QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(c['disabled']))
    return palette


def stylesheet_for(value) -> str:
    c = colors_for(value)
    assets=Path(__file__).resolve().parent/'theme_assets'
    down=(assets/f'down_{theme_name(value)}.svg').as_posix()
    up=(assets/f'up_{theme_name(value)}.svg').as_posix()
    return f"""
        QWidget {{ background:transparent; color:{c['text']}; font-family:'Segoe UI'; font-size:13px; }}
        QMainWindow,QDialog,QMessageBox,QWidget#appSurface {{ background:{c['window']}; }}
        QLabel {{ background:transparent; }}
        QWidget:disabled {{ color:{c['disabled_text']}; }}
        QGroupBox {{ background:{c['panel']}; border:1px solid {c['border']}; border-radius:8px;
                     margin-top:12px; padding-top:10px; font-weight:600; }}
        QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 5px; }}
        QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QPlainTextEdit,QTextEdit {{
            background:{c['input']}; border:1px solid {c['border']}; border-radius:5px; padding:6px;
            selection-background-color:{c['accent']}; selection-color:{c['accent_text']}; }}
        QLineEdit:focus,QComboBox:focus,QSpinBox:focus,QDoubleSpinBox:focus,QPlainTextEdit:focus {{
            border:1px solid {c['accent']}; }}
        QComboBox {{ padding-right:26px; }}
        QComboBox::drop-down {{ subcontrol-origin:padding; subcontrol-position:top right; width:24px; border:none; }}
        QComboBox::down-arrow {{ image:url("{down}"); width:12px; height:8px; }}
        QLineEdit:disabled,QComboBox:disabled,QSpinBox:disabled,QDoubleSpinBox:disabled,
        QPlainTextEdit:disabled,QTextEdit:disabled {{ background:{c['disabled']}; color:{c['disabled_text']}; }}
        QSpinBox,QDoubleSpinBox {{ padding-right:26px; }}
        QSpinBox::up-button,QSpinBox::down-button,QDoubleSpinBox::up-button,QDoubleSpinBox::down-button {{ width:24px; border:none; background:transparent; }}
        QSpinBox::up-arrow,QDoubleSpinBox::up-arrow {{ image:url("{up}"); width:10px; height:6px; }}
        QSpinBox::down-arrow,QDoubleSpinBox::down-arrow {{ image:url("{down}"); width:10px; height:6px; }}
        QPushButton,QToolButton {{ background:{c['button']}; border:1px solid {c['border']}; border-radius:6px; padding:8px 12px; }}
        QPushButton:hover,QToolButton:hover {{ background:{c['hover']}; }}
        QPushButton:pressed,QToolButton:pressed {{ background:{c['selected']}; }}
        QPushButton:checked,QToolButton:checked {{ background:{c['selected']}; color:{c['selected_text']}; border-color:{c['accent']}; }}
        QPushButton:disabled,QToolButton:disabled {{ background:{c['disabled']}; color:{c['disabled_text']}; border-color:{c['border']}; }}
        QPushButton#primaryButton {{ background:{c['accent']}; color:{c['accent_text']}; border-color:{c['accent']}; font-weight:700; }}
        QPushButton#primaryButton:hover {{ background:{c['accent_hover']}; }}
        QPushButton#primaryButton:disabled {{ background:{c['disabled']}; color:{c['disabled_text']}; border-color:{c['border']}; }}
        QPushButton#themeToggle {{ min-width:60px; padding:8px 12px; }}
        QPushButton#compactSettings {{ padding:0px; font-family:'Segoe UI Symbol'; font-size:16px; }}
        QCheckBox {{ spacing:6px; background:transparent; }}
        QProgressBar {{ border:1px solid {c['border']}; border-radius:5px; background:{c['disabled']};
                       color:{c['progress_text']}; text-align:center; min-height:18px; }}
        QProgressBar::chunk {{ background:{c['progress']}; border-radius:4px; }}
        QSlider::groove:horizontal {{ background:{c['border']}; height:4px; border-radius:2px; }}
        QSlider::sub-page:horizontal {{ background:{c['accent']}; border-radius:2px; }}
        QSlider::handle:horizontal {{ background:{c['panel']}; border:1px solid {c['scroll']}; width:12px;
                                      margin:-5px 0; border-radius:4px; }}
        QSlider::handle:horizontal:disabled {{ background:{c['disabled']}; border-color:{c['border']}; }}
        QScrollArea {{ border:none; background:transparent; }}
        QScrollBar:vertical {{ background:{c['disabled']}; width:12px; margin:2px; border:none; border-radius:6px; }}
        QScrollBar:horizontal {{ background:{c['disabled']}; height:12px; margin:2px; border:none; border-radius:6px; }}
        QScrollBar::handle:vertical {{ background:{c['scroll']}; min-height:34px; border-radius:4px; }}
        QScrollBar::handle:horizontal {{ background:{c['scroll']}; min-width:34px; border-radius:4px; }}
        QScrollBar::handle:hover {{ background:{c['accent']}; }}
        QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {{ height:0px; background:transparent; border:none; }}
        QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal {{ width:0px; background:transparent; border:none; }}
        QScrollBar::add-page,QScrollBar::sub-page {{ background:transparent; }}
        QLabel#title {{ font-size:24px; font-weight:700; }}
        QLabel#statusPill {{ background:{c['badge']}; color:{c['badge_text']}; border-radius:10px; padding:6px 12px; }}
        QLabel#jobState {{ font-size:22px; font-weight:700; color:{c['badge_text']}; }}
        QLabel#analysisCenter {{ font-size:19px; font-weight:600; color:{c['marker']}; }}
        QLabel#imageCanvas,QGraphicsView#maskCanvas {{ background:{c['canvas']}; color:{c['canvas_text']}; border:1px solid {c['border']}; }}
        QTabWidget::pane {{ background:{c['panel']}; border:1px solid {c['border']}; }}
        QTabBar::tab {{ background:{c['window']}; color:{c['muted']}; border:1px solid {c['border']};
                       border-bottom:0; padding:7px 12px; margin-right:2px; border-top-left-radius:5px; border-top-right-radius:5px; }}
        QTabBar::tab:selected {{ background:{c['panel']}; color:{c['text']}; }}
        QTabBar::tab:hover {{ background:{c['hover']}; }}
        QMenu,QAbstractItemView {{ background:{c['panel']}; color:{c['text']}; border:1px solid {c['border']};
                                  selection-background-color:{c['selected']}; selection-color:{c['selected_text']}; }}
        QMenu::item:selected {{ background:{c['selected']}; color:{c['selected_text']}; }}
        QMenu::item:disabled {{ color:{c['disabled_text']}; }}
        QHeaderView::section {{ background:{c['window']}; color:{c['text']}; border:1px solid {c['border']}; padding:5px; }}
        QToolTip {{ background:{c['panel']}; color:{c['text']}; border:1px solid {c['border']}; padding:5px; }}
        QStatusBar {{ background:{c['panel']}; border-top:1px solid {c['border']}; }}
        QSplitter::handle {{ background:{c['window']}; }}
    """


def current_theme(widget):
    while widget is not None:
        manager=getattr(widget,'theme_manager',None)
        if manager is not None:return manager.mode
        widget=widget.parentWidget()
    return 'dark'


class ThemeController:
    """Per-application preference; never changes task/model state or image pixels."""
    def __init__(self,window,button,key,dark_style,extra_light='',changed=None):
        import os
        from PySide6.QtCore import QSettings
        from PySide6.QtGui import QFontDatabase,QFont
        from PySide6.QtWidgets import QApplication
        global _fonts_ready
        if not globals().get('_fonts_ready',False):
            fonts=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
            for name in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf','msyh.ttc'):
                if (fonts/name).is_file():QFontDatabase.addApplicationFont(str(fonts/name))
            if QApplication.instance():QApplication.instance().setFont(QFont('Segoe UI',10))
            _fonts_ready=True
        self.window=window;self.button=button;self.dark_style=dark_style
        self.extra_light=extra_light;self.changed=changed
        folder=os.environ.get('BL62_THEME_SETTINGS_ROOT')
        if folder:
            Path(folder).mkdir(parents=True,exist_ok=True)
            self.settings=QSettings(str(Path(folder)/(key+'.ini')),QSettings.Format.IniFormat)
        else:self.settings=QSettings('BL62',key+'Theme')
        self.mode=theme_name(self.settings.value('theme','dark'))
        window.theme_manager=self;window.theme_button=button
        if window.centralWidget():window.centralWidget().setObjectName('appSurface')
        button.setObjectName('themeToggle');button.setAccessibleName('Color Theme')
        button.clicked.connect(self.toggle);self.apply()

    def apply(self):
        from PySide6.QtCore import Qt,QTimer
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QAbstractScrollArea,QGraphicsView
        w=self.window;positions=[]
        for area in w.findChildren(QAbstractScrollArea):
            bar=area.verticalScrollBar()
            if bar.maximum()>0:positions.append((bar,bar.value(),bar.value()==bar.maximum()))
        w.setAttribute(Qt.WidgetAttribute.WA_WindowPropagation,True)
        w.setPalette(palette_for(self.mode))
        if self.mode=='dark':style=self.dark_style+'\nQPushButton#themeToggle{min-width:116px;padding:8px 12px;}'
        else:style=stylesheet_for('light')+LIGHT_EXTRAS+self.extra_light
        w.setStyleSheet(style)
        for view in w.findChildren(QGraphicsView):view.setBackgroundBrush(QColor(colors_for(self.mode)['canvas']))
        target='Light' if self.mode=='dark' else 'Dark'
        self.button.setText('Switch to '+target);self.button.setToolTip('Switch to '+target)
        if self.changed:self.changed(self.mode)
        def restore():
            for bar,value,end in positions:
                try:bar.setValue(bar.maximum() if end else min(value,bar.maximum()))
                except RuntimeError:pass
        QTimer.singleShot(0,restore)

    def toggle(self):
        self.mode='light' if self.mode=='dark' else 'dark';self.apply()
        self.settings.setValue('theme',self.mode);self.settings.sync()


LIGHT_EXTRAS='''
QPushButton#primary,QPushButton#new{background:#2563eb;border-color:#2563eb;color:white;font-weight:600}
QPushButton#primary:hover,QPushButton#new:hover{background:#1d4ed8}
QPushButton#primary:disabled,QPushButton#new:disabled{background:#e9eef2;color:#768390;border-color:#c8d2dc}
QLabel#state{font-size:21px;color:#1e40af} QLabel#muted{color:#536476;font-size:12px}
QToolBar{background:#ffffff;border:0;border-bottom:1px solid #c8d2dc;spacing:5px;padding:5px}
QToolBar QToolButton{background:#edf2f6;border:1px solid #c8d2dc;border-radius:5px;padding:7px 10px}
QToolBar QToolButton:hover{background:#dfe8ef}
QToolBar QToolButton:pressed,QToolBar QToolButton:checked{background:#2563eb;color:white}
QListWidget,QTableWidget{background:#ffffff;alternate-background-color:#f1f4f7;border:1px solid #c8d2dc;border-radius:6px;gridline-color:#c8d2dc;outline:0}
QListWidget::item,QTableWidget::item{padding:5px}
QListWidget::item:selected,QTableWidget::item:selected{background:#e7efff;color:#1e40af}
'''
