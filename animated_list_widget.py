from PyQt6 import QtWidgets, QtGui, QtCore
from animated_list_widget_item import AnimatedListWidgetItem

class AnimatedListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # Activer le suivi de la souris pour les animations de survol
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # Retire le focus pour éviter le rectangle gris

        # Appliquer le style de base sans transition
        self.setStyleSheet("""
            QListWidget {
                border: none;
                background-color: #1E1E2F;
                border-radius: 10px;
                font-size: 14px;
                color: #FFFFFF;
            }
            QListWidget::item {
                padding: 12px;
                margin: 4px;
                border-radius: 6px;
            }
            QListWidget::item:hover {
                background-color: #FFD700;
                color: #1E1E2F;
            }
            QListWidget::item:selected {
                background-color: #FF6347;
                color: white;
            }
            QListWidget::item:focus {
                outline: none;
            }
        """)

    def add_animated_item(self, text):
        item = AnimatedListWidgetItem(text, self)
        self.addItem(item)

    def enterEvent(self, event):
        item = self.itemAt(self.mapFromGlobal(QtGui.QCursor.pos()))
        if isinstance(item, AnimatedListWidgetItem):
            item.set_hover(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        for index in range(self.count()):
            item = self.item(index)
            if isinstance(item, AnimatedListWidgetItem):
                item.set_hover(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        for index in range(self.count()):
            item = self.item(index)
            if isinstance(item, AnimatedListWidgetItem):
                item.set_selected(False)  # Réinitialiser la sélection
        if isinstance(item, AnimatedListWidgetItem):
            item.set_selected(True)
        super().mousePressEvent(event)
