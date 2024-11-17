from PyQt6 import QtWidgets, QtGui, QtCore

class AnimatedListWidgetItem(QtWidgets.QListWidgetItem):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.default_color = QtGui.QColor("#FFFFFF")
        self.hover_color = QtGui.QColor("#FFD700")  # couleur dorée pour le survol
        self.selected_color = QtGui.QColor("#FF6347")  # couleur tomate pour la sélection
        self.setForeground(self.default_color)

    def set_hover(self, hover):
        target_color = self.hover_color if hover else self.default_color
        self.animate_color(self.foreground().color(), target_color)

    def set_selected(self, selected):
        target_color = self.selected_color if selected else self.default_color
        self.animate_color(self.foreground().color(), target_color)

    def animate_color(self, start_color, end_color):
        """Anime la couleur de l'élément entre deux états."""
        animation = QtCore.QVariantAnimation()
        animation.setStartValue(start_color)
        animation.setEndValue(end_color)
        animation.setDuration(300)
        animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)

        def update_color(value):
            brush = QtGui.QBrush(value)
            self.setForeground(brush)

        animation.valueChanged.connect(update_color)
        animation.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
