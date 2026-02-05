import customtkinter as ctk
from pathlib import Path
from datetime import datetime

from src.model.logistic_regression_model import LogisticRegressionModel
from src.model.random_forest_model import RandomForestModel
from src.analyzer.sentiment_analyzer import SentimentAnalyzer


# Configuración de rutas
LOGISTIC_REGRESSION_MODEL_PATH = Path("src/model/logistic_regression_model.pkl")
RANDOM_FOREST_MODEL_PATH = Path("src/model/random_forest_model.pkl")
DATA_PATH = "data/reviews.csv"

# Configuración de colores
COLORS = {
    "bg_dark": "#212121",
    "bg_chat": "#2d2d2d", 
    "bg_input": "#40414f",
    "text_primary": "#ececec",
    "text_secondary": "#8e8ea0",
    "user_bubble": "#444654",
    "positivo": "#4ade80",      # Verde suave
    "neutral": "#fbbf24",        # Amarillo
    "negativo": "#f87171",       # Rojo suave
    "border": "#565869",
    "button_hover": "#2a2a3a",
    "accent": "#10a37f"
}


class ChatMessage(ctk.CTkFrame):
    """Widget para mostrar un mensaje en el chat"""
    
    def __init__(self, parent, text: str, is_user: bool = True, sentiment: str = None, score: float = None):
        super().__init__(parent, fg_color="transparent")
        
        # Configurar color según tipo de mensaje
        if is_user:
            bubble_color = COLORS["user_bubble"]
            text_color = COLORS["text_primary"]
            prefix = "📝 Tú:"
        else:
            bubble_color = COLORS["bg_chat"]
            text_color = COLORS.get(sentiment, COLORS["text_primary"])
            
            sentiment_emoji = {"positivo": "😊", "neutral": "😐", "negativo": "😔"}.get(sentiment, "")
            score_text = f" ({score:.1%})" if score else ""
            prefix = f"{sentiment_emoji} Análisis:"
        
        # Frame contenedor del mensaje
        message_frame = ctk.CTkFrame(self, fg_color=bubble_color, corner_radius=12)
        message_frame.pack(fill="x", padx=20, pady=5)
        
        # Label del prefijo
        prefix_label = ctk.CTkLabel(
            message_frame, 
            text=prefix,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_secondary"],
            anchor="w"
        )
        prefix_label.pack(fill="x", padx=15, pady=(10, 2))
        
        # Label del mensaje
        if is_user:
            message_text = text
        else:
            sentiment_display = sentiment.capitalize() if sentiment else ""
            score_display = f" - Confianza: {score:.1%}" if score else ""
            message_text = f"{sentiment_display}{score_display}\n\nTexto analizado: \"{text}\""
        
        message_label = ctk.CTkLabel(
            message_frame,
            text=message_text,
            font=ctk.CTkFont(size=14),
            text_color=text_color,
            anchor="w",
            justify="left",
            wraplength=600
        )
        message_label.pack(fill="x", padx=15, pady=(2, 10))


class SentimentAnalyzerGUI(ctk.CTk):
    """Interfaz gráfica principal para el analizador de sentimientos"""
    
    def __init__(self):
        super().__init__()
        
        # Configuración de la ventana
        self.title("The Smart Feedback - Análisis de Sentimientos")
        self.geometry("900x700")
        self.minsize(700, 500)
        
        # Configurar tema oscuro
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color=COLORS["bg_dark"])
        
        # Variables
        self.current_model = ctk.StringVar(value="Regresión Logística")
        self.analyzer = None
        self.messages = []
        
        # Crear interfaz
        self._create_ui()
        
        # Cargar modelo por defecto
        self._load_model()
        
        # Bind Enter para enviar
        self.bind("<Return>", lambda e: self._send_message())
    
    def _create_ui(self):
        """Crear todos los elementos de la interfaz"""
        
        # ===== ÁREA DE CHAT (Centro - Scrollable) =====
        self.chat_container = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"])
        self.chat_container.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Header
        header_frame = ctk.CTkFrame(self.chat_container, fg_color=COLORS["bg_chat"], height=60)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🧠 The Smart Feedback",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        title_label.pack(side="left", padx=20, pady=15)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Análisis de Sentimientos con IA",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        )
        subtitle_label.pack(side="left", padx=5, pady=15)
        
        # Scrollable frame para mensajes
        self.chat_scroll = ctk.CTkScrollableFrame(
            self.chat_container,
            fg_color=COLORS["bg_dark"],
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent"]
        )
        self.chat_scroll.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Mensaje de bienvenida
        self._add_welcome_message()
        
        # ===== BARRA DE INPUT (Fija abajo) =====
        self.input_bar = ctk.CTkFrame(self, fg_color=COLORS["bg_chat"], height=80)
        self.input_bar.pack(fill="x", side="bottom", padx=0, pady=0)
        self.input_bar.pack_propagate(False)
        
        # Frame interior con padding
        inner_input = ctk.CTkFrame(self.input_bar, fg_color="transparent")
        inner_input.pack(fill="both", expand=True, padx=20, pady=15)
        
        # Selector de modelo (Izquierda)
        model_frame = ctk.CTkFrame(inner_input, fg_color="transparent")
        model_frame.pack(side="left", padx=(0, 10))
        
        model_label = ctk.CTkLabel(
            model_frame,
            text="Modelo:",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        )
        model_label.pack(anchor="w")
        
        self.model_selector = ctk.CTkComboBox(
            model_frame,
            values=["Regresión Logística", "Random Forest"],
            variable=self.current_model,
            command=self._on_model_change,
            width=160,
            height=35,
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            button_color=COLORS["border"],
            button_hover_color=COLORS["accent"],
            dropdown_fg_color=COLORS["bg_input"],
            dropdown_hover_color=COLORS["accent"],
            font=ctk.CTkFont(size=12)
        )
        self.model_selector.pack()
        
        # Botones (Derecha)
        buttons_frame = ctk.CTkFrame(inner_input, fg_color="transparent")
        buttons_frame.pack(side="right", padx=(10, 0))
        
        self.send_button = ctk.CTkButton(
            buttons_frame,
            text="Enviar",
            width=80,
            height=35,
            fg_color=COLORS["accent"],
            hover_color="#0d8a6a",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._send_message
        )
        self.send_button.pack(side="left", padx=(0, 8))
        
        self.clear_button = ctk.CTkButton(
            buttons_frame,
            text="🗑️ Limpiar",
            width=90,
            height=35,
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["button_hover"],
            border_width=1,
            border_color=COLORS["border"],
            font=ctk.CTkFont(size=13),
            command=self._clear_chat
        )
        self.clear_button.pack(side="left")
        
        # Textbox (Centro - expandible)
        self.text_input = ctk.CTkTextbox(
            inner_input,
            height=50,
            fg_color=COLORS["bg_input"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            font=ctk.CTkFont(size=14),
            wrap="word"
        )
        self.text_input.pack(side="left", fill="both", expand=True, padx=10)
        self.text_input.bind("<Shift-Return>", lambda e: None)  # Permitir salto de línea con Shift+Enter
        
        # Placeholder
        self._set_placeholder()
        self.text_input.bind("<FocusIn>", self._on_focus_in)
        self.text_input.bind("<FocusOut>", self._on_focus_out)
    
    def _set_placeholder(self):
        """Establecer placeholder en el textbox"""
        self.text_input.insert("1.0", "Escribe un texto para analizar su sentimiento...")
        self.text_input.configure(text_color=COLORS["text_secondary"])
        self.placeholder_active = True
    
    def _on_focus_in(self, event):
        """Quitar placeholder al enfocar"""
        if self.placeholder_active:
            self.text_input.delete("1.0", "end")
            self.text_input.configure(text_color=COLORS["text_primary"])
            self.placeholder_active = False
    
    def _on_focus_out(self, event):
        """Restaurar placeholder si está vacío"""
        if not self.text_input.get("1.0", "end").strip():
            self._set_placeholder()
    
    def _add_welcome_message(self):
        """Añadir mensaje de bienvenida"""
        welcome_frame = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        welcome_frame.pack(fill="x", pady=40)
        
        welcome_icon = ctk.CTkLabel(
            welcome_frame,
            text="🎯",
            font=ctk.CTkFont(size=50)
        )
        welcome_icon.pack(pady=(0, 10))
        
        welcome_title = ctk.CTkLabel(
            welcome_frame,
            text="¡Bienvenido al Analizador de Sentimientos!",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        welcome_title.pack(pady=(0, 10))
        
        welcome_subtitle = ctk.CTkLabel(
            welcome_frame,
            text="Escribe un texto abajo para analizar su sentimiento.\nEl análisis te mostrará si es positivo, neutral o negativo.",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["text_secondary"],
            justify="center"
        )
        welcome_subtitle.pack()
        
        # Ejemplos
        examples_frame = ctk.CTkFrame(welcome_frame, fg_color="transparent")
        examples_frame.pack(pady=30)
        
        examples = [
            ("😊 Positivo", "\"¡Me encanta este producto, es increíble!\"", COLORS["positivo"]),
            ("😐 Neutral", "\"El producto llegó a tiempo\"", COLORS["neutral"]),
            ("😔 Negativo", "\"Muy mala calidad, no lo recomiendo\"", COLORS["negativo"])
        ]
        
        for title, example, color in examples:
            example_card = ctk.CTkFrame(examples_frame, fg_color=COLORS["bg_input"], corner_radius=10)
            example_card.pack(side="left", padx=10, pady=5)
            
            ctk.CTkLabel(
                example_card,
                text=title,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=color
            ).pack(padx=20, pady=(15, 5))
            
            ctk.CTkLabel(
                example_card,
                text=example,
                font=ctk.CTkFont(size=11),
                text_color=COLORS["text_secondary"]
            ).pack(padx=20, pady=(0, 15))
    
    def _load_model(self):
        """Cargar el modelo seleccionado"""
        model_name = self.current_model.get()
        
        try:
            if model_name == "Regresión Logística":
                self.analyzer = self._create_logistic_regression_analyzer()
            else:
                self.analyzer = self._create_random_forest_analyzer()
        except Exception as e:
            self._show_error(f"Error al cargar el modelo: {str(e)}")
    
    def _create_logistic_regression_analyzer(self) -> SentimentAnalyzer:
        """Crear analizador con Regresión Logística"""
        if not LOGISTIC_REGRESSION_MODEL_PATH.exists():
            LogisticRegressionModel.train(DATA_PATH, str(LOGISTIC_REGRESSION_MODEL_PATH))
        model = LogisticRegressionModel.load(LOGISTIC_REGRESSION_MODEL_PATH)
        return SentimentAnalyzer(model)
    
    def _create_random_forest_analyzer(self) -> SentimentAnalyzer:
        """Crear analizador con Random Forest"""
        if not RANDOM_FOREST_MODEL_PATH.exists():
            RandomForestModel.train(DATA_PATH, str(RANDOM_FOREST_MODEL_PATH))
        model = RandomForestModel.load(RANDOM_FOREST_MODEL_PATH)
        return SentimentAnalyzer(model)
    
    def _on_model_change(self, choice):
        """Callback cuando cambia el modelo"""
        self._load_model()
        self._add_system_message(f"Modelo cambiado a: {choice}")
    
    def _add_system_message(self, text: str):
        """Añadir mensaje del sistema"""
        system_frame = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        system_frame.pack(fill="x", pady=5)
        
        system_label = ctk.CTkLabel(
            system_frame,
            text=f"⚙️ {text}",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["accent"]
        )
        system_label.pack()
        
        self._scroll_to_bottom()
    
    def _send_message(self):
        """Enviar mensaje para análisis"""
        # Obtener texto
        text = self.text_input.get("1.0", "end").strip()
        
        # Validar
        if not text or self.placeholder_active:
            return
        
        # Limpiar input
        self.text_input.delete("1.0", "end")
        self.placeholder_active = False
        
        # Añadir mensaje del usuario
        user_msg = ChatMessage(self.chat_scroll, text, is_user=True)
        user_msg.pack(fill="x", pady=2)
        self.messages.append(user_msg)
        
        # Analizar
        try:
            result = self.analyzer.analyze(text)
            sentiment = result['sentiment']
            score = result['score']
            
            # Añadir respuesta
            response_msg = ChatMessage(
                self.chat_scroll, 
                text, 
                is_user=False, 
                sentiment=sentiment, 
                score=score
            )
            response_msg.pack(fill="x", pady=2)
            self.messages.append(response_msg)
            
        except Exception as e:
            self._show_error(f"Error en el análisis: {str(e)}")
        
        self._scroll_to_bottom()
    
    def _show_error(self, message: str):
        """Mostrar mensaje de error"""
        error_frame = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        error_frame.pack(fill="x", pady=5)
        
        error_label = ctk.CTkLabel(
            error_frame,
            text=f"❌ {message}",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["negativo"]
        )
        error_label.pack()
        
        self._scroll_to_bottom()
    
    def _clear_chat(self):
        """Limpiar todos los mensajes del chat"""
        for widget in self.chat_scroll.winfo_children():
            widget.destroy()
        
        self.messages.clear()
        self._add_welcome_message()
        self._add_system_message("Chat limpiado")
    
    def _scroll_to_bottom(self):
        """Scroll al final del chat"""
        self.update_idletasks()
        self.chat_scroll._parent_canvas.yview_moveto(1.0)


def main():
    app = SentimentAnalyzerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
