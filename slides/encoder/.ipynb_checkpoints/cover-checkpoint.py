from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Título em negrito
        title = MathTex(
            r"\textbf{Transformer Encoder}", 
            color=BLACK
        ).scale(1).to_edge(UP*0.3)

        self.play(Write(title))
        
        img = ImageMobject("new_encoder.jpg")
        img.scale_to_fit_height(config.frame_height * 0.8)

        # Move a imagem para a borda inferior (com pequeno offset)
        img.to_edge(DOWN, buff=0.5)

        self.add(img)
        self.wait(20)