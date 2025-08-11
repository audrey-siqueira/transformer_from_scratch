from manim import *
import math

import sys
sys.path.append("..") 
from variables import *
from manim_functions import *


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        #Input Tokens
        formatted_vector = r",\;".join(map(str, input_x))  
        vector = MathTex(r"\left[", formatted_vector, r"\right]").scale(0.5).set_color(WHITE).to_edge(LEFT, buff=0.5)
        label_vector = Tex("Input Tokens", color=WHITE).scale(0.35).next_to(vector, UP)
        self.play(Write(vector), Write(label_vector))



        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(vector, RIGHT, buff=0.7)
        self.play(Write(arrow))

        #Adding ... to embedding map
        matrix_lines = []
        order = sorted(input_x)
        matrix_lines.append(embedding_map[0])
        for idx in order:
            matrix_lines.append(["\\vdots"] * d_model)
            matrix_lines.append(embedding_map[idx])
        matrix_lines.append(["\\vdots"] * d_model)
        matrix_lines.append(embedding_map[-1])
        
        #Embedding Map
        dic = { "matrix":  {"values": matrix_lines          , "scale": 0.35, "color": PURE_GREEN},
                "title":   {"string": "Embedding Map"       , "scale": 0.35, "color": PURE_GREEN},
                "label_x": {"string": "Embedding Dimensions", "scale": 0.35, "color": PURE_GREEN, "value": d_model },
                "label_y": {"string": "Tokenizer Size"      , "scale": 0.35, "color": PURE_GREEN, "value": vocab_size}
              }
        group = build_matrix(self,dic)
        
        
        rows = group[0].get_rows()

        #idxs
        labels = [0, ""]  
        for val in order:
            labels.append(str(val))
            labels.append("")
        labels.append(len(embedding_map)-1)
        
        row_indices = VGroup(*[Tex(str(label), color=PURE_GREEN).scale(0.3).next_to(rows[i], LEFT, buff=0.35)
                      for i, label in enumerate(labels)])
        group.add(row_indices)
        group[3].shift(LEFT*0.3)
        
        group.next_to(arrow, RIGHT, buff=0.5)
        self.play(Write(group))

        #white squares
        highlights = VGroup()
        indices = list(range(2, len(rows), 2))[:3]
        for i in indices:
            rect = SurroundingRectangle(rows[i], color=WHITE, buff=0.1)
            highlights.add(rect)
        
        self.play(Create(highlights))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(group, RIGHT, buff=0.5).shift(UP * 1.6)
        self.play(Write(arrow))


        #Embedded
        dic = {  "matrix":  {"values": input_x_embedded      , "scale": 0.35, "color": WHITE},
                 "title":   {"string": "Embedding Input"     , "scale": 0.35, "color": WHITE},
                 "label_x": {"string": "Embedding Dimensions", "scale": 0.35, "color": WHITE, "value": d_model },
                 "label_y": {"string": "Tokens"              , "scale": 0.35, "color": WHITE, "value": len(input_x_embedded)}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, RIGHT, buff=0.5)
        self.play(Write(group))


        #arrow
        arrow = MathTex(r"\downarrow", color=WHITE).scale(1.3).next_to(group, DOWN, buff=0.6).shift(LEFT * 0.8)
        self.play(Write(arrow))

        #square
        sqrt_d_model = MathTex(r"\times \sqrt{\substack{\text{Embedding} \\ \text{Dimensions}}} = " + f"{math.sqrt(d_model):.0f}",color=PURE_RED).scale(0.5).next_to(arrow, RIGHT, buff=0.2)
        self.play(Write(sqrt_d_model))


        #scaled
        dic = {  "matrix":  {"values": input_x_embedded_scaled      , "scale": 0.35, "color": WHITE},
                 "title":   {"string": "Scaled Embedding Input"     , "scale": 0.35, "color": WHITE},
                 "label_x": {"string": "Embedding Dimensions"       , "scale": 0.35, "color": WHITE, "value": d_model },
                 "label_y": {"string": "Tokens"                     , "scale": 0.35, "color": WHITE, "value": len(input_x_embedded)}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, DOWN, buff=0.5).shift(RIGHT * 0.8)
        self.play(Write(group))

        self.wait(60)