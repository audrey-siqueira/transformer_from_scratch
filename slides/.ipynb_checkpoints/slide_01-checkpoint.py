from manim import *
from manim_functions import *
from variables import *
import math


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        #Input Tokens
        formatted_vector = [[f"{num}" for num in input]]
        vector = Matrix(formatted_vector).scale(0.45).set_color(WHITE).to_edge(LEFT, buff=0.2)
        label_vector = Tex("Input Tokens", color=WHITE).scale(0.5).next_to(vector, UP)
        self.play(Write(vector), Write(label_vector))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(vector, RIGHT, buff=0.5)
        self.play(Write(arrow))


        matrix_lines = embedding_map[:9] + [["\\vdots"]*d_model] + embedding_map[-1:]
        #Embedding Map
        dic = { "matrix":  {"values": matrix_lines         , "scale": 0.4, "color": PURE_GREEN},
                "title":   {"string": "Embedding Map"       , "scale": 0.5, "color": PURE_GREEN},
                "label_x": {"string": "Embedding Dimensions", "scale": 0.4, "color": PURE_GREEN, "value": d_model },
                "label_y": {"string": "Embedding Map Size" , "scale": 0.4, "color": PURE_GREEN, "value": vocab_size}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, RIGHT, buff=0.5)
        self.play(Write(group))

        #Highlights
        highlights = VGroup()
        for i in input:
            row =  group[0].get_rows()[i]
            rect = SurroundingRectangle(row, color=WHITE, buff=0.1)
            highlights.add(rect)
        self.play(Create(highlights))

        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(group, RIGHT, buff=0.5).shift(UP * 2.1)
        self.play(Write(arrow))


        #Embedded
        dic = {  "matrix":  {"values": embedded              , "scale": 0.4, "color": WHITE},
                 "title":   {"string": "Embedding Input"     , "scale": 0.5, "color": WHITE},
                 "label_x": {"string": "Embedding Dimensions", "scale": 0.4, "color": WHITE, "value": d_model },
                 "label_y": {"string": "Tokens"              , "scale": 0.4, "color": WHITE, "value": len(embedded)}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, RIGHT, buff=0.5)
        self.play(Write(group))


        #arrow
        arrow = MathTex(r"\downarrow", color=WHITE).scale(1.3).next_to(group, DOWN, buff=0.5).shift(LEFT * 0.8)
        self.play(Write(arrow))

        #square
        sqrt_d_model = MathTex(r"\times \sqrt{\substack{\text{Embedding} \\ \text{Dimensions}}} = " + f"{math.sqrt(d_model):.0f}",color=PURE_RED).scale(0.5).next_to(arrow, RIGHT, buff=0.2)
        self.play(Write(sqrt_d_model))


        #scaled
        dic = {  "matrix":  {"values": scaled                       , "scale": 0.4, "color": WHITE},
                 "title":   {"string": "Scaled Embedding Input"     , "scale": 0.5, "color": WHITE},
                 "label_x": {"string": "Embedding Dimensions"       , "scale": 0.4, "color": WHITE, "value": d_model },
                 "label_y": {"string": "Tokens"                     , "scale": 0.4, "color": WHITE, "value": len(embedded)}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, DOWN, buff=0.5).shift(RIGHT * 0.8)
        self.play(Write(group))

        self.wait(60)
