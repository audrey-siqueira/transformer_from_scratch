from manim import *
from manim_functions import *
from variables import *
import math


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        dic = { "matrix":  {"values": input_x_embedded_scaled  , "scale": 0.3, "color": WHITE},
                "title":   {"string": "Scaled Embedding Input" , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"   , "scale": 0.3, "color": WHITE, "value": d_model },
                "label_y": {"string": "Tokens"                 , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group = build_matrix(self,dic)
        group.to_edge(UP + LEFT, buff=0.5)
        self.play(Write(group))

        #plus
        plus = MathTex(r"+", color=WHITE).scale(0.7).next_to(group, DOWN, buff=0.5)
        self.play(Write(plus))

        #Trim
        dic = { "matrix":  {"values": input_x_posenc[0]           , "scale": 0.3, "color": WHITE},
                "title":   {"string": "Positional Encoding Input" , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"      , "scale": 0.3, "color": WHITE, "value": d_model },
                "label_y": {"string": "Tokens"                    , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group = build_matrix(self,dic)
        group.next_to(plus, DOWN, buff=0.5)
        self.play(Write(group))

        #equal
        equal = MathTex(r"=", color=WHITE).scale(0.7).next_to(plus, RIGHT, buff=1.5)
        self.play(Write(equal))

        #Sum
        dic = { "matrix":  {"values": input_total[0]              , "scale": 0.3, "color": WHITE},
                "title":   {"string": "Total Input"               , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"      , "scale": 0.3, "color": WHITE, "value": d_model },
                "label_y": {"string": "Tokens"                    , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group = build_matrix(self,dic)
        group.next_to(equal, RIGHT, buff=0.5)
        self.play(Write(group))

        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(group, RIGHT, buff=0.3)
        label = Text(f"Dropout = {dropout}").scale(0.2).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))


        #Dropout
        dic = { "matrix":  {"values": input_total_dropout[0]      , "scale": 0.3, "color": WHITE},
                "title":   {"string": "Total Input"               , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"      , "scale": 0.3, "color": WHITE, "value": d_model },
                "label_y": {"string": "Tokens"                    , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, RIGHT, buff=0.4)
        self.play(Write(group))

        self.wait(60)