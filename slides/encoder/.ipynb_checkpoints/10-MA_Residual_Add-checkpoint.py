from manim import *
import math
from variables import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        #output_dropout
        dic = { "matrix":  {"values": output_dropout_1[0]  , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Output}"         , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"   , "scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                 , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_1 = build_matrix(self,dic,f=True)
        group_1.to_edge(LEFT*3 + UP*6, buff=0.4)
        self.play(Write(group_1))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).move_to(group_1[0].get_center()  + RIGHT * 2)  
        label = Text(f"Residual Add").scale(0.15).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))

        #plus
        plus = MathTex(r"+", color=WHITE).scale(1).move_to(group_1[0].get_center() + RIGHT * 3)
        self.play(Write(plus))


        #total input dropout
        dic = { "matrix":  {"values": input_total_dropout[0]        , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Total Input Dropout}"         , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"        , "scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                      , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_2 = build_matrix(self,dic,f=True)
        group_2.next_to(plus, RIGHT, buff=0.5).align_to(group_1, DOWN) 
        self.play(Write(group_2))


        #equal
        equal = MathTex(r"=", color=WHITE).scale(0.5).move_to(group_2[0].get_center()  + RIGHT*1.8) 
        self.play(Write(equal))


        #add
        dic = { "matrix":  {"values": residual_output_1[0]                     , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Neural Network Input}"         , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"                 , "scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                               , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_3 = build_matrix(self,dic,f=True)
        group_3.next_to(equal, RIGHT, buff=0.5).align_to(group_1, DOWN) 
        self.play(Write(group_3))

        self.wait(60)