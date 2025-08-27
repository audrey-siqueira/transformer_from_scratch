from manim import *
import math
from variables import *

import sys
sys.path.append("..") 
from manim_functions import *


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        title = MathTex(r"\text{Masked Multi-Head Attention}", color=WHITE).scale(0.5).to_edge(UP*0.001)
        self.play(Write(title))

        #formula
        formula = MathTex(
                          r"\text{Output}",
                          r"=",
                          r"\text{Total Attention}",
                          r"\times",
                          r"\mathbf{W}_o^\top",
                          color=WHITE
                          ).scale(0.45).to_edge(UP*1.5 + LEFT)


        self.play(Write(formula))


        #AV_conti
        dic = { "matrix":  {"values": AV_cont[0]                   , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Total Attention}" , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"    ,"scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                  , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_1 = build_matrix(self,dic,f=True)
        group_1.to_edge(LEFT*3 + UP*6, buff=0.4)
        self.play(Write(group_1))

   
        
        #multiple
        times = MathTex(r"\times", color=WHITE).scale(0.5).move_to(group_1[0].get_center()  + RIGHT * 1.6) 
        self.play(Write(times))


        #w_o
        dic = { "matrix":  {"values": w_o                  , "scale": 0.3, "color": YELLOW},
                "title":   {"string": "\\text{Weights O}"  , "scale": 0.3, "color": YELLOW},
                "label_x": {"string": "Embedding Dimensions"  , "scale": 0.3, "color": YELLOW, "value": d_model},
                "label_y": {"string": "Embedding Dimensions"  , "scale": 0.3, "color": YELLOW, "value": d_model}
              }
        group_2 = build_matrix(self,dic,f=True)
        group_2.next_to(times, RIGHT, buff=0.5).align_to(group_1, DOWN)

        left = Brace( group_2[0], direction=LEFT,buff=0.45, color=WHITE)
        right = Brace( group_2[0], direction=RIGHT,buff=0.45, color=WHITE)
        trans = Tex("T", color=WHITE).scale(0.5).next_to(right, UP + RIGHT, buff=0.05)
        group_2.add(left, right, trans)
        
        self.play(Write(group_2))

        #equal
        equal = MathTex(r"=", color=WHITE).scale(0.5).move_to(group_1[0].get_center()  + RIGHT*6) 
        self.play(Write(equal))

        
        #animation
        #animation_T(self,equal,group_1,WHITE,group_2,YELLOW, 1.6)

        #output
        dic = { "matrix":  {"values": output[0]         , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Output}"  , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"   , "scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"   , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_3 = build_matrix(self,dic,f=True)
        group_3.next_to(equal, RIGHT, buff=0.4).align_to(group_1, DOWN) 
        self.play(Write(group_3))


        #arrow
        arrow = MathTex(r"\downarrow", color=WHITE).scale(1).move_to(group_3.get_center()  + DOWN * 1.6)  
        label = Text(f"Dropout = {dropout}").scale(0.15).set_color(WHITE).next_to(arrow, RIGHT, buff=0.1)
        self.play(Write(arrow), Write(label))


        #output_dropout
        dic = { "matrix":  {"values": output_dropout_1[0]  , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Output}"         , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"   , "scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                 , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_4 = build_matrix(self,dic,f=True)
        group_4.next_to(arrow, DOWN, buff=0.5) 
        self.play(Write(group_4))

        
             

        self.wait(60)