from manim import *
import math
from variables import *

import sys
sys.path.append("..") 
from manim_functions import *


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        
        title = MathTex(r"\text{Masked Multi-Head Attention}", color=WHITE).scale(0.4).to_edge(UP*0.001)
        self.play(Write(title))

        def mult_matrix(position,string_1,arg_1,string_2,arg_2,string_3,arg_3,color):

              #formula
              formula = MathTex(
                                rf"\text{{{string_3}}}",
                                r"=",
                                rf"\mathbf{{{string_1}}}",
                                r"\times",
                                rf"\mathbf{{W}}_{{{string_1}}}^\top",
                                color=WHITE
                            ).scale(0.35).to_edge(position)
              self.play(Write(formula))

              #equal
              equal = MathTex(r"=", color=WHITE).scale(0.5).next_to(formula, RIGHT, buff=0.2)
              self.play(Write(equal))


              #q,v,k
              dic = { "matrix":  {"values": arg_1[0]                                 , "scale": 0.3, "color": WHITE},
                      "title":   {"string": f"Normalized Total Input ({string_1})"   , "scale": 0.3, "color": WHITE},
                      "label_x": {"string": "Embedding Dimensions"                   , "scale": 0.3, "color": WHITE, "value": d_model },
                      "label_y": {"string": "Tokens"                                 , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                    }
              group_1 = build_matrix(self,dic)
              group_1.next_to(equal, RIGHT,  buff=0.2)
              self.play(Write(group_1))


              #multiple
              times = MathTex(r"\times", color=WHITE).scale(0.5).next_to(group_1, RIGHT, buff=0.2)
              self.play(Write(times))


              #weights Q,K,V
              dic = { "matrix":  {"values": arg_2                       , "scale": 0.3, "color": YELLOW},
                      "title":   {"string": f"Weights {string_2}"       , "scale": 0.3, "color": YELLOW},
                      "label_x": {"string": "Embedding Dimensions"      , "scale": 0.3, "color": YELLOW, "value": d_model},
                      "label_y": {"string": "Embedding Dimensions"      , "scale": 0.3, "color": YELLOW, "value": d_model}
                    }
              group_2 = build_matrix(self,dic)
              group_2.next_to(times, RIGHT,  buff=0.5)

              left = Brace( group_2[0], direction=LEFT,buff=0.45, color=WHITE)
              right = Brace( group_2[0], direction=RIGHT,buff=0.45, color=WHITE)
              trans = Tex("T", color=WHITE).scale(0.5).next_to(right, UP + RIGHT, buff=0.05)
              group_2.add(left, right, trans)

              self.play(Write(group_2))



              #equal
              equal = MathTex(r"=", color=WHITE).scale(0.5).next_to(group_2, RIGHT, buff=0.05)
              self.play(Write(equal))


              #animation
              #animation_T(self,equal,group_1,WHITE,group_2,YELLOW, 1.6)


              #Query,Key,Value
              dic = { "matrix":  {"values": arg_3[0]                    , "scale": 0.3, "color": color},
                      "title":   {"string": string_3                    , "scale": 0.3, "color": color},
                      "label_x": {"string": "Embedding Dimensions"      , "scale": 0.3, "color": color, "value": d_model},
                      "label_y": {"string": "Tokens"                    , "scale": 0.3, "color": color, "value": len(input_x)}
                    }
              group = build_matrix(self,dic)
              group.next_to(equal, RIGHT, buff=0.3)
              self.play(Write(group))



        mult_matrix(position   = LEFT*0.5 + UP*3.1 ,
                    string_1   = "q",
                    arg_1      = q,
                    string_2   = "Q",
                    arg_2      = w_q,
                    string_3   = "Query",
                    arg_3      = query,
                    color      = BLUE_C)


        mult_matrix(position   = LEFT*0.5 + UP*8.5 ,
                    string_1   = "k",
                    arg_1      = k,
                    string_2   = "K",
                    arg_2      = w_k,
                    string_3   = "Key",
                    arg_3      = key,
                    color      = PURE_RED)


        mult_matrix(position   = LEFT*0.5 + DOWN*1.7,
                    string_1   = "v",
                    arg_1      = v,
                    string_2   = "V",
                    arg_2      = w_v,
                    string_3   = "Value",
                    arg_3      = value,
                    color      = PURE_GREEN)


        self.wait(60)

        