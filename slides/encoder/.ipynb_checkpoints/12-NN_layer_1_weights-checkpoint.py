from manim import *
import math
from variables import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        #final input
        dic = { "matrix":  {"values": normalized_2[0]                 ,"scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Normalized Neural Network Input}"   , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"           , "scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                         , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_1 = build_matrix(self,dic,f=True)
        group_1.to_edge(LEFT*0.5 + UP*4, buff=0.4)
        self.play(Write(group_1))
        
        #multiple
        times = MathTex(r"\times", color=WHITE).scale(0.5).next_to(group_1[0], buff=0.2) 
        self.play(Write(times))

        
        #NN Weights 1
        dic = { "matrix":  {"values": nn_weights_1                 , "scale": 0.3, "color": YELLOW},
                "title":   {"string": "\\text{Weights (Layer 1)}"  , "scale": 0.3, "color": YELLOW},
                "label_x": {"string": "Embedding Dimensions"       , "scale": 0.3, "color": YELLOW, "value": d_model},
                "label_y": {"string": "Number of Neurons"          , "scale": 0.3, "color": YELLOW, "value": d_model*2}
              }
        group_2 = build_matrix(self,dic,f=True)
        group_2.next_to(times, RIGHT, buff=0.3).align_to(group_1, UP)  

        left = Brace( group_2[0], direction=LEFT,buff=0.35, color=WHITE)
        right = Brace( group_2[0], direction=RIGHT,buff=0.2, color=WHITE)
        trans = Tex("T", color=WHITE).scale(0.5).next_to(right, UP + RIGHT, buff=0.05)
        group_2.add(left, right, trans)
        self.play(Write(group_2))

        
        #NN Bias 1
        dic = { "matrix":  {"values": [[x] for x in nn_bias_1]  , "scale": 0.3, "color": YELLOW},
                "title":   {"string": "\\text{Bias (Layer 1)}"  , "scale": 0.3, "color": YELLOW},
                "label_x": {"string": ""                        , "scale": 0.3, "color": YELLOW, "value": ""},
                "label_y": {"string": ""                        , "scale": 0.3, "color": YELLOW, "value": ""}
              }
        group_3 = build_matrix(self,dic,f=True)
        group_3.next_to(group_2, RIGHT, buff=0.15).align_to(group_1, UP) 
        self.play(Write(group_3))

        #equal
        equal = MathTex(r"=", color=WHITE).scale(0.5).move_to(group_3[0].get_center()  + RIGHT*0.6).align_to(times, UP)  
        self.play(Write(equal))

        #animation
        nn_calculation(self, equal, group_1, WHITE, group_2, YELLOW, group_3, YELLOW, 1.6)

   
        #first out
        dic = { "matrix":  {"values": nn_first_out[0]                , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{First Layer Output}"   , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"            , "scale": 0.3, "color": WHITE, "value": d_model*2},
                "label_y": {"string": "Tokens"                       , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_4 = build_matrix(self,dic,f=True)
        group_4.next_to(equal, RIGHT, buff=0.2).align_to(group_1, UP) 
        self.play(Write(group_4))

        self.wait(60)