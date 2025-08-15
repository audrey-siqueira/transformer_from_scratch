from manim import *
import math
from variables import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        #first out
        dic = { "matrix":  {"values": nn_first_out[0]                 , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{First Layer Output}"    , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"             , "scale": 0.3, "color": WHITE, "value": d_model*2},
                "label_y": {"string": "Tokens"                        , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_1 = build_matrix(self,dic,f=True)
        group_1.to_edge(UP, buff=0.1)
        self.play(Write(group_1))
        
        #arrow
        arrow = MathTex(r"\downarrow", color=WHITE).scale(1).next_to(group_1, DOWN, buff=0.3) 
        label = Text(f"Activation Function RELU").scale(0.15).set_color(WHITE).next_to(arrow, RIGHT, buff=0.1)
        self.play(Write(arrow), Write(label))

        #Relu output
        dic = { "matrix":  {"values": nn_first_activation[0]    , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{RELU output}"     , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"       , "scale": 0.3, "color": WHITE, "value": d_model*2},
                "label_y": {"string": "Tokens"                  , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_2 = build_matrix(self,dic,f=True)
        group_2.next_to(arrow, DOWN, buff=0.3)
        self.play(Write(group_2))

        #arrow
        arrow = MathTex(r"\downarrow", color=WHITE).scale(1).next_to(group_2, DOWN, buff=0.3) 
        label = Text(f"Dropout").scale(0.15).set_color(WHITE).next_to(arrow, RIGHT, buff=0.1)
        self.play(Write(arrow), Write(label))

        #Dropout output
        dic = { "matrix":  {"values": nn_first_dropout[0]           , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Second Layer Input}"  , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"           , "scale": 0.3, "color": WHITE, "value": d_model*2},
                "label_y": {"string": "Tokens"                      , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_3 = build_matrix(self,dic,f=True)
        group_3.next_to(arrow, DOWN, buff=0.3)
        self.play(Write(group_3))

    
        self.wait(60)