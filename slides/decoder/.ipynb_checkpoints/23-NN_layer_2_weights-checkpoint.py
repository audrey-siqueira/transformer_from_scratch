from manim import *
import math
from variables import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        dic = { "matrix":  {"values": nn_first_dropout[0]           , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Second Layer Input}"  , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"           , "scale": 0.3, "color": WHITE, "value": d_model*2},
                "label_y": {"string": "Tokens"                      , "scale": 0.3, "color": WHITE, "value": len(input_x)}
               }
        group_1 = build_matrix(self,dic,f=True)
        group_1.to_edge(LEFT*0.5 + UP*3, buff=0.4)
        self.play(Write(group_1))
        
        #multiple
        times = MathTex(r"\times", color=WHITE).scale(0.5).next_to(group_1[0], buff=0.4) 
        self.play(Write(times))

        
        #NN Weights 2
        dic = { "matrix":  {"values": nn_weights_2                 , "scale": 0.3, "color": YELLOW},
                "title":   {"string": "\\text{Weights (Layer 2)}"  , "scale": 0.3, "color": YELLOW},
                "label_x": {"string": "Embedding Dimensions"       , "scale": 0.3, "color": YELLOW, "value": d_model*2},
                "label_y": {"string":  "Number of Neurons"      , "scale": 0.3, "color": YELLOW, "value": d_model}
              }
        group_2 = build_matrix(self,dic,f=True)
        group_2.next_to(times, RIGHT, buff=0.5).align_to(group_1, UP)  

        left = Brace( group_2[0], direction=LEFT,buff=0.5, color=WHITE)
        right = Brace( group_2[0], direction=RIGHT,buff=0.2, color=WHITE)
        trans = Tex("T", color=WHITE).scale(0.5).next_to(right, UP + RIGHT, buff=0.05)
        group_2.add(left, right, trans)
        self.play(Write(group_2))

        
        #NN Bias 2
        dic = { "matrix":  {"values": [[x] for x in nn_bias_2]  , "scale": 0.3, "color": YELLOW},
                "title":   {"string": "\\text{Bias (Layer 2)}"  , "scale": 0.3, "color": YELLOW},
                "label_x": {"string": ""                        , "scale": 0.3, "color": YELLOW, "value": ""},
                "label_y": {"string": ""                        , "scale": 0.3, "color": YELLOW, "value": ""}
              }
        group_3 = build_matrix(self,dic,f=True)
        group_3.next_to(group_2, RIGHT, buff=0.2).align_to(group_1, UP) 
        self.play(Write(group_3))

        #equal
        equal = MathTex(r"=", color=WHITE).scale(0.5).move_to(group_1[0].get_center()  + DOWN*3)  
        self.play(Write(equal))

        #animation
        nn_calculation(self, equal, group_1, WHITE, group_2, YELLOW, group_3, YELLOW, 3)

   
        #nn output
        dic = { "matrix":  {"values": nn_output[0]                        , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Neural Network Output}"     , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"              , "scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                            , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_4 = build_matrix(self,dic,f=True)
        group_4.next_to(equal, RIGHT, buff=0.5)
        self.play(Write(group_4))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(group_4, RIGHT, buff=0.4) 
        label = Text(f"Dropout").scale(0.15).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))


        #nn output dropout
        dic = { "matrix":  {"values": output_dropout_3[0]                   , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Neural Network Output}"        , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"                 , "scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                               , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_5 = build_matrix(self,dic,f=True)
        group_5.next_to(arrow, RIGHT, buff=0.5)
        self.play(Write(group_5))

        self.wait(60)