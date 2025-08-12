from manim import *
import math

import sys
sys.path.append("..") 
from variables import *
from manim_functions import *


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        formula = MathTex(
                              r"\text{Attention} = \text{softmax}\left(",
                              r"\frac{Q.K^\top}{\sqrt{d_k}}",
                              r"\right).V",
                              color=WHITE
                          ).scale(0.5).to_edge(UP + LEFT)

        explanation = MathTex(r"\text{where:}  \quad   d_k = 2",
                              color=WHITE).scale(0.6).next_to(formula, RIGHT, buff= 2)

        self.play(Write(formula))
        self.play(Write(explanation))

        
        def split(position,arg_1,arg_2,color_2,arg_3,color_3,arg_4,order):
            #Softmax
            dic = { "matrix":  {"values": arg_1                                                                                       , "scale": 0.3, "color": WHITE},
                    "title":   {"string": "\\text{softmax}\\left( \\frac{\\mathbf{Q} \\cdot \\mathbf{K}^\\top}{\\sqrt{d_k}} \\right)" , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""                 , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"          , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_1 = build_matrix(self,dic,f=True)
            group_1.to_edge(position, buff=0.4)
            self.play(Write(group_1))

   
            #arrow
            arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).move_to(group_1[0].get_center()  + RIGHT * 1.4)  
            label = Text(f"Dropout = {dropout}").scale(0.15).set_color(WHITE).next_to(arrow, UP, buff=0.1)
            self.play(Write(arrow), Write(label))


            #Softmax Dropout
            dic = { "matrix":  {"values": arg_2                                                                                       , "scale": 0.3, "color": WHITE},
                    "title":   {"string": "\\text{softmax}\\left( \\frac{\\mathbf{Q} \\cdot \\mathbf{K}^\\top}{\\sqrt{d_k}} \\right)" , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""           , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"     , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_2 = build_matrix(self,dic,f=True)
            group_2.next_to(arrow, RIGHT, buff=0.15).align_to(group_1, DOWN) 
            self.play(Write(group_2))


            #multiple
            times = MathTex(r"\times", color=WHITE).scale(0.5).move_to(group_2[0].get_center()  + RIGHT * 1.2) 
            self.play(Write(times))

            #h_value
            dic = { "matrix":  {"values": arg_3                       , "scale": 0.3, "color": PURE_GREEN},
                    "title":   {"string": f"\\text{{Value Head {order}}}"    , "scale": 0.3, "color": PURE_GREEN},
                    "label_x": {"string": "Dimensions"      , "scale": 0.3, "color": PURE_GREEN, "value": int(d_model/2)},
                    "label_y": {"string": "Tokens"                    , "scale": 0.3, "color": PURE_GREEN, "value": len(input_x)}
                  }
            group_3 = build_matrix(self,dic,f=True)
            group_3.next_to(times, RIGHT, buff=0.2).align_to(group_1, UP).shift(DOWN * 0.27) 
            self.play(Write(group_3))

            
            #equal
            equal = MathTex(r"=", color=WHITE).scale(0.5).move_to(group_3[0].get_center()  + RIGHT) 
            self.play(Write(equal))

            animation_R(self,equal,group_2,color_2,group_3,color_3 ,0.8) 

            #AV
            dic = { "matrix":  {"values": arg_4                              , "scale": 0.3, "color": WHITE},
                    "title":   {"string": f"\\text{{Attention Head {order}}}"     , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": "Dimensions"                          , "scale": 0.3, "color": WHITE, "value": int(d_model/2)},
                    "label_y": {"string": "Tokens"                        , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_4 = build_matrix(self,dic,f=True)
            group_4.next_to(equal, RIGHT, buff=0.2).align_to(group_1, UP).shift(DOWN * 0.27)  
            self.play(Write(group_4))

            

        

        split(position   = LEFT*0.2 + UP*4,
              arg_1      = attention_scores[0][0],
              arg_2      = attention_scores_dropout[0][0],
              color_2    = WHITE,
              arg_3      = h_value[0][0],
              color_3    = PURE_GREEN,
              arg_4      = AV[0][0],
              order      = "1")

        split(position   = LEFT*0.2 + UP*12,
              arg_1      = attention_scores[0][1],
              arg_2      = attention_scores_dropout[0][1],
              color_2    = WHITE,
              arg_3      = h_value[0][1],
              color_3    = PURE_GREEN,
              arg_4      = AV[0][1],
              order      = "2")


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).shift(RIGHT *3 + DOWN*0.4)
        label = Text("Concat").scale(0.15).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))


        #AV_cont
        dic = { "matrix":  {"values": AV_cont[0]                    , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Total Attention}"     , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"        , "scale": 0.3, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                      , "scale": 0.3, "color": WHITE, "value": len(input_x)}
              }
        group_5 = build_matrix(self,dic,f=True)
        group_5.next_to(arrow, RIGHT, buff=0.4)
        self.play(Write(group_5))

        
             

        self.wait(60)