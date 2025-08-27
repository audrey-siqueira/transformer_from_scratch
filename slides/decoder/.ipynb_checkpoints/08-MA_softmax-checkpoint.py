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

        formula = MathTex(
                              r"\text{Attention} = \text{softmax}\left(",
                              r"\frac{Q.K^\top}{\sqrt{d_k}}",
                              r"\right).V",
                              color=WHITE
                          ).scale(0.5).to_edge(UP + LEFT)

        explanation = MathTex(r"\text{where:}  \quad   d_k = 2",
                              color=WHITE).scale(0.6).next_to(formula, RIGHT, buff= 7)

        self.play(Write(formula))
        self.play(Write(explanation))


        def split(position,p,arg_1,arg_2,arg_3,arg_4):

            #QK
            dic = { "matrix":  {"values": arg_1                                     , "scale": 0.3, "color": WHITE},
                    "title":   {"string": "\\frac{\\mathbf{Q} \\cdot \\mathbf{K}^\\top}{\\sqrt{d_k}}"  , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""                                        , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"                                  , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_1 = build_matrix(self,dic, f=True)
            group_1.to_edge(position , buff=0.3)
            self.play(Write(group_1))

            #arrow
            arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.5).next_to(group_1, RIGHT, buff=0.5).shift(DOWN * 0.3)
            label = MathTex("Mask", color=WHITE).scale(0.3).next_to(arrow, UP, buff=0.07)
            self.play(Write(arrow), Write(label))


            #Mask
            dic = { "matrix":  {"values": arg_2[0][0]                               , "scale": 0.3, "color": WHITE},
                    "title":   {"string":  f"Decoder Mask"                            , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""                                        , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"                                  , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_2 = build_matrix(self,dic, f=True)
            group_2.next_to(arrow, buff=0.35).align_to(group_1,DOWN) 
            self.play(Write(group_2))

            #equal
            equal = MathTex(r"=", color=WHITE).scale(0.5).next_to(group_2, RIGHT, buff=0.3).align_to(arrow,UP) 
            self.play(Write(equal))


            #QK masked
            dic = { "matrix":  {"values": [[r"-\infty" if x == -1e9 else x for x in row] for row in arg_3]                                                      , "scale": 0.3, "color": WHITE},
                    "title":   {"string": "\\text{masked}\\left(\\frac{\\mathbf{Q} \\cdot \\mathbf{K}^\\top}{\\sqrt{d_k}}\\right)" , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""                                                          , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"                                                    , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_3 = build_matrix(self,dic,f=True)
            group_3.next_to(equal, buff=0.4).align_to(group_1,UP) 
            self.play(Write(group_3))

            
            #arrow
            arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.5).next_to(group_3, RIGHT, buff=0.4).align_to(equal,UP) 
            label = MathTex("Softmax", color=WHITE).scale(0.3).next_to(arrow, UP, buff=0.1)
            self.play(Write(arrow), Write(label))


            #QK masked animation softmax
            dic = { "matrix":  {"values": arg_3                                                      , "scale": 0.3, "color": WHITE},
                    "title":   {"string": "\\frac{\\mathbf{Q} \\cdot \\mathbf{K}^\\top}{\\sqrt{d_k}}" , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""                                                          , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"                                                    , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_4 = build_matrix(self,dic,f=True)
            

            #softmax(self, arrow, group_4, 2)

            #Softmax
            dic = { "matrix":  {"values": arg_4                                                                                       , "scale": 0.3, "color": WHITE},
                    "title":   {"string": "\\text{softmax}\\left( \\frac{\\mathbf{Q} \\cdot \\mathbf{K}^\\top}{\\sqrt{d_k}} \\right)" , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""                 , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"          , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_5 = build_matrix(self,dic,f=True)
            group_5.next_to(arrow, buff=0.5).align_to(group_3,DOWN) 
            self.play(Write(group_5))


        split(position   = LEFT*0.2 + UP*6,
              p          = 1,
              arg_1      = attention_scores_partial_original[0][0],
              arg_2      = mask,
              arg_3      = attention_scores_masked[0][0],
              arg_4      = attention_scores[0][0])
        

        split(position   = LEFT*0.2 + DOWN*5,
              p          = 2,
              arg_1      = attention_scores_partial_original[0][1],
              arg_2      = mask,
              arg_3      = attention_scores_masked[0][1],
              arg_4      = attention_scores[0][1])
              



        self.wait(60)