from manim import *
from manim_functions import *
from variables import *
import math



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
                              color=WHITE).scale(0.6).next_to(formula, RIGHT, buff= 7)

        self.play(Write(formula))
        self.play(Write(explanation))


        def split(position,p,arg_1,color_1,arg_2,color_2,arg_3,arg_4,arg_5):


            #h_Query
            dic = { "matrix":  {"values": arg_1                   , "scale": 0.3, "color": color_1},
                    "title":   {"string": f"Query Head {p}"           , "scale": 0.3, "color":  color_1},
                    "label_x": {"string": "Dimensions"           , "scale": 0.3, "color": color_1, "value": d_model//2},
                    "label_y": {"string": "Tokens"                , "scale": 0.3, "color": color_1, "value": len(input_x)}
                  }
            group_1 = build_matrix(self,dic)
            group_1.to_edge(position , buff=0.3)
            self.play(Write(group_1))

            #multiple
            times = MathTex(r"\times", color=WHITE).scale(0.5).next_to(group_1, RIGHT, buff=0.1)
            self.play(Write(times))

            #h_Key
            dic = { "matrix":  {"values": arg_2                  , "scale": 0.3, "color": color_2},
                    "title":   {"string": f"Key Head {p}"          , "scale": 0.3, "color": color_2},
                    "label_x": {"string": "Dimensions"          , "scale": 0.3, "color": color_2, "value": d_model//2},
                    "label_y": {"string": "Tokens"               , "scale": 0.3, "color": color_2, "value": len(input_x)}
                  }
            group_2 = build_matrix(self,dic)
            group_2.next_to(times, buff=0.35)

            left = Brace( group_2[0], direction=LEFT,buff=0.3, color=WHITE)
            right = Brace( group_2[0], direction=RIGHT,buff=0.3, color=WHITE)
            trans = Tex("T", color=WHITE).scale(0.3).next_to(right, UP + RIGHT, buff=0.05)
            group_2.add(left, right, trans)

            self.play(Write(group_2))


            #equal
            equal = MathTex(r"=", color=WHITE).scale(0.5).next_to(group_2, RIGHT, buff=0.05)
            self.play(Write(equal))


            #animation
            animation_T(self,equal,group_1,color_1,group_2,color_2,0.8)


            #QK
            dic = { "matrix":  {"values": arg_3                                     , "scale": 0.3, "color": WHITE},
                    "title":   {"string": "\\mathbf{Q} \\cdot \\mathbf{K}^{\\top}"  , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""                                        , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"                                  , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_3 = build_matrix(self,dic, f=True)
            group_3.next_to(equal, buff=0.2).align_to(group_1,UP) 
            self.play(Write(group_3))


            #arrow
            arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.5).next_to(group_3, RIGHT, buff=0.1).align_to(equal,UP) 
            label = MathTex(r"\frac{\text{1}}{\sqrt{d_k}}", color=WHITE).scale(0.3).next_to(arrow, UP, buff=0.1)
            self.play(Write(arrow), Write(label))


            #QK/root
            dic = { "matrix":  {"values": arg_4                                                       , "scale": 0.3, "color": WHITE},
                    "title":   {"string": "\\frac{\\mathbf{Q} \\cdot \\mathbf{K}^\\top}{\\sqrt{d_k}}" , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""                                                          , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"                                                    , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_4 = build_matrix(self,dic,f=True)
            group_4.next_to(arrow, buff=0.2).align_to(group_3,DOWN) 
            self.play(Write(group_4))


            #arrow
            arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.5).next_to(group_4, RIGHT, buff=0.2).align_to(equal,UP) 
            label = MathTex(r"Softmax", color=WHITE).scale(0.3).next_to(arrow, UP, buff=0.1)
            self.play(Write(arrow), Write(label))

            softmax(self, arrow, group_4)

            #Softmax
            dic = { "matrix":  {"values": arg_5                                                                                       , "scale": 0.3, "color": WHITE},
                    "title":   {"string": "\\text{softmax}\\left( \\frac{\\mathbf{Q} \\cdot \\mathbf{K}^\\top}{\\sqrt{d_k}} \\right)" , "scale": 0.3, "color": WHITE},
                    "label_x": {"string": ""                 , "scale": 0.3, "color": WHITE, "value": ""},
                    "label_y": {"string": "Tokens"          , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                  }
            group_5 = build_matrix(self,dic,f=True)
            group_5.next_to(arrow, buff=0.4).align_to(group_3,DOWN) 
            self.play(Write(group_5))






        split(position   = LEFT*0.2 + UP*6,
              p          = 1,
              arg_1      = h_query[0][0],
              color_1    = BLUE_C,
              arg_2      = h_key[0][0],
              color_2    = PURE_RED,
              arg_3      = QK[0][0],
              arg_4      = attention_scores_partial[0][0],
              arg_5      = attention_scores[0][0])
        

        split(position   = LEFT*0.2 + DOWN*5,
              p          = 2,
              arg_1      = h_query[0][1],
              color_1    = BLUE_C,
              arg_2      = h_key[0][1],
              color_2    = PURE_RED,
              arg_3      = QK[0][1],
              arg_4      = attention_scores_partial[0][1],
              arg_5      = attention_scores[0][1])
              



        self.wait(60)