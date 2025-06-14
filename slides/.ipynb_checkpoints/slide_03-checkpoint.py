from manim import *
from manim_functions import *
from variables import *

#%%manim -qh -v WARNING generate


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        dic = { "matrix":  {"values": scaled                   , "scale": 0.4, "color": WHITE},
                "title":   {"string": "Scaled Embedding Input" , "scale": 0.5, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"   , "scale": 0.4, "color": WHITE, "value": d_model },
                "label_y": {"string": "Tokens"                 , "scale": 0.4, "color": WHITE, "value": len(embedded)}
              }
        group = build_matrix(self,dic)
        group.to_edge(UP + LEFT, buff=0.5)
        self.play(Write(group))

        #plus
        plus = MathTex(r"+", color=WHITE).scale(1).next_to(group, DOWN, buff=0.5)
        self.play(Write(plus))

        #Trim
        dic = { "matrix":  {"values": trim[0]                     , "scale": 0.4, "color": WHITE},
                "title":   {"string": "Positional Encoding Input" , "scale": 0.5, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"      , "scale": 0.4, "color": WHITE, "value": d_model },
                "label_y": {"string": "Tokens"                    , "scale": 0.4, "color": WHITE, "value": len(embedded)}
              }
        group = build_matrix(self,dic)
        group.next_to(plus, DOWN, buff=0.5)
        self.play(Write(group))

        #equal
        equal = MathTex(r"=", color=WHITE).scale(1).next_to(plus, RIGHT, buff=1.5)
        self.play(Write(equal))

        #Sum
        dic = { "matrix":  {"values": sum[0]                      , "scale": 0.4, "color": WHITE},
                "title":   {"string": "Total Input"               , "scale": 0.5, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"      , "scale": 0.4, "color": WHITE, "value": d_model },
                "label_y": {"string": "Tokens"                    , "scale": 0.4, "color": WHITE, "value": len(input)}
              }
        group = build_matrix(self,dic)
        group.next_to(equal, RIGHT, buff=0.5)
        self.play(Write(group))

        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(group, RIGHT, buff=0.3)
        label = Text(f"Dropout = {dropout}").scale(0.2).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))


        #Dropout
        dic = { "matrix":  {"values": sum_dropout[0]              , "scale": 0.4, "color": WHITE},
                "title":   {"string": "Total Input"               , "scale": 0.5, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"      , "scale": 0.4, "color": WHITE, "value": d_model },
                "label_y": {"string": "Tokens"                    , "scale": 0.4, "color": WHITE, "value": len(input)}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, RIGHT, buff=0.4)
        self.play(Write(group))

        self.wait(60)