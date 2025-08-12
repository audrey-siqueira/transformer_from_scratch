from manim import *
import math

import sys
sys.path.append("..") 
from variables import *
from manim_functions import *



class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        def split(position,string_1,arg_1,string_2,arg_2,color):

            #Query,Key,Value
            dic = { "matrix":  {"values": arg_1[0]                    , "scale": 0.3, "color": color},
                    "title":   {"string": string_1                    , "scale": 0.3, "color": color},
                    "label_x": {"string": "Embedding Dimensions"      , "scale": 0.3, "color": color, "value": d_model},
                    "label_y": {"string": "Tokens"                    , "scale": 0.3, "color": color, "value": len(input_x)}
                  }
            group = build_matrix(self,dic)
            group.to_edge(position, buff=0.3)
            self.play(Write(group))


            #arrow
            arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(group, RIGHT, buff=0.8)
            label = Text(f"Head Distribution").scale(0.2).set_color(WHITE).next_to(arrow, UP, buff=0.1)
            self.play(Write(arrow), Write(label))


            #h_Query , h_Key, h_Value
            dic = { "matrix":  {"values": arg_2[0][0]                 , "scale": 0.3, "color": color},
                    "title":   {"string": f"{string_2} 1"             , "scale": 0.3, "color": color},
                    "label_x": {"string": "Dimensions"                 , "scale": 0.3, "color": color, "value": d_model//2},
                    "label_y": {"string": "Tokens"                    , "scale": 0.3, "color": color, "value": len(input_x)}
                  }
            head_1 = build_matrix(self,dic)
            head_1.next_to(arrow, buff=0.8)
            self.play(Write(head_1))

            #h_Query , h_Key, h_Value
            dic = { "matrix":  {"values": arg_2[0][1]                 , "scale": 0.3, "color": color},
                    "title":   {"string": f"{string_2} 2"             , "scale": 0.3, "color": color},
                    "label_x": {"string": "Dimensions"                , "scale": 0.3, "color": color, "value": d_model//2},
                    "label_y": {"string": "Tokens"                    , "scale": 0.3, "color": color, "value": len(input_x)}
                  }
            head_2 = build_matrix(self,dic)
            head_2.next_to(head_1, buff=0.5)
            self.play(Write(head_2))

        split(position   = LEFT + UP*0.05 ,
              string_1   = "Query",
              arg_1      = query,
              string_2   = "Query head",
              arg_2      = h_query,
              color      = BLUE_C)

        split(position   = LEFT ,
                string_1   = "Key",
                arg_1      = key,
                string_2   = "Key head",
                arg_2      = h_key,
                color      = PURE_RED)

        split(position   = LEFT + DOWN*0.05 ,
                string_1   = "Value",
                arg_1      = value,
                string_2   = "Value head",
                arg_2      = h_value,
                color      = PURE_GREEN)



        self.wait(60)