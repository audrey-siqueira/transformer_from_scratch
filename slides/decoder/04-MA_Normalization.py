from manim import *
import math
from variables import *

import sys
sys.path.append("..") 
from manim_functions import *



class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        formula = MathTex(r"\text{Normalization(x)} = \alpha \cdot \frac{x - \mu}{\sigma + \varepsilon} + \text{bias}",
                          color=WHITE).scale(0.7).to_edge(UP + LEFT)
        

        explanation = MathTex(r"\text{where:} \quad \varepsilon = 10^{-6}",
                              color=WHITE).scale(0.6).next_to(formula, RIGHT, buff= 5.5)

        self.play(Write(formula))
        self.play(Write(explanation))
       

        #total input dropout
        dic = { "matrix":  {"values": input_total_dropout[0]    ,"scale": 0.35, "color": WHITE},
                "title":   {"string": "\\text{Total Input}"     , "scale": 0.35, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"    , "scale": 0.35, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                  , "scale": 0.35, "color": WHITE, "value": len(input_x)}
              }
        group_1 = build_matrix(self,dic,f=True)
        group_1.to_edge(LEFT*0.2 + UP*7, buff=0.4)
        self.play(Write(group_1))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.6).move_to(group_1[0].get_center()  + RIGHT*2.1)  
        label = Text("Normalize").scale(0.15).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))
        
        target = arrow.get_center()
  
        animations = []
        norm_matrix = []
        for r, row_mobj in enumerate(group_1[0].get_rows()):
            row_vals = np.array([float(el.get_tex_string()) for el in row_mobj])
            row_vals = np.array(row_vals, dtype=np.float64)
            row = VGroup(*row_mobj).copy().set_color(BLACK).move_to(target + RIGHT*1.5)

            mean_val = np.mean(row_vals,dtype=np.float64)
            num_str = " + ".join([f"{x:.4f}" for x in row_vals])
            mean_calc = MathTex(r"\mu = \frac{" + num_str + "}{" + str(len(row_vals)) + "} = " + f"{mean_val:.4f}", color=PURE_RED)
            mean_calc.scale(0.4).next_to(target, RIGHT + UP*3, buff=0.4)


            std_val = np.std(row_vals, ddof=1, dtype=np.float64)  
            diff_squares = [(x - mean_val)**2 for x in row_vals]
            diff_squares_str = " + ".join([f"({x:.4f} - {mean_val:.4f})^2" for x in row_vals])
        
            std_calc = MathTex(r"\sigma = \sqrt{\frac{" + diff_squares_str + r"}{" + str(len(row_vals)) + r"-1}}" + " = " + f"{std_val:.4f}",color=PURE_GREEN)
            std_calc.scale(0.4).next_to(target, RIGHT + DOWN*0.5, buff=0.4)
       

            anim_elements = [mean_calc , std_calc]
            group = VGroup(*anim_elements)
            self.play(Write(group))


            E = 0.000001
            norm_row = []
            for i, x in enumerate(row_vals):                
                norm_val = (x - mean_val) / (std_val + E) 
                norm_expr = MathTex( r"\text{Norm}(x_{" + str(i+1) + r"}) = " +f"\\frac{{{x:.4f} - {mean_val:.4f}}}{{{std_val:.4f} + {E}}}  = {norm_val:.4f}",color=WHITE)
                norm_expr.scale(0.4).next_to(target, RIGHT + DOWN*5, buff=0.4)            
                sub_group = VGroup(norm_expr)
                self.play(Write(sub_group))
                self.play(FadeOut(sub_group, run_time=3))

                norm_row.append(float(norm_val))

            norm_matrix.append(norm_row)
    
            self.play(FadeOut(group))


        #partial norm
        dic = { "matrix":  {"values": norm_matrix                          ,"scale": 0.35, "color": WHITE},
                "title":   {"string": "\\text{Partial Normalization}"     , "scale": 0.35, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"               , "scale": 0.35, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                             , "scale": 0.35, "color": WHITE, "value": len(input_x)}
              }
        group_2 = build_matrix(self,dic,f=True)
        group_2.next_to(group_1,RIGHT*3.3, buff=0.4)
        self.play(Write(group_2))



        #multiple
        times = MathTex(r"\times", color=WHITE).scale(0.7).next_to(group_2[0], UP, buff=0.7)
        self.play(Write(times))


        #Alpha
        formatted_vector = [[f"{num:.4f}" for num in alpha_1]]
        vector = Matrix(formatted_vector, h_buff=2).scale(0.35).set_color(YELLOW).next_to(times, UP, buff=0.2)
        label_vector = Tex("Alpha", color=YELLOW).scale(0.35).next_to(vector, UP*0.5)
        self.play(Write(vector), Write(label_vector))



        #plus
        plus = MathTex(r"+", color=WHITE).scale(0.6).next_to(group_2[0], DOWN, buff=0.7)
        self.play(Write(plus))


        #Bias
        formatted_vector = [[f"{num:.4f}" for num in bias_1]]
        vector = Matrix(formatted_vector, h_buff=2).scale(0.35).set_color(YELLOW).next_to(plus, DOWN, buff=0.5)
        label_vector = Tex("Bias", color=YELLOW).scale(0.35).next_to(vector, UP*0.5)
        self.play(Write(vector), Write(label_vector))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.6).move_to(group_2[0].get_center()  + RIGHT*2.1)  
        label = Text("Normalize").scale(0.15).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))



        #final input
        dic = { "matrix":  {"values": norm_x_1[0]                          ,"scale": 0.35, "color": WHITE},
                "title":   {"string": "\\text{Normalized Total Input}"     , "scale": 0.35, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"               , "scale": 0.35, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                             , "scale": 0.35, "color": WHITE, "value": len(input_x)}
              }
        group_3 = build_matrix(self,dic,f=True)
        group_3.next_to(group_2,RIGHT*3.3, buff=0.4)
        self.play(Write(group_3))

      

        self.wait(60)