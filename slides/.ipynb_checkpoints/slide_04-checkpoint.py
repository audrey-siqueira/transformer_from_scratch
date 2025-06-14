from manim import *
from manim_functions import *
from variables import *

#%%manim -ql -v WARNING generate


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        formula = MathTex(r"\text{Normalization(x)} = \alpha \cdot \frac{x - \mu}{\sigma + \varepsilon} + \text{bias}",
                          color=WHITE).scale(0.7).to_edge(UP + LEFT)
        

        explanation = MathTex(r"\text{where:} \quad \alpha = 1 \quad \varepsilon = 10^{-6}  \quad   \text{bias} = 0",
                              color=WHITE).scale(0.6).next_to(formula, RIGHT, buff= 2)

        self.play(Write(formula))
        self.play(Write(explanation))
       

        #total input dropout
        dic = { "matrix":  {"values": sum_dropout[0]             ,"scale": 0.4, "color": WHITE},
                "title":   {"string": "\\text{Total Input}"      , "scale": 0.4, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"    , "scale": 0.4, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                  , "scale": 0.4, "color": WHITE, "value": len(input)}
              }
        group_1 = build_matrix(self,dic,f=True)
        group_1.to_edge(LEFT*0.2 + UP*6, buff=0.4)
        self.play(Write(group_1))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.6).move_to(group_1[0].get_center()  + RIGHT*2.3)  
        label = Text("Normalize").scale(0.15).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))
        
        target = arrow.get_center()
  
        animations = []
        for r, row_mobj in enumerate(group_1[0].get_rows()):
            row_vals = np.array([float(el.get_tex_string()) for el in row_mobj])
            row = VGroup(*row_mobj).copy().set_color(BLACK).move_to(target + RIGHT*1.5)

            mean_val = np.mean(row_vals)
            num_str = " + ".join([f"{x:.4f}" for x in row_vals])
            mean_calc = MathTex(r"\mu = \frac{" + num_str + "}{" + str(len(row_vals)) + "} = " + f"{mean_val:.4f}", color=PURE_RED)
            mean_calc.scale(0.4).next_to(target, RIGHT + UP*3, buff=0.4)


            std_val = np.std(row_vals)  
            diff_squares = [(x - mean_val)**2 for x in row_vals]
            diff_squares_str = " + ".join([f"({x:.4f} - {mean_val:.4f})^2" for x in row_vals])
        
            std_calc = MathTex(r"\sigma = \sqrt{\frac{" + diff_squares_str + "}{" + str(len(row_vals)) + "}} = " + f"{std_val:.4f}", color=PURE_GREEN)
            std_calc.scale(0.4).next_to(target, RIGHT + DOWN*0.5, buff=0.4)
       

            anim_elements = [mean_calc , std_calc]
            group = VGroup(*anim_elements)
            self.play(Write(group))


            E = 0.000001
            alpha = 1.0
            bias  = 0.0
            norm_calcs = []
            for i, x in enumerate(row_vals):
                norm_val = alpha * (x - mean_val) / (std_val + E) + bias
                norm_expr = MathTex( r"\text{Norm}(x_{" + str(i+1) + r"}) = " +f"{alpha:.1f} \\cdot \\frac{{{x:.4f} - {mean_val:.4f}}}{{{std_val:.4f} + {E}}} + {bias:.1f} = {norm_val:.4f}",color=WHITE)
                norm_expr.scale(0.4).next_to(target, RIGHT + DOWN*5, buff=0.4)

                sub_group = VGroup(norm_expr)
                self.play(Write(sub_group))
                self.play(FadeOut(sub_group, run_time=3))
    
            self.play(FadeOut(group))

        #final input
        dic = { "matrix":  {"values": normalized[0]                        ,"scale": 0.4, "color": WHITE},
                "title":   {"string": "\\text{Normalized Total Input}"     , "scale": 0.4, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"               , "scale": 0.4, "color": WHITE, "value": d_model},
                "label_y": {"string": "Tokens"                             , "scale": 0.4, "color": WHITE, "value": len(input)}
              }
        group_2 = build_matrix(self,dic,f=True)
        group_2.next_to(group_1,RIGHT*3.5, buff=0.4)
        self.play(Write(group_2))

      

        self.wait(60)