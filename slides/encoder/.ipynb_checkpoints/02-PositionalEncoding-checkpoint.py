from manim import *
import math

import sys
sys.path.append("..") 
from variables import *
from manim_functions import *


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        title = Text("Positional Encoding Formulas",color=WHITE).scale(0.3).to_edge(UP + LEFT)
        self.play(Write(title))

        equal = MathTex(r"=", color=WHITE).scale(0.5).next_to(title, RIGHT, buff=0.3)
        self.play(Write(equal))

        form_1 = MathTex(
            r"PE(pos, 2i) = \sin\left(",
            r"\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}",
            r"\right)",
            color=WHITE).scale(0.5)

        comma = MathTex(",").scale(0.5).set_color(WHITE).next_to(form_1, RIGHT, buff=0.3)

        form_2 = MathTex(
            r"PE(pos, 2i+1) = \cos\left(",
            r"\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}",
            r"\right)",
            color=WHITE).scale(0.5).next_to(comma, RIGHT, buff=0.3)

        forms = VGroup(form_1,comma,form_2).next_to(equal, RIGHT, buff=0.2)        
        self.play(Write(forms))

        #positions
        vector = Matrix(np.array(position, dtype=int).tolist()).scale(0.3).set_color(WHITE).next_to(title, DOWN, buff=1.5).to_edge(LEFT, buff=0.5)
        title = Tex("Positions", color=WHITE).scale(0.4).next_to(vector, UP)
        label_y = Tex(f"Tokens = {len(input_x_embedded)}", color=WHITE).scale(0.3)
        label_y.rotate(-PI / 2)
        label_y.next_to(vector, LEFT, buff = 0.2)
        self.play(Write(vector), Write(title), Write(label_y))

        #multiple
        times = MathTex(r"\times", color=WHITE).scale(0.5).next_to(vector, RIGHT, buff=0.3)
        self.play(Write(times))


        #div_term
        form = r"\frac{1}{10000^{\frac{2i}{d_{\text{model}}}}}"
        table_data = [[f"{val:.3f}" for val in div_term for _ in (0, 1)]]

        table = Table(
            table_data,
            element_to_mobject= lambda i: Text(i, color=WHITE).scale(0.5),
            row_labels= [MathTex(form, color=WHITE).scale(1)],
            col_labels=  [ VGroup( Text(f"Dim = {d}", color=RED).scale(0.5), Text(f"i = {d//2}", color=WHITE).scale(0.5)).arrange(DOWN, buff=0.2) for d in range(d_model)],
            top_left_entry=Text("Denominator", color=WHITE).scale(0.5),
            include_outer_lines=True,
            line_config={"stroke_color": WHITE, "stroke_width": 1},
        ).scale(0.4).next_to(times, RIGHT, buff=0.3)#.shift(UP * 0.55)
        self.play(Create(table))

        #equal
        equal = MathTex(r"=", color=WHITE).scale(0.5).next_to(times, RIGHT, buff=6)
        self.play(Write(equal))

        #pos/div_term
        p = np.array(position)
        d = np.array([ val for val in div_term for _ in (0, 1)] )
        result = p * d
        result = [[f"{num:.2f}" for num in row] for row in result]
        matrix = Matrix(result,h_buff=1.5,v_buff=1,).scale(0.3).set_color(WHITE).next_to(equal, RIGHT, buff=2)
        label_matrix = MathTex(r"\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}=", color=WHITE).scale(0.5).next_to(matrix, LEFT)
        self.play(Write(matrix), Write(label_matrix))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.3).next_to(matrix, RIGHT, buff=0.2)
        self.play(Write(arrow))

        #rule
        sin_text = Tex(r"$\sin()$ if Dim is even", color=WHITE)
        cos_text = Tex(r"$\cos()$ if Dim is odd", color=WHITE)
        forms = VGroup(sin_text, cos_text).arrange(DOWN, aligned_edge=LEFT).scale(0.4)
        brace = Brace(forms, LEFT, color=WHITE).next_to(arrow, RIGHT, buff=0.1)
        forms.next_to(brace, RIGHT, buff=0.05)
        self.play(GrowFromCenter(brace), Write(forms))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.5).next_to(vector, DOWN, buff=3).to_edge(LEFT).shift(UP * 0.2)
        self.play(Write(arrow))


        #Positional Encoding Map
        dic = {  "matrix":  {"values":  pe[0]                   , "scale": 0.3, "color": PURE_GREEN},
                 "title":   {"string": "Positional Encoding Map", "scale": 0.3, "color": PURE_GREEN},
                 "label_x": {"string": "Embedding Dimensions"   , "scale": 0.3, "color": PURE_GREEN, "value": d_model },
                 "label_y": {"string": "Tokens"                 , "scale": 0.3, "color": PURE_GREEN, "value": seq_len}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, RIGHT, buff=0.6)
        self.play(Write(group))

        #Highlights
        #highlights = VGroup()
        #for i in range(len(input_x)):
        #    row = group[0].get_rows()[i]
        #    rect = SurroundingRectangle(row, color=WHITE, buff=0.1)
        #    highlights.add(rect)
        #self.play(Create(highlights))

        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(0.5).next_to(group, RIGHT, buff=0.3)
        self.play(Write(arrow))

        #position
        vector = Matrix(np.array(position, dtype=int).tolist()).scale(0.3).set_color(WHITE).next_to(arrow, RIGHT, buff=0.7)
        label_vector = Tex("Positions", color=WHITE).scale(0.3).next_to(vector, UP)
        label_y = Tex(f"Tokens = {len(input_x_embedded)}", color=WHITE).scale(0.3)
        label_y.rotate(-PI / 2)
        label_y.next_to(vector, LEFT, buff = 0.2)
        self.play(Write(vector), Write(label_vector), Write(label_y))

        #equal
        equal = MathTex(r"=", color=WHITE).scale(0.5).next_to(vector, RIGHT, buff=0.5)
        self.play(Write(equal))


        #Trim
        dic = { "matrix":  {"values": input_x_posenc[0]           , "scale": 0.3, "color": WHITE},
                "title":   {"string": "Positional Encoding Input" , "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Embedding Dimensions"      , "scale": 0.3, "color": WHITE, "value": d_model },
                "label_y": {"string": "Tokens"                    , "scale": 0.3, "color": WHITE, "value": len(input_x_embedded)}
              }
        group = build_matrix(self,dic)
        group.next_to(equal, RIGHT, buff=0.5)
        self.play(Write(group))

        self.wait(60)