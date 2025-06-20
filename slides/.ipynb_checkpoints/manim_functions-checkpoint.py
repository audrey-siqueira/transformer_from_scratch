from manim import *


def build_matrix(self, dic, f=False):
    formatted_matrix = [[f"{num:.4f}" if isinstance(num, (float, int)) else num for num in row] for row in dic["matrix"]["values"]]

    matrix = Matrix(formatted_matrix,h_buff=2,v_buff=1.5,).scale(dic["matrix"]["scale"]).set_color(dic["matrix"]["color"])
    if f:
        title = MathTex(dic["title"]["string"], color=dic["title"]["color"]).scale(dic["title"]["scale"]).next_to(matrix, UP)
    else:
        title = Tex(dic["title"]["string"], color=dic["title"]["color"]).scale(dic["title"]["scale"]).next_to(matrix, UP)

    group = VGroup(matrix, title)

    if dic['label_x']['string']:
        label_x = Tex(f"{dic['label_x']['string']} = {dic['label_x']['value']}", color=dic["label_x"]["color"]).scale(dic["label_x"]["scale"]).next_to(matrix, DOWN)
        group.add(label_x)
    if dic['label_y']['string']:
        label_y = Tex(f"{dic['label_y']['string']} = {dic['label_y']['value']}", color=dic["label_y"]["color"]).scale(dic["label_y"]["scale"]).next_to(matrix, LEFT)
        label_y.rotate(-PI / 2)
        label_y.next_to(matrix, LEFT)
        group.add(label_y)

    return group


def animation_R(self,equal,group_1,color_1,group_2,color_2,dis ):
    target = equal.get_center()
  

    animations = []
    for r, row_mobj in enumerate(group_1[0].get_rows()):
        row_vals = np.array([float(el.get_tex_string()) for el in row_mobj])
        row = VGroup(*row_mobj).copy().set_color(color_1).move_to(target + RIGHT)
        scalar = MathTex(r"\times", color=WHITE).scale(0.5).move_to(row.get_right() + RIGHT*0.2)

        for c, col_mobj in enumerate(group_2[0].get_columns()):
            col_vals = np.array([float(el.get_tex_string()) for el in col_mobj])
            col = VGroup(*col_mobj).copy().set_color(color_2).move_to(scalar.get_right() + RIGHT*0.4)
            equal  = MathTex(r"=", color=WHITE).scale(0.5).move_to(col.get_right() + RIGHT*0.3)
            dot = MathTex(f"{row_vals @ col_vals:.4g}", color=WHITE).scale(0.3).move_to(equal.get_right() + RIGHT*0.3)

            anim_elements = [row, scalar, col, equal, dot]

            group = VGroup(*anim_elements)
            self.play(Write(group))
            self.play(FadeOut(group, run_time=3))




def animation_T(self,equal,group_1,COLOR_1,group_2,COLOR_2,dis ):
    target = equal.get_center()
    scalar = MathTex(r"\times", color=WHITE).scale(0.5).move_to(target + RIGHT * dis)
    equal  = MathTex(r"=", color=WHITE).scale(0.5).move_to(target + RIGHT * (2*dis))

    animations = []
    for r, row_mobj in enumerate(group_1[0].get_rows()):
        row_vals = np.array([float(el.get_tex_string()) for el in row_mobj])
        row = VGroup(*row_mobj).copy().set_color(COLOR_1).move_to(target + UP * 0.25 + RIGHT * dis)

        for c, col_mobj in enumerate(group_2[0].get_rows()):
            col_vals = np.array([float(el.get_tex_string()) for el in col_mobj])
            col = VGroup(*col_mobj).copy().set_color(COLOR_2).move_to(target + DOWN * 0.25 + RIGHT * dis)

            dot = MathTex(f"{row_vals @ col_vals:.4g}", color=WHITE).scale(0.3).move_to(target + RIGHT * (2*dis+0.4))

            anim_elements = [row, scalar, col, equal, dot]

            group = VGroup(*anim_elements)
            self.play(Write(group))
            self.play(FadeOut(group, run_time=3))


def nn_calculation(self, equal, group_1, COLOR_1, group_2, COLOR_2, group_3,COLOR_3,dis):
    target = equal.get_center()
    bias_vals = [float(el.get_tex_string()) for el in group_3[0].get_columns()[0]] 
    for r, row_mobj in enumerate(group_1[0].get_rows()):
        row_vals = np.array([float(el.get_tex_string()) for el in row_mobj])
        row = VGroup(*row_mobj).copy().set_color(COLOR_1).move_to(target + UP  + RIGHT*0.9 * dis)
        scalar = MathTex(r"\times", color=WHITE).scale(0.5).next_to(row, DOWN, buff=0.2)
        operations = VGroup(row, scalar)

        for c, col_mobj in enumerate(group_2[0].get_rows()):
            col_vals = np.array([float(el.get_tex_string()) for el in col_mobj])
            col = VGroup(*col_mobj).copy().set_color(COLOR_2).move_to(target + RIGHT*0.9 * dis)
            equal = MathTex(r"=", color=WHITE).scale(0.5).next_to(col, RIGHT, buff=0.2)
            dot = MathTex(f"{row_vals @ col_vals:.4g}", color=WHITE).scale(0.3).next_to(equal, RIGHT, buff=0.2)
            plus = MathTex(r"+", color=COLOR_3).scale(0.5).next_to(dot, RIGHT, buff=0.1)

            bias_val = bias_vals[c]
            bias = MathTex(f"{bias_val:.4g}", color=COLOR_3).scale(0.3).next_to(plus, RIGHT, buff=0.1)

            igual = MathTex(r"=", color=WHITE).scale(0.5).next_to(bias, RIGHT, buff=0.1)
            result = MathTex(f"{row_vals @ col_vals +  bias_val:.4g}", color=WHITE).scale(0.3).next_to(igual, RIGHT, buff=0.1) 
        

            line = VGroup(col, equal, dot, plus, bias, igual, result)
            if len(operations) == 2:
                line.next_to(operations[0], DOWN, buff=0.45).shift(RIGHT*1.3)
            else:
                line.next_to(operations[-1], DOWN, buff=0.3)

            operations.add(line)

        self.play(Write(operations))
        self.wait(5)
        self.play(FadeOut(operations, run_time=1))
        
      



def softmax(self, equal, group_1):
    target = equal.get_center()

    for r, row_mobj in enumerate(group_1[0].get_rows()):
        row_vals = np.array([float(el.get_tex_string()) for el in row_mobj])

        exps = np.exp(row_vals)
        soft = exps / np.sum(exps)
        sum_exp = np.sum(exps)

        for i, val in enumerate(row_vals):
            # e^{x_i}
            numerator = MathTex(rf"e^{{{val:.2f}}}", color=WHITE).scale(0.3)

            # denominador e^{x1} + e^{x2} + ...
            denom_str = "+".join([f"e^{{{x:.4f}}}" for x in row_vals])
            denominator = MathTex(rf"{denom_str}", color=WHITE).scale(0.3)

            # Fração
            frac = MathTex(rf"\frac{{e^{{{val:.4f}}}}}{{{denom_str}}}", color=WHITE).scale(0.35)
            frac.move_to(target + RIGHT*1.4)

            # Resultado numérico
            result = MathTex(f"= {soft[i]:.4f}", color=WHITE).scale(0.3)
            result.next_to(frac, RIGHT, buff=0.05)

            group = VGroup(frac, result)
            self.play(Write(group))
            self.play(FadeOut(group, run_time=3))