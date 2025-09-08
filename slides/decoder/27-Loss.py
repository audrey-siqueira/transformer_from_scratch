from manim import *
import math
from variables import *

import sys
sys.path.append("..") 
from manim_functions import *


class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK


        title = MathTex(r"\text{Loss Function}", color=WHITE).scale(0.5).to_edge(UP*0.3)
        self.play(Write(title))

        #Decoder Input
        formatted_vector = ", ".join(map(str, words)) 
        vector = Tex(f"[{formatted_vector}]").scale(0.35).set_color(WHITE).to_edge(LEFT+UP*3, buff=0.5)
        label_vector = Tex("Decoder Input Words", color=WHITE).scale(0.35).next_to(vector, UP)
        self.play(Write(vector), Write(label_vector))

        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1.5).next_to(vector, RIGHT, buff=0.4)
        label = Text(f"Shift Left").scale(0.2).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))

        
        #Label Words
        formatted_vector = ", ".join(map(str,['REFRÃO' ,'[EOS]', '[PAD]'] )) 
        vector = Tex(f"[{formatted_vector}]").scale(0.35).set_color(WHITE).next_to(arrow, RIGHT, buff=0.4)
        label_vector = Tex("Label Words", color=WHITE).scale(0.35).next_to(vector, UP)
        self.play(Write(vector), Write(label_vector))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1.5).next_to(vector, RIGHT, buff=0.4)
        label = Text(f"Portuguese Tokenizer").scale(0.2).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))
        

        #Label Tokens
        tokens = tokenizer_src.encode('REFRÃO [EOS] [PAD]', add_special_tokens=False).ids
        formatted_vector = r",\;".join(map(str, tokens))  
        vector = MathTex(r"\left[", formatted_vector, r"\right]").scale(0.5).set_color(WHITE).next_to(arrow, RIGHT, buff=0.4)
        label_vector = Tex("Label Tokens", color=WHITE).scale(0.35).next_to(vector, UP)
        self.play(Write(vector), Write(label_vector))

        #arrow
        arrow = MathTex(r"\downarrow", color=WHITE).scale(1).next_to(vector, DOWN, buff=0.4)
        self.play(Write(arrow))


        #Decoder Logits Output
        dic = { "matrix":  {"values": proj_output[0]         , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Transformer Output Logits}", "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Dimensions"           , "scale": 0.3, "color": WHITE, "value": proj_vocab_size},
                "label_y": {"string": "Tokens"               , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                      }
        
        vals = ellipsis_cols(proj_output[0], left=2, right=2, dots=2, symbol=r"\vdots")
        dic_disp = {**dic, "matrix": {**dic["matrix"], "values": vals}}
        
        group_1 = build_matrix(self, dic_disp, f=True)   # <- usar dic_disp aqui
        group_1.to_edge(LEFT, buff=0.2)
        self.play(Write(group_1))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(group_1, RIGHT, buff=0.4)
        self.play(Write(arrow))



        #Decoder Logits Output Softmax
        import torch
        import torch.nn.functional as F
        softmax_out = F.softmax(torch.tensor(proj_output[0]) , dim=1).tolist()


        dic = { "matrix":  {"values": softmax_out        , "scale": 0.3, "color": WHITE},
                "title":   {"string": "\\text{Transformer Output Logits Softmax}", "scale": 0.3, "color": WHITE},
                "label_x": {"string": "Dimensions"           , "scale": 0.3, "color": WHITE, "value": proj_vocab_size},
                "label_y": {"string": "Tokens"               , "scale": 0.3, "color": WHITE, "value": len(input_x)}
                      }
        
        vals = ellipsis_cols(softmax_out, left=2, right=2, dots=2, symbol=r"\vdots")
        dic_disp = {**dic, "matrix": {**dic["matrix"], "values": vals}}
        
        group_2 = build_matrix(self, dic_disp, f=True)   # <- usar dic_disp aqui
        group_2.next_to(arrow,RIGHT, buff=0.4)
        self.play(Write(group_2))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(group_2, RIGHT, buff=0.4)
        self.play(Write(arrow))


        selected_probs =  [ [round(float(softmax_out[i][tokens[i]]), 5)] for i in range(len(tokens)) ]
        
        #positions
        vector = Matrix(selected_probs,v_buff=1.5).scale(0.3).set_color(WHITE).next_to(arrow, RIGHT, buff=0.6)
        title = Tex("Label Tokens Softmax", color=WHITE).scale(0.3).next_to(vector, UP)
        label_y = Tex(f"Tokens = {len(input_x_embedded)}", color=WHITE).scale(0.3)
        label_y.rotate(-PI / 2)
        label_y.next_to(vector, LEFT, buff = 0.2)
        self.play(Write(vector), Write(title), Write(label_y))

        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).to_edge(LEFT*2+DOWN*2.5,buff=0.3)
        label = Text(f"Ignore Padding Token").scale(0.2).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))


        #positions
        vector = Matrix(selected_probs[:-1],v_buff=1.5).scale(0.3).set_color(WHITE).next_to(arrow, RIGHT, buff=1.1)
        title = Tex("Label Tokens Softmax", color=WHITE).scale(0.3).next_to(vector, UP)
        label_y = Tex(f"Tokens = {len(input_x_embedded)-1}", color=WHITE).scale(0.3)
        label_y.rotate(-PI / 2)
        label_y.next_to(vector, LEFT, buff = 0.2)
        self.play(Write(vector), Write(title), Write(label_y))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(vector, RIGHT, buff=0.5)
        label = Text(f"-Log").scale(0.2).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))



        #positions
        vector = Matrix(selected_probs[:-1],v_buff=1.5).scale(0.3).set_color(WHITE).next_to(arrow, RIGHT, buff=1.8)
        title = Tex("Label Tokens Softmax", color=WHITE).scale(0.3).next_to(vector, UP)
        label_y = Tex(f"Tokens = {len(input_x_embedded)-1}", color=WHITE).scale(0.3)
        label_y.rotate(-PI / 2)
        label_y.next_to(vector, LEFT, buff = 0.2)
        log_tex = MathTex(r"-\log", color=WHITE).scale(0.5).next_to(vector, LEFT, buff=0.8)
        lbrace = MathTex(r"\{", color=WHITE).scale(1.3).next_to(vector, LEFT, buff=0.4) 
        rbrace = MathTex(r"\}", color=WHITE).scale(1.3).next_to(vector, RIGHT, buff=0.4)
        self.play(Write(vector), Write(title), Write(label_y),Write(log_tex), Write(lbrace), Write(rbrace))


        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(vector, RIGHT, buff=1)
        self.play(Write(arrow))


        #positions
        logs=np.round([-np.log(v) for v in selected_probs[:-1]], 5) 
        vector = Matrix( logs ,v_buff=1.5).scale(0.3).set_color(WHITE).next_to(arrow, RIGHT, buff=0.8)
        title = Tex("- Log (Label Tokens Softmax)", color=WHITE).scale(0.3).next_to(vector, UP)
        label_y = Tex(f"Tokens = {len(input_x_embedded)-1}", color=WHITE).scale(0.3)
        label_y.rotate(-PI / 2)
        label_y.next_to(vector, LEFT, buff = 0.2)
        self.play(Write(vector), Write(title), Write(label_y))



        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(vector, RIGHT, buff=0.6)
        label = Text(f"Mean").scale(0.2).set_color(WHITE).next_to(arrow, UP, buff=0.1)
        self.play(Write(arrow), Write(label))


    
        #positions
        vector = Matrix(  [[np.mean(logs)]],v_buff=1.5).scale(0.3).set_color(WHITE).next_to(arrow, RIGHT, buff=0.5)
        title = Tex("Cross-Entropy Loss", color=WHITE).scale(0.3).next_to(vector, UP)
        self.play(Write(vector), Write(title))


      

      
        """
        #arrow
        arrow = MathTex(r"\longrightarrow", color=WHITE).scale(1).next_to(group, RIGHT, buff=0.5).shift(UP * 1.6)
        self.play(Write(arrow))


        #Embedded
        dic = {  "matrix":  {"values": input_x_embedded      , "scale": 0.35, "color": WHITE},
                 "title":   {"string": "Embedding Input"     , "scale": 0.35, "color": WHITE},
                 "label_x": {"string": "Embedding Dimensions", "scale": 0.35, "color": WHITE, "value": d_model },
                 "label_y": {"string": "Tokens"              , "scale": 0.35, "color": WHITE, "value": len(input_x_embedded)}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, RIGHT, buff=0.5)
        self.play(Write(group))


        #arrow
        arrow = MathTex(r"\downarrow", color=WHITE).scale(1.3).next_to(group, DOWN, buff=0.6).shift(LEFT * 0.8)
        self.play(Write(arrow))

        #square
        sqrt_d_model = MathTex(r"\times \sqrt{\substack{\text{Embedding} \\ \text{Dimensions}}} = " + f"{math.sqrt(d_model):.0f}",color=PURE_RED).scale(0.5).next_to(arrow, RIGHT, buff=0.2)
        self.play(Write(sqrt_d_model))


        #scaled
        dic = {  "matrix":  {"values": input_x_embedded_scaled      , "scale": 0.35, "color": WHITE},
                 "title":   {"string": "Scaled Embedding Input"     , "scale": 0.35, "color": WHITE},
                 "label_x": {"string": "Embedding Dimensions"       , "scale": 0.35, "color": WHITE, "value": d_model },
                 "label_y": {"string": "Tokens"                     , "scale": 0.35, "color": WHITE, "value": len(input_x_embedded)}
              }
        group = build_matrix(self,dic)
        group.next_to(arrow, DOWN, buff=0.5).shift(RIGHT * 0.8)
        self.play(Write(group))

        """

        self.wait(60)