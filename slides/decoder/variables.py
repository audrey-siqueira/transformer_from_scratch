from pathlib import Path
import json


SLIDES_DIR = Path(__file__).resolve().parent.parent      # .../slides
VARS_DIR   = SLIDES_DIR / "vars"                  # .../slides/vars

def load(name: str):
    path = VARS_DIR / name                        # sempre em slides/vars
    # print(f"Lendo: {path}")  # debug opcional
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



#DECODER INPUT EMBEDDING
InputEmbedding = load("InputEmbedding_1.json")
d_model          = InputEmbedding["d_model"]
vocab_size       = InputEmbedding["vocab_size"]
embedding_map    = InputEmbedding["embedding_weight"]
input_x          = InputEmbedding["input_x"][0]
input_x_embedded = InputEmbedding["embedded"][0]
input_x_embedded_scaled   = InputEmbedding["scaled"][0]


from tokenizers import Tokenizer
path = "/mnt/custom-file-systems/efs/fs-0a0b857e0cd732c6d_fsap-0c621f83c851d0d4c/transformer/tokenizers/tokenizer_pt.json"
tokenizer_src = Tokenizer.from_file(str(path))
words = tokenizer_src.decode([int(i) for i in input_x], skip_special_tokens=False)


#DECODER POSITIONAL ENCODING
PositionalEncoding = load("PositionalEncoding_1.json")
seq_len               = PositionalEncoding["seq_len"]
dropout               = PositionalEncoding["dropout"]
position              = PositionalEncoding["position"]  
pe                    = PositionalEncoding["posenc"]
div_term              = PositionalEncoding["div_term"]
input_x_posenc        = PositionalEncoding["trim"]
input_total           = PositionalEncoding["sum"]
input_total_dropout   = PositionalEncoding["sum_dropout"]




#DECODER MULTIHEAD ATTENTION NORMALIZATION
Normalization_Multihead_Attention = load("LayerNormalization_3.json")
eps_1        = Normalization_Multihead_Attention["eps"]
alpha_1      = Normalization_Multihead_Attention["alpha"]
bias_1       = Normalization_Multihead_Attention["bias"]
mean_1       = Normalization_Multihead_Attention["mean"]
std_1        = Normalization_Multihead_Attention["std"]
normalized_1 = Normalization_Multihead_Attention["normalization"]



#DECODER MULTIHEAD ATTENTION 
Multihead_Attention = load("MultiHeadAttentionBlock_1.json")

mask    = Multihead_Attention["mask"]

h       = Multihead_Attention["h"]
d_k     = Multihead_Attention["d_k"]

q       = Multihead_Attention["q"]
w_q     = Multihead_Attention["w_q_weight"]
query   = Multihead_Attention["query"]
h_query = Multihead_Attention["h_query"]

k       = Multihead_Attention["k"]
w_k     = Multihead_Attention["w_k_weight"]
key     = Multihead_Attention["key"]
h_key   = Multihead_Attention["h_key"]


v       = Multihead_Attention["v"]
w_v     = Multihead_Attention["w_v_weight"]
value   = Multihead_Attention["value"]
h_value = Multihead_Attention["h_value"]      

QK                                 = Multihead_Attention["QK"]
attention_scores_partial_original  = Multihead_Attention["attention_scores_partial_original"]
attention_scores_partial           = Multihead_Attention["attention_scores_partial"]
attention_scores_masked            = Multihead_Attention["attention_scores_masked"]
attention_scores                   = Multihead_Attention["attention_scores"]
attention_scores_dropout           = Multihead_Attention["attention_scores_dropout"]

AV = Multihead_Attention["attention_values"]

AV_cont  = Multihead_Attention["AV_cont"]
w_o      = Multihead_Attention["w_o_weight"]
output   = Multihead_Attention["output"]


#DECODER MULTIHEAD ATTENTION  RESIDUAL ADD 
Residual_Multihead_Attention = load("ResidualConnection_2.json")
dropout_p_1       = Residual_Multihead_Attention["dropout_p"]
input_x_1         = Residual_Multihead_Attention["input_x"]
norm_x_1          = Residual_Multihead_Attention["norm_x"]
output_1          = Residual_Multihead_Attention["layer_output"]
output_dropout_1  = Residual_Multihead_Attention["layer_output_dropout"]
residual_output_1 = Residual_Multihead_Attention["add"]


#DECODER CROSS NORMALIZATION
Normalization_Neural_Network = load("LayerNormalization_4.json")
eps_2        = Normalization_Neural_Network["eps"]
alpha_2      = Normalization_Neural_Network["alpha"]
bias_2       = Normalization_Neural_Network["bias"]
mean_2       = Normalization_Neural_Network["mean"]
std_2        = Normalization_Neural_Network["std"]
normalized_2 = Normalization_Neural_Network["normalization"]



#DECODER CROSS
Cross_Attention = load("MultiHeadAttentionBlock_2.json")

cross_mask    = Cross_Attention["mask"]

cross_h       = Cross_Attention["h"]
cross_d_k     = Cross_Attention["d_k"]

cross_q       = Cross_Attention["q"]
cross_w_q     = Cross_Attention["w_q_weight"]
cross_query   = Cross_Attention["query"]
cross_h_query = Cross_Attention["h_query"]

cross_k       = Cross_Attention["k"]
cross_w_k     = Cross_Attention["w_k_weight"]
cross_key     = Cross_Attention["key"]
cross_h_key   = Cross_Attention["h_key"]


cross_v       = Cross_Attention["v"]
cross_w_v     = Cross_Attention["w_v_weight"]
cross_value   = Cross_Attention["value"]
cross_h_value = Cross_Attention["h_value"]      

cross_QK                                 = Cross_Attention["QK"]
cross_attention_scores_partial_original  = Cross_Attention["attention_scores_partial_original"]
cross_attention_scores_partial           = Cross_Attention["attention_scores_partial"]
cross_attention_scores_masked            = Cross_Attention["attention_scores_masked"]
cross_attention_scores                   = Cross_Attention["attention_scores"]
cross_attention_scores_dropout           = Cross_Attention["attention_scores_dropout"]

cross_AV = Cross_Attention["attention_values"]

cross_AV_cont  = Cross_Attention["AV_cont"]
cross_w_o      = Cross_Attention["w_o_weight"]
cross_output   = Cross_Attention["output"]



#DECODER MULTIHEAD ATTENTION  RESIDUAL ADD 
Residual_Multihead_Attention = load("ResidualConnection_3.json")
dropout_p_2       = Residual_Multihead_Attention["dropout_p"]
input_x_2         = Residual_Multihead_Attention["input_x"]
norm_x_2          = Residual_Multihead_Attention["norm_x"]
output_2          = Residual_Multihead_Attention["layer_output"]
output_dropout_2  = Residual_Multihead_Attention["layer_output_dropout"]
residual_output_2 = Residual_Multihead_Attention["add"]



#DECODER NEURAL NETWORK NORMALIZATION
Normalization_Neural_Network = load("LayerNormalization_5.json")
eps_3        = Normalization_Neural_Network["eps"]
alpha_3      = Normalization_Neural_Network["alpha"]
bias_3       = Normalization_Neural_Network["bias"]
mean_3       = Normalization_Neural_Network["mean"]
std_3        = Normalization_Neural_Network["std"]
normalized_3 = Normalization_Neural_Network["normalization"]


#DECODER NEURAL NETWORK 
Neural_Network = load("FeedForwardBlock_1.json")
nn_weights_1 = Neural_Network["linear_1_weight"]
nn_bias_1 = Neural_Network["linear_1_bias"]
nn_first_out = Neural_Network["first_out"]
nn_first_activation = Neural_Network["first_activation"]
nn_first_dropout = Neural_Network["first_dropout"]
nn_weights_2 = Neural_Network["linear_2_weight"]
nn_bias_2 = Neural_Network["linear_2_bias"]
nn_output = Neural_Network["result"]



#DECODER NEURAL NETWORK  RESIDUAL ADD 
Residual_Neural_Network = load("ResidualConnection_4.json")
dropout_p_3 = Residual_Neural_Network["dropout_p"]
input_x_3   = Residual_Neural_Network["input_x"]
norm_x_3    = Residual_Neural_Network["norm_x"]
output_3    = Residual_Neural_Network["layer_output"]
output_dropout_3  = Residual_Neural_Network["layer_output_dropout"]
residual_output_3 = Residual_Neural_Network["add"]

#DECODER OUTPUT NORMALIZATION
Normalization_Neural_Network = load("LayerNormalization_6.json")
eps_4        = Normalization_Neural_Network["eps"]
alpha_4      = Normalization_Neural_Network["alpha"]
bias_4       = Normalization_Neural_Network["bias"]
mean_4       = Normalization_Neural_Network["mean"]
std_4        = Normalization_Neural_Network["std"]
normalized_4 = Normalization_Neural_Network["normalization"]


#DECODER PORJECTION LAYER
Projection_Layer = load("ProjectionLayer_0.json")
proj_vocab_size = Projection_Layer["vocab_size"]
proj_weights = Projection_Layer["weight"]
proj_bias = Projection_Layer["bias"]
proj_input = Projection_Layer["input_x"]
proj_output = Projection_Layer["output"]

