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


#DECODER RESIDUAL ADD 
Residual_Multihead_Attention = load("ResidualConnection_2.json")
dropout_p_1       = Residual_Multihead_Attention["dropout_p"]
input_x_1         = Residual_Multihead_Attention["input_x"]
norm_x_1          = Residual_Multihead_Attention["norm_x"]
output_1          = Residual_Multihead_Attention["layer_output"]
output_dropout_1  = Residual_Multihead_Attention["layer_output_dropout"]
residual_output_1 = Residual_Multihead_Attention["add"]


"""
#ENCODER NEURAL NETWORK NORMALIZATION
Normalization_Neural_Network = load("LayerNormalization_1.json")
eps_2        = Normalization_Neural_Network["eps"]
alpha_2      = Normalization_Neural_Network["alpha"]
bias_2       = Normalization_Neural_Network["bias"]
mean_2       = Normalization_Neural_Network["mean"]
std_2        = Normalization_Neural_Network["std"]
normalized_2 = Normalization_Neural_Network["normalization"]


#ENCODER NEURAL NETWORK 
Neural_Network = load("FeedForwardBlock_0.json")
nn_weights_1 = Neural_Network["linear_1_weight"]
nn_bias_1 = Neural_Network["linear_1_bias"]
nn_first_out = Neural_Network["first_out"]
nn_first_activation = Neural_Network["first_activation"]
nn_first_dropout = Neural_Network["first_dropout"]
nn_weights_2 = Neural_Network["linear_2_weight"]
nn_bias_2 = Neural_Network["linear_2_bias"]
nn_output = Neural_Network["result"]



#ENCODER NEURAL NETWORK  RESIDUAL ADD 
Residual_Neural_Network = load("ResidualConnection_1.json")
dropout_p_2 = Residual_Neural_Network["dropout_p"]
input_x_2   = Residual_Neural_Network["input_x"]
norm_x_2    = Residual_Neural_Network["norm_x"]
output_2    = Residual_Neural_Network["layer_output"]
output_dropout_2  = Residual_Neural_Network["layer_output_dropout"]
residual_output_2 = Residual_Neural_Network["add"]

"""