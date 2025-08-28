# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = False,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 创建一个可训练的参数，维度为dim，初始值为1

    def _norm(self, x):
        # mean：对 x^2 的最后一个维度的值求均值，保持张量维度；再加上eps
        # rsqrt：求平方根的倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) 


    def forward(self, x):
        # float()：转为float类型
        # type_as(x)：将张量转换为与x相同的数据类型
        return self.weight * self._norm(x.float()).type_as(x) 


# 计算旋转位置编码的余弦和正弦矩阵
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    # 生成不同维度的角频率（基础频率）
    # 公式：1 / (theta ** (2i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成不同位置的索引
    t = torch.arange(end, device=freqs.device)
    # 生成不同位置的索引，与基础频率进行外积
    # 公式：t * freqs
    freqs = torch.outer(t, freqs).float()
    # 生成形状为 [end, dim] 的余弦和正弦矩阵，用于后续向量旋转
    # cat(): 在指定维度上拼接张量
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

# 应用旋转位置编码
# 二维旋转矩阵是[[cos, -sin], [sin, cos]]，对于向量[a, b]，旋转后为[cos*a-sin*b, sin*a+cos*b]
# 然后把a和b有关的部分提取出来，得到[cos*a, sin*a]和[cos*b, -sin*b]
# 将向量[a, b]视为复数a+ib，旋转θ即为两者相乘(a​+ib​)(cosθ​+isinθ​)=(cos*a-sin*b)+i(sin*a+cos*b)
# 同类项合并可得 a=cos*a-sin*b, b=sin*a+cos*b
# 而，与二维旋转矩阵相同
# 所以可以将二维旋转矩阵视为复数旋转矩阵，复数旋转矩阵的旋转操作相当于交换实部虚部并取反虚部
# 位置信息蕴含在旋转角度中
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 将输入向量的后一半维度取负，并与前一半维度拼接，实现向量旋转
    def rotate_half(x):
        # 因为q和k的形状通常是[batch, num_heads, seq_len, head_dim]，
        # head_dim 是单个注意力头的特征维度，所以要对最后一维进行旋转，对前三维旋转没有意义
        # [...,num:]等价于[:,:,:,num:]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # 通过复数乘法实现向量旋转
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # 将输入张量的dim_2维重复n_rep次，实现向量重复，是注释方法的高效替代实现
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads # 查询头数
        self.n_local_kv_heads = self.num_key_value_heads # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # q头数与k,v头数的比例
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        # x: (bsz, seq_len, hidden_size)
        bsz, seq_len, _ = x.shape 
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len]) # 动态匹配对应长度的位置编码

        # kv_cache实现
        # 如果使用kv cache的话，xk和xv的seq_len长度就会是当前seq_len+last_seq_len，后面手动实现sdpa时scores的计算好像会出问题
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1) # dim=1 是因为dim_1是seq_len维度，需要按照seq_len维度拼接
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None # 根据use_cache参数决定是否保存past_kv

        # k,v头数与查询头数相等
        xq, xk, xv = (
            # 多头注意力计算公式要求头维度 (num_heads) 位于序列长度维度 (seq_len) 之前，以便并行计算多个头的注意力分数。
            # (batch_size, seq_len, n_local_heads, head_dim) -> (batch_size, n_local_heads, seq_len, head_dim)
            xq.transpose(1, 2), 
            # 扩展k,v头数，使之与q头数匹配
            # (batch_size, seq_len, n_local_kv_heads, head_dim) -> (batch_size, n_local_kv_heads*n_rep, seq_len, head_dim)
            repeat_kv(xk, self.n_rep).transpose(1, 2), 
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        # 使用flash attention
        if self.flash and seq_len != 1:
            # 训练时使用dropout防止过拟合
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            # 使用填充掩码
            if attention_mask is not None:
                # 将注意力掩码扩展到多头注意力分数矩阵的维度 
                # seq_len就是current_seq_len，last_seq_len是kv cache的长度，若没有就是0
                # (bsz, seq) -> (bsz, 1, 1, seq) -> (bsz, n_local_heads, seq_len, seq_len)
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                # 将注意力掩码转换为布尔类型，以便在后续的缩放点积注意力计算中使用
                attn_mask = attn_mask.bool() if attention_mask is not None else None
            # QK^T/sqrt(d_k),加了缩放因子sqrt(d_k)的注意力就是缩放点积注意力。
            # 该方法是pytorch2.0中的官方实现：is_causal=True表示使用因果掩码，在分数的上三角部分设置为-inf，确保模型不能看到未来的信息
            # attn_mask是一个可选的注意力掩码，True代表要屏蔽的位置，False代表要保留的位置
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else: # 手动实现注意力分数 QK^T/sqrt(d_k)
            # @ 规定乘以最后两个维度，即矩阵乘法，前面的维度当做批次，进行并行计算
            # scores:(bsz, n_local_heads, seq_len, seq_len)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 进行因果mask，确保模型只能看到当前及之前的token
            # triu() 取输入矩阵的上三角部分，diagonal=1 从主对角线(左上->右下)往上一斜线开始保留矩阵元素
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device), # 这里好像没考虑使用kv cache的情况
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask
            # 使用填充掩码
            # 填充mask会直接告诉模型每个位置是否需要参与注意力计算
            if attention_mask is not None:
                # (bsz, seq_len) -> (bsz, 1, 1, seq_len) -> (bsz, n_local_heads, seq_len, seq_len)
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # unsqueeze只是形状变了，并没有改变值
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            # softmax计算注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # (bsz, n_local_heads, seq_len, head_dim)
            output = scores @ xv # 最后乘Q_V矩阵
        # (bsz, n_local_heads, seq_len, head_dim) -> (bsz, seq_len, n_local_heads * head_dim)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # (bsz, seq_len, n_local_heads * head_dim) = (bsz, seq_len, hidden_size)
        output = self.resid_dropout(self.o_proj(output)) # 残差连接前的dropout
        return output, past_kv

# 专家网络：独立的子网络，通常是一个前馈神经网络FFN或者MLP
# 门控GLU:相比较传统的升降维FFN，多了一个可学习的门控层gate_proj，在通过激活函数后可以决定升维后的信息通过的概率
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok # 每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts # 可路由的专家总数

        self.scoring_func = config.scoring_func # 打分函数 (当前仅支持softmax)
        self.alpha = config.aux_loss_alpha # 辅助损失的权重系数
        self.seq_aux = config.seq_aux # 辅助损失是否按序列级别计算

        self.norm_topk_prob = config.norm_topk_prob # 是否对 top-k 权重归一化
        self.gating_dim = config.hidden_size # 输入隐藏层的维度
        # 门控权重矩阵 [n_experts, hidden_size]
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    # 在层创建时初始化参数;如果不初始化，参数是全零或者全相同的，网络就没法学习
    def reset_parameters(self) -> None:
        import torch.nn.init as init
        # kaiming初始化，专为ReLU激活函数设计;参数a为了兼容 LeakyReLU 这类带负斜率的激活函数
        # 如果是普通 ReLU，就设 a=0。如果是 LeakyReLU(负斜率 = 1/√5)，那就用 a=√5
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # (bsz, seq_len, hidden_size)
        bsz, seq_len, h = hidden_states.shape
        # (bsz*seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, h)
        # 进行矩阵乘法 hidden_states@self.weight^T,不使用偏置项;
        # (bsz*seq_len, hidden_size) * (hidden_size,n_routed_experts)->(bsz*seq_len,n_routed_experts)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            # (bsz*seq_len,n_routed_experts)
            scores = logits.softmax(dim=-1) # scores是每个token(bsz*seq_len)对专家(n_routed_experts)的概率分布
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        # 在指定维度上选择评分的前k个元素：选择评分最高的前K个专家，即经过softmax以后评分高的；
        # weight和idx形状相同，与scores只在最后一个维度上不同，都是(bsz*seq_len,top_k)
        # weight的值是scores的值，idx的值是scores的索引，例如scores=[[0.1,0.2,0.3,0.4],[0.5,0.2,0.3,0.4]]
        # 那么topk_weight=[[0.4,0.3],[0.5,0.4]],代表专家概率，topk_idx=[[3,2],[0,3]],代表专家索引号
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # 如果选择多个专家可能需要权重归一化，保证其权重和为1
        if self.top_k > 1 and self.norm_topk_prob:
            # keepdim:保持求和后维度数量不变；1e-20:添加一个极小值避免分母为零，例如权重全为零的情况
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # MoE存在部分专家经常被选择，而有的几乎不用，导致负载不均衡。因此需要引入辅助损失。
        # 如果是训练状态并且alpha>0
        if self.training and self.alpha > 0.0:
            # (bsz*seq_len,n_routed_experts)
            scores_for_aux = scores
            aux_topk = self.top_k
            # (bsz, seq_len*top_k)
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) # 记录每个token的专家选择结果
            # 辅助损失是否按照seq_len级别运算，每个样本(batch)独立算一个专家负载均衡情况，再对 batch 平均
            # 确保每个样本内部 token 的专家使用是均衡的，避免出现某个样本内所有 token 都被路由到同一个专家
            if self.seq_aux:
                # (bsz, seq_len, n_routed_experts)
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # (bsz, n_routed_experts),用来记录每个batch内token分配到专家的次数，例如[[1,0,1,0]]就代表第一个batch内，前四个token分配到的专家次数
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # scatter_add_:分布式加法操作，统计每个专家被选择的次数。
                #              在 dim=1（专家维度）上，根据 topk_idx_for_aux_loss（每个 token 选出的专家编号），把 1 累加到对应专家
                # div_:归一化计数，相当于计算实际使用次数 / 理论平均次数。
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # 计算每个 batch 的 token 在专家上的分数取平均,然后乘每个专家的使用次数ce，再对专家求和，再取平均，并做一个系数，从而得到aux_loss
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else: # 在所有batch下对专家的计算，可能会出现某些专家一直使用，某些一直没有使用，但是整体平均下来是均衡的
                # 把所有token选择的专家转为one-hot向量 (bsz*seq_len*top_k, n_routed_experts)
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 对所有 token 的选择结果取平均 → 得到每个专家被选中的概率，表示实际专家的使用分布
                ce = mask_ce.float().mean(0)
                # 表示 gating 预测的专家分布
                Pi = scores_for_aux.mean(0)
                # 乘上专家数量，得到实际使用的专家分布
                fi = ce * self.n_routed_experts
                # 两个分布(向量)的点积越大，代表着两个分布越相似
                aux_loss = (Pi * fi).sum() * self.alpha
        else: # 推理状态下不使用辅助损失
            aux_loss = 0
        # (bsz*seq_len,top_k),(bsz*seq_len,top_k)
        return topk_idx, topk_weight, aux_loss

# 混合专家网络(MOE)
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        # 保留原始输入、形状
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制对每个 token 选择前 k 个专家，并输出专家的索引、选在该专家的概率(score)，平衡损失
        # (bsz*seq_len,top_k) bsz*seq_len代表token数，top_k代表专家的索引/得分，例如(3,2)，那么topk_idx[[2,0],[2,0],[2,0]]的意思是token0用了2,0两个专家，token1...以此类推
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # (bsz*seq_len, hidden)
        x = x.view(-1, x.shape[-1])
        # 把专家索引拉平成一维数组，方便后面按专家分组
        # (bsz*seq_len*top_k) [[2,0],[2,0],[2,0]] -> [2,0,2,0,2,0]
        flat_topk_idx = topk_idx.view(-1)
        # 如果是训练阶段
        if self.training:
            # 复制输入，每个token在第dim维度上重复num_experts_per_tok次，例如
            # [[1,2],[3,4]]->在第0维重复两次->[[1,2],[1,2],[3,4],[3,4]]
            # (bsz*seq_len, hidden)->(bsz*seq_len*num_experts_per_tok, hidden)
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 创建一个与输入x形状相同的空张量y，用于存储专家处理后的结果
            # (bsz*seq_len*num_experts_per_tok, hidden)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                # 遍历专家，按照索引flat_topk_idx找到对应专家处理的token，将其放到y中对应位置存储
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            # 把多个专家的结果按照权重加权平均;sum会直接在形状上压缩掉一维
            # (bsz*seq_len,top_k,hidden) * (bsz*seq_len,top_k,1) -> (bsz*seq_len,hidden)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # (bsz,seq_len,hidden)
            y = y.view(*orig_shape)
        else: # 推理阶段
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            # share expert是一个兜底性的专家，保证所有 token 至少能走一条稳定路径
            for expert in self.shared_experts:
                # 当经过route expert以后，再加上shared_experts的输出，强制把通用专家的输出融合进去
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # x:(bsz*seq_len, hidden) flat_expert_indices:(bsz*seq_len*top_k) flat_expert_weights:(bsz*seq_len*top_k,1)
        # 用于累加专家对token的贡献 (bsz*seq_len, hidden)
        expert_cache = torch.zeros_like(x)
        # 对展平的专家索引进行排序，得到排序后的索引所对应的数组下标 (bsz*seq_len*top_k)
        idxs = flat_expert_indices.argsort()
        # bincount() 得到长度为 n_routed_experts 的数组，统计每个专家在 flat_expert_indices 中出现的次数
        # 例如 torch.bincount(tensor([0, 0, 1, 2, 1, 3, 0, 2])) -> tensor([3, 2, 2, 1])
        # cumsum() 把计数转换为“累积结束索引”数组，便于用 idxs 做切片。转cpu再转numpy,使用numpy的cumsum处理(其实torch也有cumsum)
        # 例如 counts [3,5,2] => cumsum [3,8,10]，表示专家0条目在 idxs[:3]，专家1在 idxs[3:8]，专家2在 idxs[8:10]
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 由于idxs是展平后的专家索引，即(bsz*seq_len,top_k)->(bsz*seq_len*top_k),那么整除top_k就能还原回专家所对应的token
        # 例如topk_idx是[[2,0],[2,0],[2,0]]，代表有3个token，且每个token都是2号与0号专家
        # 展平得[2,0,2,0,2,0],排序得idxs[1,3,5,4,2,0]，
        # 整除top_k(专家数)得 token_idxs[0,1,2,2,1,0],
        # 即idxs=1是token0的专家，idxs=3是token1的专家，idxs=5是token2的专家，idxs=4是token2的专家
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert): # 按照专家分批处理
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1] # i ≠ 0, start_idx = 上一个专家的结束索引
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx] # 取出x中索引为exp_token_idx的token，交给指定的专家去计算
            # 对token进行专家计算(FFN)
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 从排序后的idxs中取出对应位置的专家索引，然后按照索引取出对专家的打分，在原地与expert_out进行逐元素相乘(.mul_)，并保存在expert_out中
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将经过专家计算的expert_out值，在dim维度上，加到expert_cache的指定位置上
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
