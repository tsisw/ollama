package qwen3vl

import (
	"cmp"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
)

type TextOptions struct {
	hiddenSize,
	numHeads,
	numKVHeads,
	keyLength,
	valueLength int

	eps,
	ropeBase,
	ropeScale float32
	mropeSections []int

	numExperts, numExpertsUsed int
	normTopKProb               bool
}

func (o TextOptions) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

func (o TextOptions) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	return fast.RoPE(ctx, states, positions, o.headDim(), o.ropeBase, 1./o.ropeScale,
		rope.WithTypeMRoPE(),
		rope.WithMRoPESections(o.mropeSections),
	)
}

type TextAttention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *TextAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, batchSize)
	key = key.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)
	value = value.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)

	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(opts.headDim())), cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return sa.Output.Forward(ctx, attention)
}

type TextMLP interface {
	Forward(ml.Context, ml.Tensor, *TextOptions) ml.Tensor
}

type sparse struct {
	Router *nn.Linear      `gguf:"ffn_gate_inp"`
	Gate   *nn.LinearBatch `gguf:"ffn_gate_exps"`
	Up     *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down   *nn.LinearBatch `gguf:"ffn_down_exps"`
}

func (mlp *sparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)
	routerLogits := mlp.Router.Forward(ctx, hiddenStates)

	routingWeights := routerLogits.Softmax(ctx)
	selectedExperts := routingWeights.TopK(ctx, opts.numExpertsUsed)
	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, selectedExperts)
	if opts.normTopKProb {
		routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
		routingWeights = routingWeights.Div(ctx, routingWeights.SumRows(ctx))
		routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
	}

	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates, selectedExperts).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates, selectedExperts))

	experts := mlp.Down.Forward(ctx, hiddenStates, selectedExperts)
	experts = experts.Mul(ctx, routingWeights)

	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	return nextStates
}

type dense struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *dense) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ *TextOptions) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).
		SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type TextLayer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	*TextAttention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	TextMLP
}

func (d *TextLayer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = d.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.TextAttention.Forward(ctx, hiddenStates, positions, cache, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = d.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.TextMLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

type TextModel struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	Layers []TextLayer `gguf:"blk"`

	Options *TextOptions
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.Options.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

var _ model.Model = (*Model)(nil)

func newTextModel(c fs.Config) *TextModel {
	layers := make([]TextLayer, c.Uint("block_count"))
	for i := range layers {
		if strings.HasSuffix(c.String("general.architecture"), "moe") {
			layers[i].TextMLP = &sparse{}
		} else {
			layers[i].TextMLP = &dense{}
		}
	}

	m := TextModel{
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		Layers: layers,
		Options: &TextOptions{
			hiddenSize:     int(c.Uint("embedding_length")),
			numHeads:       int(c.Uint("attention.head_count")),
			numKVHeads:     int(c.Uint("attention.head_count_kv")),
			keyLength:      int(c.Uint("attention.key_length")),
			valueLength:    int(c.Uint("attention.value_length")),
			eps:            c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:       c.Float("rope.freq_base"),
			ropeScale:      c.Float("rope.scaling.factor", 1),
			numExperts:     int(c.Uint("expert_count")),
			numExpertsUsed: int(c.Uint("expert_used_count")),
			normTopKProb:   c.Bool("norm_top_k_prob", true),
			mropeSections: slices.Collect(func(yield func(int) bool) {
				for _, section := range c.Ints("mrope_sections", []int32{24, 20, 20}) {
					if !yield(int(section)) {
						return
					}
				}
			}),
		},
	}

	return &m
}
