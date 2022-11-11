# Reversing engineering attention-only transformers

## One-layer transformers

I recently became interested in Anthropic's [work](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) on interpretability. Particularly the sudden emergence of induction heads when you go from 1 to 2-layer transformers. 

## Task

Predict a sequence of 24 letters where the sequence is made up of 4 different blocks of 6 letters. 
Example: ABCDEFABCDEFABCDEFABCDEF

So induction heads\* should be able to fully solve this task. However, in 1-layer transformers the key matrix is a linear function of the input tokens hence the model can't find the last occurrence of the current token.

##  Model

I used a simplified version of the decoder-only transformer architecture.

Simplifications:
- No layer normalization
- No MLP blocks
- No bias
- Sinusoidal positional encoding is added to Q and K before the attention is computed(to avoid adding it to the residual stream)

Hyperparameters:
- 1 layer
- 4 heads
- 24 token sequence
- Inner dimension: 32
- Learning rate: 0.001
- Weight decay: 1.0
- Steps: 10000
- Batch size: 256
- Total params: 26x32 + 4x32^2 + 26x32 = 5760 

The model was trained on randomly generated data.

## Results

### 1-layer model without smeared keys

Below are some examples of model completions using greedy decoding.

Prompt: ABCDEF<br />
Completion: ECECECECECECECECEC<br />
Correct completion: ABCDEFABCDEFABCDEF<br />
<br />
Prompt: ABCDEFABCDEF<br />
Completion: CFCFCFCFCFCF<br />
Correct completion: ABCDEFABCDEF<br />
<br />
Prompt: XYZRSTXYZRST<br />
Completion: TSTSZSTSTSZST<br />
Correct completion: XYZRSTXYZRST<br />
<br />
Out of distribution example.<br />
<br />
Prompt: ABCDEFXYZRST<br />
Completion: FEFEFEFEFEFE<br />
Correct completion: N/A<br />
<br />
Prompt: QWERTYXYZRST<br />
Completion: YTYTYTYTYTYT<br />
Correct completion: N/A<br />

## Observations
The model:
 - Fails at this task
 - Never repeats the same token twice in a row
 - Copies seemingly random tokens from the prompt even in out of distribution examples

# Analysis

As discussed above, the one-layer model fails because it can't use K-composition to find the last occurrence of the current token. 

Let's try looking at the eigenvalues of the OV circuits in heads 1, 2, 3 and 4. Positive eigenvalues indicate copying behaviour. Recall that k is an eigenvalue of M if and only if Mv = kv for some vector v. If k > 0 this implies that the probability of the token(or set of tokens) has increased. Note tokens are initially encoded as one-hot vectors\**.

To measure how positive the eigenvalues we will look at the sum k/|k|<br />
Head 1: 0.6860927<br />
Head 2: 0.85695416<br />
Head 3: 0.98481447<br />
Head 4: 0.5779195<br />

This indicates that head 3 is almost entirely dedicated to copying behaviour and provides some evidence for the third observation. The other heads also have some copying behaviour but it is less convincing.
Unfortunately, the attention mechanism is non-linear so we'll use observational evidence. Let's take a look at the average attention weights for heads 1, 2, 3 and 4 in a random sample of data.

Head 1:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head1.png)<br />
Head 2:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head2.png)<br />
Head 3:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head3.png)<br />
Head 4:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head4.png)<br />

As you can see, most of the attention is on the current token and a couple of tokens before it. The attention scores are also broadly similar across heads.

Let's approximate and say that the attention matrices are the same for all heads. So we can factor out the attention weights and consider the sum of the OV circuits.

Then let's look at the eigenvalues of the sum of the OV circuits.

Sum of OV circuits: 0.9997909<br />

This is very close to 1.0 so the model is just copying the past couple of tokens. This would explain the third observation.

What about the "direct path" (the embedding matrix followed by the unembedding matrix)? Usually, this part of the network learns simple bigram statistics (like "Obama" follows "Barack"). Let's the sum k/|k| of the eigenvalues of the direct path.<br />

Direct path: -0.9995849<br />

This is very close to -1.0. This tells that for a given input token, the direct path reduces the probability of that token. So if "A" is the current token direct path will ensure that "A" is not predicted as the completion. This would explain the second observation. This makes sense because the probability of "A" following "A" is 1/26. The function of the direct path, in this case, is very different from in Anthropic's work.

### 1-layer model with smeared keys

What are smeared keys?<br />
Essentially each key becomes the weighted average of the current key and the previous one. The exact weighting is learned during training. 

How does this help?

This wasn't obvious to me at first. But let's think of the query matrix as a set of padlocks. Where q\*k("\*" is the dot product) is how well the key fits the padlock. 
Consider the sequence ABCDEFABCDEFABCDEFABCDEF. In this example, the "B" key would look a bit like the "A" key. Because the query is a linear function of "A" it follows that all the "B" positions can have a high attention score. So the model can find the last occurrence of the current token.

### Examples

Prompt: ABCDEF<br />
Completion: EFEFEFEFEFEFEFEFEF<br />
Correct completion: ABCDEFABCDEFABCDEF<br />
<br />
Prompt: ABCDEFABCDEF<br />
Completion: ABCDEFABCDEF<br />
Correct completion: ABCDEFABCDEF<br />
<br />
Prompt: XYZRSTXYZRST<br />
Completion: XYZRSTXYZRST<br />
Correct completion: XYZRSTXYZRST<br />
<br />
Out of distribution example.<br />
<br />
Prompt: ABCDEFXYZRST<br />
Completion: FXYZRSTFXYZR<br />
Correct completion: N/A<br />
<br />
Prompt: QWERTYXYZRST<br />
Completion: YXYXYXYXYXYX<br />
Correct completion: N/A<br />

## Observations
The model:
 - Correctly completes the sequences when the prompt is 12 letters long 
 - Never repeats the same token twice in a row even in out of distribution examples

# Analysis

Again, let's look at the sign of the eigenvalues of the OV circuits and the direct path(Using k/|k| metric).<br />

Direct path: -0.9998577<br />
Head 1: -0.71228284<br />
Head 2: 1.0<br />
Head 3: 1.0<br />
Head 4: -0.856813<br />


The eigenvalues of the direct path are almost all negative explaining the first observation in the same way as before. Heads 2 and 3 seem to be copying heads whereas 1 and 4 looks like "anti-copying" heads. 

Below are the attention patterns for heads 1 and 2(3 and 4 are similar).<br />

Head 1:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_smeared_head1.png)

Head 2:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_smeared_head2.png)

<br />
From the second image, we can see that heads 2 and 3 attend to the tokens 5 back from the current token. Corresponding to the token that should be predicted. Heads 1 and 4 attend to the current letter and the few letters prior. Heads 1 and 4 seem to implement a more advanced version of the direct path algorithm by reducing the probability of the last couple of tokens. These "anti-induction heads" are also different from  Anthropic's work which almost exclusively copies. It is also interesting that 2-layer transformers seem to perform worse than the 1-layer transformer with smeared keys. 

## Two-layer transformers

The hyperparameters and task are the same as before but now we have 26x32+4x32^2+4x32^2+26x32 = 9856 different weights.

## Examples

Below are some examples of model completions using greedy decoding.


Prompt: ABCXYZABCDEF<br />
Completion: ABCDEFABBCEF<br />
Correct completion: ABCDEFABCDEF<br />
<br />

Prompt: AAABBBAAABBB <br />
Completion: ABABABABABAB <br />
Correct completion: AAABBBAAABBB<br />
<br />

Out of distribution example.<br />

Prompt: ABCXYZABCDEF<br />
Completion: ABCXYZABCYYZ<br />
Correct completion: N/A<br />
<br />

## Observations

The model:
 - (Almost) Correctly completes the sequences when the prompt is 12 letters long 
 - Never repeats the same token twice in a row even when this is the correct completion
 - Copies the first instance of a pattern in out-of-distribution examples 

## Analysis

Let's start by looking at first-layer attention patterns.

Head 1:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/two_layer_head_1_layer_1.png)<br />
Head 2:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/two_layer_head_2_layer_1.png)<br />
Head 3:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/two_layer_head_3_layer_1.png)<br />
Head 4:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/two_layer_head_4_layer_1.png)<br />

Let's look again at the eigenvalues of the OV circuits and the direct path(Using k/|k| metric).<br />

Layer 1:

Head 1: -0.9999998-5.327597e-16j <br />
Head 2: -0.94031566-2.6753054e-16j <br />
Head 3: -0.9329939+0j <br />
Head 4: -0.9999997+7.125459e-16j <br />

Above I neglected to mention that the eigenvalues of the OV circuits are complex numbers but because the next letter always consists of one of the previous 6 so the circuit amplifies/reduces past tokens rather than generating unseen tokens. Hence, the imaginary part is always near 0.

So the first layer stops the model from repeating the last ~3 letters(including the current letter). Notice that head 2 always attends to the previous letter(that's important for later).

The second layer is a bit more complicated. Let's look at the attention patterns of the second layer.

Head 1:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/two_layer_head_1_layer_2.png)<br />

(The other heads are similar)<br />

Layer 2:

Head 1: 0.9998253+1.1311128e-16j <br />
Head 2: 0.9999484+0j <br />
Head 3: 0.99989885+4.20817e-13j <br />
Head 4: 0.99702746+1.5782087e-10j <br />

So these heads are dedicated to copying.

The nxn attentions look rather noisy but some deductions can be made. Firstly, most heads attend to the current letter and the few letters prior. But the tokens being copied are not the original ones but the outputs of layer 1 which have a reduced probability of being the last 3 tokens of that letter. This in effect means that the last ~4-6 letters will not be repeated. But look at the faint diagonal lines from positions 0-5 these tokens are unaffected by the anti-copying of the first layer and are "weakly" copied. This is why the model copies the first instance of a pattern in out-of-distribution examples.

How does the model generate this diagonal pattern?

Both the query and key matrix can be composed with heads in layer 1 to generate more interesting attention patterns. This is called K-composition and Q-composition respectively. 

What happens if we disable K-composition?

Layer 2 Head 1 without K-composition:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/layer2_head_1_no_K_composition.png)

Now the faint diagonal lines are gone! And instead, it copies tokens unaffected by the first layers or just copies all the tokens.

Let's look at the part of the circuit that takes heads from layer 1 and composes them with the query and key matrix.

Specifically, the A * (W_E^T)(W_QK)(W_OV_1_2)(W_E) part of the circuit. This term is summed over all heads in layer 1. ("*" is elementwise multiplication)

Where:
- A is the attention pattern of head 2 from layer 1
- W_QK is the product of the query and key matrix in the second layer heads
- W_OV_1_2 is the OV circuit of head 2 in layer 1
- W_E is the embedding matrix

Let's look at the eigenvalues of this term for all heads in layer 2 for each head in layer 1.

Layer 1 head 1<br />
Head 1 layer 2:  (0.89388597-2.2985782e-09j) <br />
Head 2 layer 2:  (-0.6828638+0j) <br />
Head 3 layer 2:  (-0.92538565+3.692424e-11j) <br />
Head 4 layer 2:  (0.70200497+6.051526e-16j) <br />

Layer 1 head 2<br />
Head 1 layer 2:  (0.99992883+6.803108e-15j) <br />
Head 2 layer 2:  (0.999999-4.2807158e-14j)<br />
Head 3 layer 2:  (0.99996066+0j) <br />
Head 4 layer 2:  (0.9999983-3.4481166e-13j)<br />

Layer 1 head 3<br />
Head 1 layer 2:  (-0.3635076+0j)<br />
Head 2 layer 2:  (-0.63910097+0j)<br />
Head 3 layer 2:  (0.9450826-2.4281412e-09j)<br />
Head 4 layer 2:  (-0.4290508+5.9449112e-09j)<br />

Layer 1 head 4<br />
Head 1 layer 2:  (-0.9442232+0j)<br />
Head 2 layer 2:  (0.91947246-5.316366e-09j)<br />
Head 3 layer 2:  (0.2616711-2.5381839e-08j)<br />
Head 4 layer 2:  (0.32606995+0j)<br />

All of these compositions between heads seem somewhat random except for the second head in layer 1 which gives overwhelmingly positive eigenvalues. Remember that the attention pattern of head 2 attends to the token *before* the current one. This means all the tokens are copied then only the letter before the current one is retained.

The effect is that the keys are shifted forward enabling the current token to find previous copies of the letter before it. This explains why removing K-composition removes the diagonal lines.

Finally, we note that the second observation is explained by the direct path having negative eigenvalues(-0.9995 using the metric k/|k|).

# Conclusion

Algorithm 1) One-layer transformer
- Copy the last ~10 letters but favour letters close to the current(Heads)
- Don't repeat the same letter twice in a row (Direct path)

Algorithm 2) One-layer transformer with smeared keys
- Copy the letter 5 back from the current letter (Heads 2 and 3)
- Don't repeat any of the ~3 previous letters(including the current letter) (Heads 1, 4 and the direct path)

Algorithm 3) Two-layer transformer
- Don't repeat the last ~4-6 letters
    - Don't repeat the same letter twice in a row (Direct path)
    - Don't repeat the last ~3 letters (Heads in layer 1)
    - Copy the last ~3 logits (Heads in layer 2) because the logits come from layer 1 this stop the last ~4-6 letters from being repeated
- "Weakly" copy the first occurrence of the next letter (Heads in layer 2 using K-composition with head 2 in layer 1)

The model was trained and tested on just (256+64)\*10000/26^6 = 1.04% of all possible sequences. Yet we can be reasonably confident the (admittedly toy) network will behave as expected on in and out of distribution examples. (Speculation) My guess is that "anti-induction heads" emerge due to the ratio of heads to possible tokens being 4 to 6. Hence, the model can meaningfully improve by eliminating bad choices. In "trained on the internet" models, the ratio of heads to tokens is much smaller so eliminating bad choices is not very important. Also of note is that 1-layer smeared key models have 2 copying and 2 anti-copying heads whereas 2-layer models have 1-layer for copying and another for anti-copying. 


\* "Induction heads is a circuit whose function is to look back over the sequence for previous instances of the current token (call it A), find the token that came after it last time (call it B), and then predict that the same completion will occur again." (Quote from https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

\** Why are all the matrices multiplied in the wrong order? Because the weights are transposed in the code.
Note that (AB)^T = (B^T)(A^T) and eigenvalues do not change under transposition.






