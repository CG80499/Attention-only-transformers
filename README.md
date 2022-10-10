# Attempting to reverse engineer one-layer transformers

I recently became interested became in Anthropic's [work](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) on interpretability. Particularly the sudden emergence of induction heads when you go from 1 to 2-layer transformers. 

## Task

Predict a sequence of 24 letters where the sequence is made up of 4 different blocks of 6 letters. 
Example: ABCDEFABCDEFABCDEFABCDEF

As a reminder, induction heads find the last occurrence of the current token and then predict that the pattern occurs again. So they should be able to fully solve this task. However, in 1-layer transformers the key matrix is linearly dependent on the input matrix hence the model can't find the last occurrence of the current token.

##  Model

I used a simplified version of the decoder-only transformer architecture.

Simplifications:
- No layer normalization
- No MLP blocks
- Sinusoidal positional encoding is added to Q and K before the attention is computed(to avoid adding it to the residual stream)

Hyperparameters:
- 1-layer
- 4-head
- 24-token sequence
- Inner dimension: 32
- Learning rate: 0.001
- Weight decay: 1.0
- Steps: 10000
- Batch size: 256
- Total params: 26x32 + 4x32^2 + 26x32 = 5760

The model was trained on randomly generated data.

## Results

### 1-layer model without smeared keys

Below are some examples of model completions. 

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
Correct completion: -<br />
<br />
Prompt: QWERTYXYZRST<br />
Completion: YTYTYTYTYTYT<br />
Correct completion: -<br />

## Observations
The model:
 - Fails at this task
 - Never repeats the same token twice in a row
 - Copies seemingly random tokens from the prompt even in out of distribution examples

# Analysis

As discussed above, the one-layer model fails because it can't use K-composition to find the last occurrence of the current token. 

Let's try looking at the eigenvalues of the OV circuits in heads 1, 2, 3 and 4. Positive eigenvalues indicate copying behaviour. Recall that k is an eigenvalue of M if and only if Mv = kv for some vector v. If k > 0 this implies that the probability of the token(or set of tokens) has increased. Note tokens are initially encoded as one-hot vectors\*.

To measure how positive the eigenvalues we will look at the sum k/|k|<br />
Head 1: 0.6860927<br />
Head 2: 0.85695416<br />
Head 3: 0.98481447<br />
Head 4: 0.5779195<br />

This indicates that head 3 is almost entirely dedicated to copying behaviour and provides some evidence for the second observation. The other heads also have some copying behaviour but it is less convincing.
Unfortunately, the attention mechanism is non-linear so don't have the same mathematical tools at our disposal so the evidence is weaker. Let's take a look at the average attention weights for heads 1, 2, 3 and 4 in a random sample of data.

Head 1:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head1.png)<br />
Head 2:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head2.png)<br />
Head 3:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head3.png)<br />
Head 4:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head4.png)<br />

As you can see, most of the attention is on the current token and a couple of tokens before it. The attention scores are also broadly similar across heads.

Let's do an approximation and say that the attention scores are the same for all heads. So we can factor out the attention weights and consider the sum of the OV circuits.

Then let's look at the eigenvalues of the sum of the OV circuits.

Sum of OV circuits: 0.9997909<br />

This is very close to 1.0(meaning just copying behaviour) so the model is just copying the past couple of tokens. This would explain the third observation.

What about the "direct path" (the embedding matrix followed by the unembedding matrix)? Usually, this part of the network learns simple bigram statistics (like "Obama" follows "Barack"). Let's the sum k/|k| of the eigenvalues of the direct path.<br />

Direct path: -0.9995849<br />

This is very close to -1.0. This tells that for a given input token, the direct path reduces the probability of that token. So if "A" is the current token direct path will ensure that "A" is not predicted as the completion. This would explain the second observation. This makes sense because the probability of "A" following "A" is 1/26. The function of the direct path, in this case, is very different from in Anthropic's work.

\* Why are all the matrices multiplied in the wrong order? Because the weights are transposed in the code.
Note that (AB)^T = (B^T)(A^T) and eigenvalues do not change under transposition.

### 1-layer model with smeared keys

What are smeared keys?<br />
Essentially each key becomes the weighted average of the current key and the previous one. The exact weighting is learned during training. 

How does this help?

This wasn't obvious to me at first. But let's think of the query matrix as a set of padlocks. Where q\*k("*" is the dot product) is how well the key fits the padlock. 
Consider the sequence ABCDEFABCDEFABCDEFABCDEF. In this example, the "B" key would look a bit like the "A" key. Because the query is linearly dependent on "A" it follows that all the "B" positions can have a high attention score. So the model can find the last occurrence of the current token.

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
Correct completion: -<br />
<br />
Prompt: QWERTYXYZRST<br />
Completion: YXYXYXYXYXYX<br />
Correct completion: -<br />

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


The eigenvalues of the direct path are almost all negative explaining the first observation in the same as before. Heads 2 and 3 seem to copy heads whereas 1 and 4 looks like "anti-copying" heads. 

Below are the attention patterns for heads 1 and 2(3 and 4 are similar).<br />

Head 1:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_smeared_head1.png)

Head 2:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_smeared_head2.png)

<br />
From the second image, we can see that heads 2 and 3 attend to the tokens 5 back from the current token. Corresponding to the token that should be predicted. Heads 1 and 4 attend to the current letter and the few letters prior. Heads 1 and 4 seem to implement a more advanced version of the direct path algorithm by reducing the probability of the last couple of tokens. These "anti-induction heads" are also different from  Anthropic's work which almost exclusively copies. It is also interesting that 2-layer transformers seem to perform worse than the 1-layer transformer with smeared keys. 


# Conclusion

Algorithm 1)
- Copy the last ~3 letters (Heads)
- Don't repeat the same letter twice in a row (Direct path)

Algorithm 2)
 - Copy the letter 5 back from the current letter (Heads 2 and 3)
 - Don't repeat any of the ~3 previous letters(including the current letter) (Heads 1, 4 and the direct path)

In future, I would like to deduce more about the attention matrix directly from the weights rather via observational methods. The model was trained and tested on just (256+64)*10000/26^6 = 1.04% of possible sequences. Yet we can be reasonably confident the (admittedly toy) network will behave as expected on in and out of distribution examples. (Speculation) My guess is that anti-induction heads emerge due to the ratio of heads to possible tokens being 4 to 6. Hence, the model can meaningfully improve by eliminating bad choices. In "trained on the internet" models, the ratio of heads to tokens is much smaller so eliminating bad choices is not very important.